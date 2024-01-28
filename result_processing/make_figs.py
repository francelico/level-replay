import argparse
import os
import logging
from collections import defaultdict

import matplotlib.pyplot as plt
import matplotlib as mpl
import numpy as np
import seaborn as sns
from torchvision import utils as vutils
import sys
import re
from rliable import metrics, plot_utils
from rliable import library as rly
from util import *

from result_processing.util import dedupe_legend_labels_and_handles
import result_processing.file_reader as reader
from level_replay.utils import DotDict


sns.set_theme(style="whitegrid",
              palette='colorblind',
              rc={'text.usetex': True},
              # font_scale=1.5,
              )
# Setting usetex=True in sns does not work with the style,
# so we set it separately here
mpl.rc('font', **{'family': 'serif'})
mpl.rc('text', usetex=True)


logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[
        logging.FileHandler("debug.log"),
        logging.StreamHandler(sys.stdout)
    ]
)

ENV_NAMES = 'bigfish,heist,climber,caveflyer,jumper,fruitbot,plunder,coinrun,ninja,leaper,' \
            'maze,miner,dodgeball,starpilot,chaser,bossfight'.split(',')

PLOT_FILE_FORMAT = 'pdf'

# {batch_name: {kw: val}}
PLOTTING = {
    's1-value_l1_s2-random_bf-0.25_l2-1.0_fs-0.5_fe-1.0': {'label': '$S=S^V, P_{S_2}=\mathcal{U}$'},
    's1-value_l1_s2-instance_pred_log_prob_bf-0.25_l2-0.1_fs-0.5_fe-1.0': {'label': '$S=S^V, S_2=S^{\mathrm{MI}}$'},
    's1-value_l1_s2-off_bf-0.0_l2-1.0_fs-0.0_fe-1.0': {'label': '$S=S^V$'},
    's1-random_s2-off_bf-0.0_l2-1.0_fs-0.0_fe-1.0': {'label': '$P_S=\mathcal{U}$'},
    's1-instance_pred_log_prob_s2-off_bf-0.0_l2-1.0_fs-0.0_fe-1.0': {'label': '$S=S^{\mathrm{MI}}$'},
    }
# Set plotting colors from a seaborn palette.
plotting_color_palette = sns.color_palette('colorblind', n_colors=len(PLOTTING))
PLOTTING = {
    method: {'color': plotting_color_palette[i], **kwargs}
    for i, (method, kwargs) in enumerate(PLOTTING.items())
}
# Use the order in the PLOTTING dict for the legend.
METHOD_ORDER = list([d['label'] for k, d in PLOTTING.items()])

LABELS = {method : d['label'] for method, d in PLOTTING.items()}

def parse_args():
    parser = argparse.ArgumentParser(description='Fig')
    parser.add_argument(
        '--debug',
        action="store_true",
        help='Debug mode.')
    parser.add_argument(
        '--base_path',
        type=str,
        default='/home/francelico/dev/PhD/procgen/results/results',
        help='Base path to experiment results directories.')
    parser.add_argument(
        '--output_path',
        type=str,
        default='/home/francelico/dev/PhD/procgen/results/figs',
        help='Base path to store figures and other outputs.')
    parser.add_argument(
        '--plot_rliable',
        action="store_true",
        help='Plot aggregate plots.')
    parser.add_argument(
        '--x_axis',
        type=str,
        default='num_updates',
        help='X axis for plots.')
    parser.add_argument(
        '--ignore_runs',
        type=str,
        default='baserun',
        help='runs in directory to ignore.')
    parser.add_argument(
        '--ignore_extra_pids',
        action="store_true",
        help='Ignore extra pids in directory.')
    parser.add_argument(
        '--seeds',
        type=str,
        default=None,
        help='seeds to pick in each directory.')
    parser.add_argument(
        '--rolling_window',
        type=int,
        default=100,
        help='seeds to pick in each directory.')

    return parser.parse_args()

def adjust_legend(fig, ax, ncol=3):
    # Make room for legend in the figure
    fig.tight_layout(rect=(0, 0, 1, 0.93))

    handles, labels = ax.get_legend_handles_labels()
    labels, handles = dedupe_legend_labels_and_handles(labels, handles)
    # Sort the labels and handles given the METHOD_ORDER
    order = [METHOD_ORDER.index(l) for l in labels]
    labels = [l for l, _ in sorted(zip(labels, order), key=lambda pair: pair[1])]
    handles = [h for h, _ in sorted(zip(handles, order), key=lambda pair: pair[1])]

    ax.legend(handles, labels,
              ncol=ncol,
              bbox_to_anchor=(0.5, 0.95),
              bbox_transform=fig.transFigure,
              loc='center',
              frameon=False)

def plot_aggregates(run_experiment, env_families, checkpoints, save_to, normalise=False,
                    eval_metrics=['test_returns', 'solved_rate'], **kwargs):
    save_dir = os.path.join(save_to, f"rliable")
    os.makedirs(save_dir, exist_ok=True)

    for key in env_families:
        envs = env_families[key]
        logger.info(f"Plotting {key}...")
        new_dir = os.path.join(save_dir, f"{key}")
        os.makedirs(new_dir, exist_ok=True)

        for metric in eval_metrics:
            if normalise:
                n = 'N'
            else:
                n = ''
            fig, ax = run_experiment.plot_eval_interval_estimates(metric, envs=envs, checkpoints=checkpoints, normalise=normalise, **kwargs)
            fig.tight_layout()
            plt.savefig(os.path.join(new_dir, f"{key}_{metric}{n}.{PLOT_FILE_FORMAT}"))
            plt.close(fig)

BASELINE_SCORES = {
    'bigfish': {'random': (3.7, 1.2), 'plr': (10.9, 2.8)},
    'bossfight': {'random': (7.7, 0.4), 'plr': (8.9, 0.4)},
    'caveflyer': {'random': (5.4, 0.8), 'plr': (6.3, 0.5)},
    'chaser': {'random': (5.2, 0.7), 'plr': (6.9, 1.2)},
    'climber': {'random': (5.9, 0.6), 'plr': (6.3, 0.8)},
    'coinrun': {'random': (8.6, 0.4), 'plr': (8.8, 0.5)},
    'dodgeball': {'random': (1.7, 0.2), 'plr': (1.8, 0.5)},
    'fruitbot': {'random': (27.3, 0.9), 'plr': (28.0, 1.3)},
    'heist': {'random': (2.8, 0.9), 'plr': (2.9, 0.5)},
    'jumper': {'random': (5.7, 0.4), 'plr': (5.8, 0.5)},
    'leaper': {'random': (4.2, 1.3), 'plr': (6.8, 1.2)},
    'maze': {'random': (5.5, 0.4), 'plr': (5.5, 0.8)},
    'miner': {'random': (8.7, 0.7), 'plr': (9.6, 0.6)},
    'ninja': {'random': (6.0, 0.4), 'plr': (7.2, 0.4)},
    'plunder': {'random': (5.1, 0.6), 'plr': (8.7, 2.2)},
    'starpilot': {'random': (26.8, 1.5), 'plr': (27.9, 4.4)},
}

def get_final_train_scores_approximation(logr):
    update = logr.num_updates
    all_updates = logr.logs['total_student_grad_updates'].to_numpy()
    ids = np.argsort(np.abs(all_updates - update))[:len(logr.pid_dirs)]
    scores = logr.logs['train_eval:mean_episode_return_mavg'][ids].to_numpy()
    logr.final_train_eval_scores = scores
    logr.final_train_eval_mean = np.mean(scores)
    logr.final_train_eval_std = np.std(scores)

def update_baseline_scores(BASELINE_SCORES, logreaders, baseline='random'):
    for logr in logreaders:
        BASELINE_SCORES[logr.env_name][baseline] = (logr.final_test_eval_mean, logr.final_test_eval_std)
        if baseline == 'random':
            if not hasattr(logr, 'final_train_eval_scores'):
                get_final_train_scores_approximation(logr)
            BASELINE_SCORES[logr.env_name]['random_train_set'] = (logr.final_train_eval_mean, logr.final_train_eval_std)

def mean_normalised_scores(log_readers, baseline='random', mode='test'):

    mean_scores = []
    std_scores = []
    for logr in log_readers:
        if mode == 'test':
            logr.normalised_scores = logr.final_test_eval_scores / BASELINE_SCORES[logr.env_name][baseline][0]
            mean_scores.append(logr.final_test_eval_mean / BASELINE_SCORES[logr.env_name][baseline][0])
            std_scores.append(logr.final_test_eval_std / BASELINE_SCORES[logr.env_name][baseline][0])
        elif mode == 'train':
            update = logr.num_updates
            all_updates = logr.logs['total_student_grad_updates'].to_numpy()
            ids = np.argsort(np.abs(all_updates - update))[:len(logr.pid_dirs)]
            if not hasattr(logr, 'final_train_eval_scores'):
                get_final_train_scores_approximation(logr)
            scores = logr.final_train_eval_scores
            logr.normalised_scores_train = scores / BASELINE_SCORES[logr.env_name][baseline][0]
            logr.train_set_normalised_scores_train = \
                scores / BASELINE_SCORES[logr.env_name][f'{baseline}_train_set'][0]
            mean_scores.append(scores.mean() / BASELINE_SCORES[logr.env_name][baseline][0])
            std_scores.append(scores.std() / BASELINE_SCORES[logr.env_name][baseline][0])
            value_l1 = logr.logs['level_value_loss_mavg'][ids].to_numpy()#TODO: change to
            logr.value_l1 = value_l1
            # final_train_eval:final_train_eval_value_loss when available for more accuracy
            logr.normalised_value_l1 = value_l1 / scores.mean()
    mean_scores = np.array(mean_scores)
    std_scores = np.array(std_scores)
    std = 1/len(std_scores) * np.sqrt((std_scores**2).sum())
    return mean_scores.mean(), std

def compute_stats(log_readers, stat_keys=None, update=-1, **kwargs):

    if stat_keys is None:
        stat_keys = ['instance_pred_accuracy_train', 'instance_pred_prob_train', 'instance_pred_entropy_train',
                    'instance_pred_accuracy', 'instance_pred_prob', 'instance_pred_entropy', 'generalisation_gap',
                    'instance_pred_accuracy_stale', 'instance_pred_prob_stale', 'instance_pred_entropy_stale',
                     'level_value_loss',
                    'mutual_information', 'mutual_information_stale', 'generalisation_bound', 'generalisation_bound_stale',
                     'mutual_information_u', 'mutual_information_u_stale', 'generalisation_bound_u', 'generalisation_bound_u_stale',
                     'overgen_gap', 'overgen_gap_stale']
    if update == -1:
        update = log_readers[0].num_updates
        assert all([logr.num_updates == update for logr in log_readers])
    stats = {}
    for stat in stat_keys:
        mean_stats = []
        if kwargs.get('mavg', False):
            stat_lookup = f"{stat}_mavg"
        else:
            stat_lookup = stat
        for logr in log_readers:
            all_updates = logr.logs['total_student_grad_updates'].to_numpy()
            ids = np.argsort(np.abs(all_updates - update))[:len(logr.pid_dirs)]
            extracted_stats = logr.logs[stat_lookup][ids].to_numpy()
            if update == log_readers[0].num_updates:
                if not hasattr(logr, 'final_stats'):
                    logr.final_stats = DotDict({})
                logr.final_stats[stat] = extracted_stats
            mean_stats.append(extracted_stats.mean())
        mean_stats = np.array(mean_stats)
        stats[stat] = (mean_stats.mean(), mean_stats.std())
    return stats

def plot_rliable(log_readers_dict, baseline='random', **kwargs):

    # Investigating which game to pick for overgen gap extra exp.
    # for algo, log_readers in log_readers_dict.items():
    #     for logr in log_readers:
    #         env_name = logr.env_name
    #         if env_name in ['miner']:
    #             env_idx = ENV_NAMES.index(env_name)
    #             logs = logr.logs
    #             sns.lineplot(x='step', y='overgen_gap_stale', data=logs, label=f'{algo}').set(title=env_name, xlabel='step', ylabel='overgengap (stale)')
    # plt.show()
    # scores_test_m = {}
    # scores_train_by_test_m = {}
    # scores_train_by_train_m = {}
    # gen_gaps_m = {}
    # gen_gaps_no_norm_m = {}
    # overgen_gaps_m_s = {}
    # overgen_gaps_m = {}
    # MI_m = {}
    # for method in gen_gaps:
    #     scores_test_m[method] = test_scores_norm_by_test[method].mean(axis=0)
    #     scores_train_by_test_m[method] = train_scores_norm_by_test[method].mean(axis=0)
    #     scores_train_by_train_m[method] = train_scores_norm_by_train[method].mean(axis=0)
    #     gen_gaps_m[method] = gen_gaps[method].mean(axis=0)
    #     gen_gaps_no_norm_m[method] = gen_gaps_no_norm[method].mean(axis=0)
    #     overgen_gaps_m_s[method] = overgen_gaps_stale[method].mean(axis=0)
    #     overgen_gaps_m[method] = overgen_gaps[method].mean(axis=0)
    #     MI_m[method] = mutual_infos_u_stale[method].mean(axis=0)


    labels = LABELS
    qck_lbl_getter = {
        'vl1-MI': LABELS['s1-value_l1_s2-instance_pred_log_prob_bf-0.25_l2-0.1_fs-0.5_fe-1.0'],
        'vl1-U': LABELS['s1-value_l1_s2-random_bf-0.25_l2-1.0_fs-0.5_fe-1.0'],
        'U': LABELS['s1-random_s2-off_bf-0.0_l2-1.0_fs-0.0_fe-1.0'],
        'vl1': LABELS['s1-value_l1_s2-off_bf-0.0_l2-1.0_fs-0.0_fe-1.0'],
        'MI': LABELS['s1-instance_pred_log_prob_s2-off_bf-0.0_l2-1.0_fs-0.0_fe-1.0']
    }
    colors = None

    test_scores_norm_by_test = {}
    train_scores_norm_by_test = {}
    train_scores_norm_by_train = {}
    vl = {}
    norm_vl = {}
    gen_gaps = {}
    gen_gaps_no_norm = {}
    accuracies = {}
    accuracies_stale = {}
    iprobs = {}
    iprobs_stale = {}
    entropies = {}
    entropies_stale = {}
    mutual_infos = {}
    mutual_infos_stale = {}
    generalisation_bounds = {}
    generalisation_bounds_stale = {}
    mutual_infos_u = {}
    mutual_infos_u_stale = {}
    generalisation_bounds_u = {}
    generalisation_bounds_u_stale = {}
    overgen_gaps = {}
    overgen_gaps_stale = {}
    for key, log_readers in log_readers_dict.items():
        if key not in labels:
            continue
        test_score_norm_by_test = [[] for _ in range(len(ENV_NAMES))]
        train_score_norm_by_test = [[] for _ in range(len(ENV_NAMES))]
        train_score_norm_by_train = [[] for _ in range(len(ENV_NAMES))]
        value_l1 = [[] for _ in range(len(ENV_NAMES))]
        norm_value_l1 = [[] for _ in range(len(ENV_NAMES))]
        gen_gap = [[] for _ in range(len(ENV_NAMES))]
        gen_gap_no_norm = [[] for _ in range(len(ENV_NAMES))]
        accuracy = [[] for _ in range(len(ENV_NAMES))]
        accuracy_stale = [[] for _ in range(len(ENV_NAMES))]
        iprob = [[] for _ in range(len(ENV_NAMES))]
        iprob_stale = [[] for _ in range(len(ENV_NAMES))]
        entropy = [[] for _ in range(len(ENV_NAMES))]
        entropy_stale = [[] for _ in range(len(ENV_NAMES))]
        mutual_info = [[] for _ in range(len(ENV_NAMES))]
        mutual_info_stale = [[] for _ in range(len(ENV_NAMES))]
        generalisation_bound = [[] for _ in range(len(ENV_NAMES))]
        generalisation_bound_stale = [[] for _ in range(len(ENV_NAMES))]
        mutual_info_u = [[] for _ in range(len(ENV_NAMES))]
        mutual_info_u_stale = [[] for _ in range(len(ENV_NAMES))]
        generalisation_bound_u = [[] for _ in range(len(ENV_NAMES))]
        generalisation_bound_u_stale = [[] for _ in range(len(ENV_NAMES))]
        overgen_gap = [[] for _ in range(len(ENV_NAMES))]
        overgen_gap_stale = [[] for _ in range(len(ENV_NAMES))]
        for logr in log_readers:
            env_name = logr.env_name
            env_idx = ENV_NAMES.index(env_name)
            if hasattr(logr, 'normalised_scores'):
                test_score_norm_by_test[env_idx] = logr.normalised_scores
            else:
                test_score_norm_by_test[env_idx] = logr.final_test_eval_scores / BASELINE_SCORES[env_name][baseline][0]
            if hasattr(logr, 'train_set_normalised_scores_train'):
                train_score_norm_by_train[env_idx] = logr.train_set_normalised_scores_train
            else:
                train_score_norm_by_train[env_idx] = logr.final_train_eval_scores \
                                             / BASELINE_SCORES[env_name][f'{baseline}_train_set'][0]
            if hasattr(logr, 'normalised_value_l1'):
                norm_value_l1[env_idx] = logr.normalised_value_l1
            else:
                norm_value_l1[env_idx] = logr.final_value_l1 / BASELINE_SCORES[env_name][baseline][0]
            value_l1[env_idx] = logr.value_l1
            train_score_norm_by_test[env_idx] = logr.normalised_scores_train
            gen_gap[env_idx] = logr.normalised_scores_train - logr.normalised_scores
            gen_gap_no_norm[env_idx] = logr.final_train_eval_scores - logr.final_test_eval_scores
            accuracy[env_idx] = logr.final_stats.instance_pred_accuracy
            accuracy_stale[env_idx] = logr.final_stats.instance_pred_accuracy_stale
            iprob[env_idx] = logr.final_stats.instance_pred_prob
            iprob_stale[env_idx] = logr.final_stats.instance_pred_prob_stale
            entropy[env_idx] = logr.final_stats.instance_pred_entropy
            entropy_stale[env_idx] = logr.final_stats.instance_pred_entropy_stale
            mutual_info[env_idx] = logr.final_stats.mutual_information
            mutual_info_stale[env_idx] = logr.final_stats.mutual_information_stale
            generalisation_bound[env_idx] = logr.final_stats.generalisation_bound
            generalisation_bound_stale[env_idx] = logr.final_stats.generalisation_bound_stale
            mutual_info_u[env_idx] = logr.final_stats.mutual_information_u
            mutual_info_u_stale[env_idx] = logr.final_stats.mutual_information_u_stale
            generalisation_bound_u[env_idx] = logr.final_stats.generalisation_bound_u
            generalisation_bound_u_stale[env_idx] = logr.final_stats.generalisation_bound_u_stale
            overgen_gap[env_idx] = logr.final_stats.overgen_gap
            overgen_gap_stale[env_idx] = logr.final_stats.overgen_gap_stale
        test_scores_norm_by_test[labels[key]] = np.array(test_score_norm_by_test).T
        train_scores_norm_by_test[labels[key]] = np.array(train_score_norm_by_test).T
        train_scores_norm_by_train[labels[key]] = np.array(train_score_norm_by_train).T
        vl[labels[key]] = np.array(value_l1).T
        norm_vl[labels[key]] = np.array(norm_value_l1).T
        gen_gaps[labels[key]] = np.array(gen_gap).T
        gen_gaps_no_norm[labels[key]] = np.array(gen_gap_no_norm).T
        accuracies[labels[key]] = np.array(accuracy).T
        accuracies_stale[labels[key]] = np.array(accuracy_stale).T
        iprobs[labels[key]] = np.array(iprob).T
        iprobs_stale[labels[key]] = np.array(iprob_stale).T
        entropies[labels[key]] = np.array(entropy).T
        entropies_stale[labels[key]] = np.array(entropy_stale).T
        mutual_infos[labels[key]] = np.array(mutual_info).T
        mutual_infos_stale[labels[key]] = np.array(mutual_info_stale).T
        generalisation_bounds[labels[key]] = np.array(generalisation_bound).T
        generalisation_bounds_stale[labels[key]] = np.array(generalisation_bound_stale).T
        mutual_infos_u[labels[key]] = np.array(mutual_info_u).T
        mutual_infos_u_stale[labels[key]] = np.array(mutual_info_u_stale).T
        generalisation_bounds_u[labels[key]] = np.array(generalisation_bound_u).T
        generalisation_bounds_u_stale[labels[key]] = np.array(generalisation_bound_u_stale).T
        overgen_gaps[labels[key]] = np.array(overgen_gap).T
        overgen_gaps_stale[labels[key]] = np.array(overgen_gap_stale).T

    def compute_correlations(v1, v2):
        lin_coef = np.corrcoef(v1, v2)
        dist_corr = calculate_dist_corr(v1, v2)
        kendall, kendall_p = calculate_Kendall(v1, v2)
        print(f'linear correlation coef = \n {lin_coef}, \n'
              f'distance correlation coef = {dist_corr}, \n'
              f'kendall = {kendall}, kendall p ={kendall_p} \n')
        return lin_coef, dist_corr, kendall, kendall_p

    all_mutual_infos = np.array([mutual_infos_u_stale[key] for key in mutual_infos_u_stale]).flatten()
    all_gen_gaps = np.array([gen_gaps_no_norm[key] for key in gen_gaps_no_norm]).flatten()
    all_gen_gaps_norm = np.array([gen_gaps[key] for key in gen_gaps]).flatten()
    all_vl = np.array([vl[key] for key in vl]).flatten()
    all_norm_vl = np.array([norm_vl[key] for key in norm_vl]).flatten()

    print("MI/GENGAP CORRELATION ANALYSIS")
    mi_gengap_lin_coef, mi_gengap_dist_corr, mi_gengap_kendall, mi_gengap_kendall_p = compute_correlations(all_mutual_infos, all_gen_gaps)
    print("MI/GENGAP_NORM CORRELATION ANALYSIS")
    mi_gengap_norm_lin_coef, mi_gengap_norm_dist_corr, mi_gengap_norm_kendall, mi_gengap_norm_kendall_p = compute_correlations(all_mutual_infos, all_gen_gaps_norm)
    print("MI/VL CORRELATION ANALYSIS")
    mi_vl_lin_coef, mi_vl_dist_corr, mi_vl_kendall, mi_vl_kendall_p = compute_correlations(all_mutual_infos, all_vl)
    print("MI/VL_NORM CORRELATION ANALYSIS")
    mi_normvl_lin_coef, mi_normvl_dist_corr, mi_normvl_kendall, mi_normvl_kendall_p = compute_correlations(all_mutual_infos, all_norm_vl)
    print("VL/GENGAP CORRELATION ANALYSIS")
    vl_gengap_lin_coef, vl_gengap_dist_corr, vl_gengap_kendall, vl_gengap_kendall_p = compute_correlations(all_vl, all_gen_gaps)
    print("NORM_VL/GENGAP_NORM CORRELATION ANALYSIS")
    normvl_gengap_norm_lin_coef, normvl_gengap_norm_dist_corr, normvl_gengap_norm_kendall, normvl_gengap_norm_kendall_p = compute_correlations(all_norm_vl, all_gen_gaps_norm)

    aggregate_func = lambda x: np.array([
        # metrics.aggregate_median(x),
        # metrics.aggregate_iqm(x),
        metrics.aggregate_mean(x),
        # metrics.aggregate_optimality_gap(x)
    ])

    aggregate_train_scores_norm_by_train, aggregate_train_score_norm_by_train_cis = rly.get_interval_estimates(
        train_scores_norm_by_train, aggregate_func, reps=50000)

    aggregate_test_scores_norm_by_test, aggregate_test_scores_norm_by_test_cis = rly.get_interval_estimates(
        test_scores_norm_by_test, aggregate_func, reps=50000)

    aggregate_gaps, aggregate_gaps_cis = rly.get_interval_estimates(
        gen_gaps, aggregate_func, reps=50000)

    join_agg_scores = lambda x,y: np.concatenate([x, y], axis=0)
    join_agg_cis = lambda x,y: np.concatenate([x, y], axis=-1)
    make_algo_pairs = lambda id_pairs, scores: {f'{qck_lbl_getter[pair[0]]}~{qck_lbl_getter[pair[1]]}':
                                                (scores[qck_lbl_getter[pair[0]]], scores[qck_lbl_getter[pair[1]]])
                                                for pair in id_pairs}
    aggregate_scores_j = {key: join_agg_scores(aggregate_train_scores_norm_by_train[key], test_scores_norm_by_test[key]) for key in aggregate_train_scores_norm_by_train}
    aggregate_scores_j = {key: join_agg_scores(aggregate_scores_j[key], aggregate_gaps[key]) for key in aggregate_scores_j}

    aggregate_score_cis_j = {key: join_agg_cis(aggregate_train_score_norm_by_train_cis[key], aggregate_test_scores_norm_by_test_cis[key]) for key in aggregate_train_score_norm_by_train_cis}
    aggregate_score_cis_j = {key: join_agg_cis(aggregate_score_cis_j[key], aggregate_gaps_cis[key]) for key in aggregate_score_cis_j}

    pairs = [
        ("vl1-U", "vl1"),
        ("vl1-MI", "vl1"),
        ("vl1", "U"),
        ("MI", "U"),
    ]

    algorithm_pairs = make_algo_pairs(pairs, test_scores_norm_by_test)
    algorithm_pairs_gap = make_algo_pairs(pairs, gen_gaps)

    average_probabilities, average_prob_cis = \
        rly.get_interval_estimates(algorithm_pairs, metrics.probability_of_improvement, reps=2000)

    average_probabilities_gap, average_prob_cis_gap = \
        rly.get_interval_estimates(algorithm_pairs_gap, metrics.probability_of_improvement, reps=2000)

    ### PLOTTING CODE

    # SCATTER PLOT OF MI TO GEN GAP
    def save_scatter_plot(filename, x, y, xlabel, ylabel, **kwargs):
        fig, ax = plt.subplots(figsize=(10, 10))
        ax.set_xlabel(xlabel)
        ax.set_ylabel(ylabel)
        for key in x:
            ax.scatter(x[key].mean(axis=0), y[key].mean(axis=0), label=key)
        ax.legend()
        adjust_legend(fig, ax)
        plt.tight_layout()
        plt.savefig(os.path.join(args.output_path, f"{filename}.{PLOT_FILE_FORMAT}"))
        plt.close(fig)

    save_scatter_plot("MI_gengap_scatter", mutual_infos_u_stale, gen_gaps_no_norm, 'Mutual Information', 'GenGap')
    save_scatter_plot("MI_normgengap_scatter", mutual_infos_u_stale, gen_gaps, 'Mutual Information', 'Normalised '
                                                                                                  'GenGap')

    save_scatter_plot("MI_vl_scatter", mutual_infos_u_stale, vl, 'Mutual Information', '$\ell_1$ value loss')
    save_scatter_plot("MI_normvl_scatter", mutual_infos_u_stale, norm_vl, 'Mutual Information', 'Normalised $\ell_1$ '
                                                                                                'value loss')

    save_scatter_plot("vl_gengap_scatter", vl, gen_gaps_no_norm, '$\ell_1$ value loss', 'GenGap')
    save_scatter_plot("normvl_normgengap_scatter", norm_vl, gen_gaps, 'Normalised $\ell_1$ value loss', 'Normalised GenGap')

    # PLOT AGGREGATE SCORES
    def save_score_intervals_plot(filename, joined_agg_scores, joined_agg_scores_cis, **kwargs):
        # PLOT MEAN NORM SCORES AND GAP
        if 'subfigure_width' in kwargs:
            num_metrics = len(joined_agg_scores[list(joined_agg_scores.keys())[0]])
            figsize = (kwargs['subfigure_width'] * num_metrics * 1.4, kwargs['row_height'] * len(joined_agg_scores) * 4)
        else:
            figsize = None
        fig, ax = plot_utils.plot_interval_estimates(joined_agg_scores, joined_agg_scores_cis, figsize=figsize, **kwargs)
        if isinstance(ax, np.ndarray):
            for a in ax:
                a.spines[['bottom']].set_color('black')
        else:
            ax.spines[['bottom']].set_color('black')
        plt.tight_layout()
        plt.savefig(os.path.join(args.output_path, f"{filename}.{PLOT_FILE_FORMAT}"))
        plt.close(fig)

    save_score_intervals_plot("procgen_score_aggregates_mean_score_gaps", aggregate_scores_j, aggregate_score_cis_j,
                                metric_names=['Mean Normalised Train Score',
                                              'Mean Normalised Test Score',
                                              'Mean Normalised GenGap'],
                                algorithms=list(aggregate_scores_j.keys()),
                                colors=colors,
                                xlabel=f'',
                                subfigure_width=4.0,
                                row_height=0.15,
                                left=0.0,
                                xlabel_y_coordinate=0.1)

    # PLOT PROB OF IMPROVEMENT
    def save_probablity_of_improvement_plot(filename, avg_p, avg_p_cis, **kwargs):
        # PLOT PROB OF IMPROVEMENT GAP
        axes = plot_utils.plot_probability_of_improvement(avg_p, avg_p_cis,
                                                          figsize=(6, 3),
                                                          pair_separator='~',
                                                          **kwargs)
        if isinstance(axes, np.ndarray):
            for a in axes:
                a.spines[['bottom']].set_color('black')
        else:
            axes.spines[['bottom']].set_color('black')
        plt.tight_layout()
        plt.savefig(os.path.join(args.output_path, f"{filename}.{PLOT_FILE_FORMAT}"))

    save_probablity_of_improvement_plot("procgen_prob_of_improvement", average_probabilities, average_prob_cis,
                                        xlabel='P(X $>$ Y), Normalised Score')
    save_probablity_of_improvement_plot("procgen_prob_of_improvement_gap", average_probabilities_gap, average_prob_cis_gap,
                                        xlabel='P(X $>$ Y), Normalised Generalisation Gap')

if __name__ == '__main__':

    logger = logging.getLogger(__name__)
    args=parse_args()
    os.makedirs(args.output_path, exist_ok=True)
    seeds = args.seeds.split(',') if args.seeds else None
    if seeds is not None:
        args.ignore_extra_pids = True

    all_run_names = [name for name in os.listdir(args.base_path) if os.path.isdir(os.path.join(args.base_path, name))]
    ignore_patterns = args.ignore_runs.split(',')
    run_names = [name for name in all_run_names if not any([re.search(pattern, name) for pattern in ignore_patterns])]
    log_readers = [reader.LogReader(run_name, args.base_path, args.output_path, rolling_window=args.rolling_window, seeds=seeds, ignore_extra=args.ignore_extra_pids) for run_name in run_names]
    log_readers = [log_reader for log_reader in log_readers if log_reader.completed]

    temp = defaultdict(list)
    for log_reader in log_readers:
        # group different envs together
        run_name_no_env = re.sub(r'e-\w+_', '', log_reader.run_name)
        temp[run_name_no_env].append(log_reader)
    log_readers = temp

    for key in list(log_readers.keys()):
        if len(log_readers[key]) < 16:
            del log_readers[key]
        if key == 's1-value_l1_s2-off_bf-0.0_l2-1.0_fs-0.0_fe-1.0':
            update_baseline_scores(BASELINE_SCORES, log_readers[key], baseline='plr')
        elif key == 's1-random_s2-off_bf-0.0_l2-1.0_fs-0.0_fe-1.0':
            update_baseline_scores(BASELINE_SCORES, log_readers[key], baseline='random')

    stats = {}
    for run_id in log_readers:
        scn = mean_normalised_scores(log_readers[run_id])
        st = {'normalised_score': scn}
        scn_tr = mean_normalised_scores(log_readers[run_id], mode='train')
        st.update({'normalised_score_train': scn_tr})
        st.update(compute_stats(log_readers[run_id]))
        stats[run_id] = DotDict(st)

    plot_rliable(log_readers)

    # Careful, no checks to verify number of seeds is the same across runs
    print("Done")
