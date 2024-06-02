import argparse
import os
import logging
from collections import defaultdict

import matplotlib.pyplot as plt
import matplotlib as mpl
import numpy as np
import pandas as pd
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
from result_processing.make_figs_numlvl import separate_legend


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
    's1-value_l1_s2-off_bf-0.0_l2-1.0_fs-0.0_fe-1.0': {'label': '$S=S^V$'},
    's1-random_s2-off_bf-0.0_l2-1.0_fs-0.0_fe-1.0': {'label': '$P_S=\mathcal{U}$'},
    's1-instance_pred_log_prob_s2-off_bf-0.0_l2-1.0_fs-0.0_fe-1.0': {'label': '$S=S^{\mathrm{MI}}$'},
    's1-value_l1_s2-random_bf-0.25_l2-1.0_fs-0.5_fe-1.0': {'label': '$S=S^V, P_{S^\prime}=\mathcal{U}$'},
    's1-value_l1_s2-instance_pred_log_prob_bf-0.25_l2-0.1_fs-0.5_fe-1.0': {'label': '$S=S^V, S^\prime=S^{\mathrm{MI}}$'},
    }
# Set plotting colors from a seaborn palette.
plotting_color_palette = list(reversed(sns.color_palette('colorblind', n_colors=len(PLOTTING))))
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
        '--plot_train_eval_curves',
        action="store_true",
        help='Plot training and evaluation curves per environment and across all environments.')
    parser.add_argument(
        '--save_result_tables',
        action="store_true",
        help='Save result tables in tex format.')
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

def update_baseline_scores(BASELINE_SCORES, logreaders, baseline='random'):
    for logr in logreaders:
        BASELINE_SCORES[logr.env_name][baseline] = (logr.final_test_eval_mean, logr.final_test_eval_std)
        if baseline == 'random':
            BASELINE_SCORES[logr.env_name]['random_train_set'] = (logr.final_train_eval_mean, logr.final_train_eval_std)

def mean_normalised_scores(log_readers, baseline='random', mode='test', error='std'):

    all_scores = []
    for logr in log_readers:
        if mode == 'test':
            logr.normalised_scores = logr.final_test_eval_scores / BASELINE_SCORES[logr.env_name][baseline][0]
            all_scores.append(logr.normalised_scores)
        elif mode == 'train':
            update = logr.num_updates
            all_updates = logr.logs['total_student_grad_updates'].to_numpy()
            ids = np.argsort(np.abs(all_updates - update))[:len(logr.pid_dirs)]
            scores = logr.final_train_eval_scores
            logr.normalised_scores_train = scores / BASELINE_SCORES[logr.env_name][baseline][0]
            logr.train_set_normalised_scores_train = \
                scores / BASELINE_SCORES[logr.env_name][f'{baseline}_train_set'][0]
            all_scores.append(logr.normalised_scores_train)
            value_l1 = logr.logs['level_value_loss_mavg'][ids].to_numpy()
            logr.value_l1 = value_l1
            logr.normalised_value_l1 = value_l1 / scores.mean()
    all_scores = np.stack(all_scores).mean(axis=0) # aggregated across environments
    return all_scores

def compute_stats(log_readers, stat_keys=None, update=-1, per_env=True, **kwargs):

    if stat_keys is None:
        stat_keys = ['instance_pred_accuracy_train', 'instance_pred_prob_train', 'instance_pred_entropy_train',
                    'instance_pred_accuracy', 'instance_pred_prob', 'instance_pred_entropy', 'generalisation_gap',
                    'instance_pred_accuracy_stale', 'instance_pred_prob_stale', 'instance_pred_entropy_stale',
                     'level_value_loss',
                    'mutual_information', 'mutual_information_stale', 'generalisation_bound', 'generalisation_bound_stale',
                     'mutual_information_u', 'mutual_information_u_stale', 'generalisation_bound_u', 'generalisation_bound_u_stale',
                     'shift_gap', 'shift_gap_stale']
    if update == -1:
        update = log_readers[0].num_updates
        assert all([logr.num_updates == update for logr in log_readers])
    stats = {}
    for stat in stat_keys:
        all_stats = []
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
            if per_env:
                stats[f'{stat}:{logr.env_name}'] = extracted_stats
                if f'final_train_eval_scores:{logr.env_name}' not in stats:
                    stats[f'final_train_eval_scores:{logr.env_name}'] = logr.final_train_eval_scores
                if f'final_test_eval_scores:{logr.env_name}' not in stats:
                    stats[f'final_test_eval_scores:{logr.env_name}'] = logr.final_test_eval_scores
            all_stats.append(extracted_stats)
        all_stats = np.stack(all_stats).mean(axis=0)
        stats[stat] = all_stats #(all_stats.mean(), all_stats.std())
    return stats

def plot_rliable(log_readers_dict, baseline='random', **kwargs):

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
    generalisation_bounds = {}
    generalisation_bounds_stale = {}
    generalisation_bounds_u = {}
    generalisation_bounds_u_stale = {}
    shift_gaps = {}
    shift_gaps_stale = {}
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
        generalisation_bound = [[] for _ in range(len(ENV_NAMES))]
        generalisation_bound_stale = [[] for _ in range(len(ENV_NAMES))]
        generalisation_bound_u = [[] for _ in range(len(ENV_NAMES))]
        generalisation_bound_u_stale = [[] for _ in range(len(ENV_NAMES))]
        shift_gap = [[] for _ in range(len(ENV_NAMES))]
        shift_gap_stale = [[] for _ in range(len(ENV_NAMES))]
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
            generalisation_bound[env_idx] = logr.final_stats.generalisation_bound
            generalisation_bound_stale[env_idx] = logr.final_stats.generalisation_bound_stale
            generalisation_bound_u[env_idx] = logr.final_stats.generalisation_bound_u
            generalisation_bound_u_stale[env_idx] = logr.final_stats.generalisation_bound_u_stale
            shift_gap[env_idx] = logr.final_stats.shift_gap
            shift_gap_stale[env_idx] = logr.final_stats.shift_gap_stale
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
        generalisation_bounds[labels[key]] = np.array(generalisation_bound).T
        generalisation_bounds_stale[labels[key]] = np.array(generalisation_bound_stale).T
        generalisation_bounds_u[labels[key]] = np.array(generalisation_bound_u).T
        generalisation_bounds_u_stale[labels[key]] = np.array(generalisation_bound_u_stale).T
        shift_gaps[labels[key]] = np.array(shift_gap).T
        shift_gaps_stale[labels[key]] = np.array(shift_gap_stale).T

    def compute_correlations(v1, v2):
        lin_coef = np.corrcoef(v1, v2)
        dist_corr = calculate_dist_corr(v1, v2)
        kendall, kendall_p = calculate_Kendall(v1, v2)
        print(f'linear correlation coef = \n {lin_coef}, \n'
              f'distance correlation coef = {dist_corr}, \n'
              f'kendall = {kendall}, kendall p ={kendall_p} \n')
        return lin_coef, dist_corr, kendall, kendall_p

    all_mutual_infos = np.array([mutual_infos[key] for key in mutual_infos]).flatten()
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

    join_agg_scores = lambda x,y: np.concatenate([x, y], axis=0)
    join_agg_cis = lambda x,y: np.concatenate([x, y], axis=-1)
    join_agg = lambda sc_a, ci_a, sc_b, ci_b: ({key: join_agg_scores(sc_a[key], sc_b[key]) for key in sc_a},
                                               {key: join_agg_cis(ci_a[key], ci_b[key]) for key in ci_a})
    make_algo_pairs = lambda id_pairs, scores: {f'{qck_lbl_getter[pair[0]]}~{qck_lbl_getter[pair[1]]}':
                                                (scores[qck_lbl_getter[pair[0]]], scores[qck_lbl_getter[pair[1]]])
                                                for pair in id_pairs if (qck_lbl_getter[pair[0]] in scores and qck_lbl_getter[pair[1]] in scores)}

    aggregate_test_scores_norm_by_test, aggregate_test_scores_norm_by_test_cis = rly.get_interval_estimates(
        test_scores_norm_by_test, aggregate_func, reps=50000)
    aggregate_mutual_infos, aggregate_mutual_infos_cis = rly.get_interval_estimates(mutual_infos, aggregate_func, reps=50000)
    aggregate_gaps, aggregate_gaps_cis = rly.get_interval_estimates(
        gen_gaps, aggregate_func, reps=50000)
    aggregate_train_scores_norm_by_train, aggregate_train_score_norm_by_train_cis = rly.get_interval_estimates(
        train_scores_norm_by_train, aggregate_func, reps=50000)

    agg_sc, agg_ci = join_agg(aggregate_mutual_infos, aggregate_mutual_infos_cis, aggregate_gaps, aggregate_gaps_cis)
    agg_sc, agg_ci = join_agg(agg_sc, agg_ci, aggregate_train_scores_norm_by_train, aggregate_train_score_norm_by_train_cis)
    agg_sc, agg_ci = join_agg(agg_sc, agg_ci, aggregate_test_scores_norm_by_test, aggregate_test_scores_norm_by_test_cis)

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
        fig, ax = plt.subplots() #plt.subplots(figsize=(10, 10))
        ax.set_xlabel(xlabel)
        ax.set_ylabel(ylabel)
        for key in x:
            ax.scatter(x[key].mean(axis=0), y[key].mean(axis=0), label=key)
        ax.legend()
        adjust_legend(fig, ax)
        plt.tight_layout()
        plt.savefig(os.path.join(args.output_path, f"{filename}.{PLOT_FILE_FORMAT}"))
        plt.close(fig)

    save_scatter_plot("MI_gengap_scatter", mutual_infos, gen_gaps_no_norm, '$\mathrm{I}(L;b)$', 'GenGap')
    save_scatter_plot("MI_normgengap_scatter", mutual_infos, gen_gaps, '$\mathrm{I}(L;b)$', 'Normalised '
                                                                                                  'GenGap')

    save_scatter_plot("MI_vl_scatter", mutual_infos, vl, '$\mathrm{I}(L;b)$', '$\ell_1$ value loss')
    save_scatter_plot("MI_normvl_scatter", mutual_infos, norm_vl, '$\mathrm{I}(L;b)$', 'Normalised $\ell_1$ '
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

    algorithms = [lbl for lbl in METHOD_ORDER if lbl in agg_sc]
    save_score_intervals_plot("procgen_aggregates_gengap_MI_train_test", agg_sc, agg_ci,
                                metric_names=['$\mathrm{I}(L;b)$',
                                              'Normalised GenGap',
                                              'Normalised train score',
                                              'Normalised test score'],
                                algorithms=list(reversed(algorithms)),
                                colors=colors,
                                xlabel=f'',
                                subfigure_width=3.0,
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

def plot_per_env_curves(log_readers, metric, filename, make_legend_fig=False):

    def ax_formator(ax):
        # ax.grid(True, which='major', linewidth=0.5, color='black', alpha=0.5)
        ax.grid(False)
        ax.spines[['right', 'top','bottom', 'left']].set_color('black')
        ax.tick_params(bottom=True, left=True, )
        ax.xaxis.set_major_locator(plt.MaxNLocator(2))
        ax.set_xlim(0, 25)
        ax.yaxis.set_major_locator(plt.MaxNLocator(3))
        for item in ([ax.title, ax.xaxis.label, ax.yaxis.label] +
                     ax.get_xticklabels() + ax.get_yticklabels()):
            item.set_fontsize(20)

    fig, axes = plt.subplots(4,4)
    fig.set_size_inches(20, 20)
    for xpid, label in LABELS.items():
        if xpid not in log_readers:
            print(f"Warning: missing data for {xpid}")
            continue
        for logr in log_readers[xpid]:
            env_name = logr.env_name
            logs = logr.logs
            logs['stepM'] = logs['step'] / 1e6
            env_idx = ENV_NAMES.index(env_name)
            ax = axes.flatten()[env_idx]
            ax.set_title(env_name.capitalize())
            sns.lineplot(x='stepM', y=metric, data=logs, ax=ax, errorbar='se', estimator='mean', alpha=0.75, **PLOTTING[xpid])
            ax.set(xlabel='', ylabel='')

    legend_fig = None
    for ax in axes.flatten():
        ax_formator(ax)
        if legend_fig is None and make_legend_fig:
            legend_fig = separate_legend(ax, ncol=len(log_readers))
            legend_fig.savefig(os.path.join(args.output_path, f"{filename}_legend.{PLOT_FILE_FORMAT}"))
        ax.legend_ = None
    for ax_v in axes[:, 0]:
        ax_v.set_ylabel('Score')
    for ax_h in axes[-1, :]:
        ax_h.set_xlabel('Step [1e6]')
    fig.tight_layout()
    fig.savefig(os.path.join(args.output_path, f"{filename}.{PLOT_FILE_FORMAT}"))

def plot_all_env_curves(log_readers, metric, filename, baseline='random', ylabel='Score', make_legend_fig=False):

    def ax_formator(ax):
        # ax.grid(True, which='major', linewidth=0.5, color='black', alpha=0.5)
        ax.grid(False)
        ax.spines[['right', 'top']].set_visible(False)
        ax.spines[['bottom', 'left']].set_color('black')
        ax.tick_params(bottom=True, left=True, )
        ax.xaxis.set_major_locator(plt.MaxNLocator(2))
        ax.set_xlim(0, 25)
        ax.yaxis.set_major_locator(plt.MaxNLocator(4))
        for item in ([ax.title, ax.xaxis.label, ax.yaxis.label] +
                     ax.get_xticklabels() + ax.get_yticklabels()):
            item.set_fontsize(20)

    fig, ax = plt.subplots()
    fig.set_size_inches(10, 16)
    for xpid, label in LABELS.items():
        if xpid not in log_readers:
            print(f"Warning: missing data for {xpid}")
            continue
        dfs = []
        for logr in log_readers[xpid]:
            env_name = logr.env_name
            dfs.append(pd.DataFrame({'stepM': logr.logs['step'] / 1e6,
                                     metric: logr.logs[metric] / BASELINE_SCORES[env_name][baseline][0]}))
        logs = pd.concat(dfs).reset_index(drop=True)
        sns.lineplot(x='stepM', y=metric, data=logs, ax=ax, errorbar='se', estimator='mean', alpha=0.75, **PLOTTING[xpid])
    ax.set(xlabel='Step [1e6]', ylabel=ylabel)
    ax_formator(ax)
    ax.legend()
    if make_legend_fig:
        legend_fig = separate_legend(ax, ncol=len(log_readers))
        legend_fig.savefig(os.path.join(args.output_path, f"{filename}_legend.{PLOT_FILE_FORMAT}"))
    ax.legend_ = None
    fig.tight_layout()
    fig.savefig(os.path.join(args.output_path, f"{filename}.{PLOT_FILE_FORMAT}"))

def make_result_table_latex(stats, filename, quantity='final_test_eval_scores', rescale_f=1.0, agg_quantity=None,
                            agg_quantity_label=None, rescale_f_agg=1.0, decimals=1, bold_best=None, bold_p=0.05):
    """This function creates a pandas dataframe and uses the pandas.DataFrame.to_latex method to write a latex
    table to the output filename.
    The table columns are arranged as [env_name, *method_labels]
    The first 16 rows are the mean score and standard deviation for each procgen game. The last row contains the mean and standard deviation of the normalised final scores across all games.
    Scores are formatted as f{mean_score} \pm f{std_score}
    contains the different  contains the mean and standard deviation of the final non-normalised scores for each environment
    """

    if bold_best == 'max':
        best_fn = lambda x: max(x, key=x.get)
    elif bold_best == 'min':
        best_fn = lambda x: min(x, key=x.get)
    else:
        best_fn = None

    def get_bold_wrap_fn(scores, bold_p, best_fn=None):
        avg_scores = {label: scores.mean() for label, scores in row_scores.items()}
        best_label = best_fn(avg_scores)
        bold = {}
        for label in scores:
            if label == best_label:
                bold[label] = True
            else:
                bold[label] = welch_test(row_scores[best_label], row_scores[label])[1] > bold_p
        if all([b for b in bold.values()]):
            bold = {label: False for label in scores}
        bold_fns = {}
        for label in scores:
            if bold[label]:
                bold_fns[label] = lambda x: f"\\textbf{{{x}}}"
            else:
                bold_fns[label] = lambda x: x
        return bold_fns

    save_to = os.path.join(args.output_path, filename)
    label2method = {lab: met for met, lab in LABELS.items()}
    cols = ['Environment'] + METHOD_ORDER
    df = pd.DataFrame(columns=cols)
    for env_name in ENV_NAMES:
        row = [env_name.capitalize()]
        row_scores = {}
        for label in METHOD_ORDER:
            method_id = label2method[label]
            if method_id in stats:
                scores = stats[method_id][f'{quantity}:{env_name}']
                row_scores[label] = scores
        if best_fn is not None:
            bold_fns = get_bold_wrap_fn(row_scores, bold_p, best_fn)
        else:
            bold_fns = {label: lambda x: x for label in row_scores}
        for label in METHOD_ORDER:
            if label in row_scores:
                scores = row_scores[label]
                table_entry = f"{scores.mean()*rescale_f:.{decimals}f} $\pm$ {scores.std()*rescale_f:.{decimals}f}"
                row.append(bold_fns[label](table_entry))
            else:
                row.append('N/A')
        df.loc[len(df)] = row
    if agg_quantity is not None:
        row = [agg_quantity_label]
        row_scores = {}
        for label in METHOD_ORDER:
            method_id = label2method[label]
            if method_id in stats:
                scores = stats[method_id][agg_quantity]
                row_scores[label] = scores
        if best_fn is not None:
            bold_fns = get_bold_wrap_fn(row_scores, bold_p, best_fn)
        else:
            bold_fns = {label: lambda x: x for label in row_scores}
        for label in METHOD_ORDER:
            if label in row_scores:
                scores = row_scores[label]
                table_entry = f"{scores.mean() * rescale_f_agg:.{decimals}f} $\pm$ {scores.std() * rescale_f_agg:.{decimals}f}"
                row.append(bold_fns[label](table_entry))
            else:
                row.append('N/A')
        df.loc[len(df)] = row

    table = df.to_latex(index=False, escape=False)
    # table = df.style.to_latex(index=False, escape=False)
    with open(save_to, "w") as f:
        f.write(table)
    return table

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

    if args.save_result_tables:
        make_result_table_latex(stats, 'test_result_table.tex', quantity='final_test_eval_scores', rescale_f=1.0,
                                agg_quantity='normalised_score', agg_quantity_label='Normalised Test Scores (\%)',
                                rescale_f_agg=100, decimals=1, bold_best='max')
        make_result_table_latex(stats, 'train_result_table.tex', quantity='final_train_eval_scores', rescale_f=1.0,
                                agg_quantity='normalised_score_train', agg_quantity_label='Normalised Train Scores (\%)',
                                rescale_f_agg=100, decimals=1, bold_best='max')
        make_result_table_latex(stats, 'mutual_information_table.tex', quantity='mutual_information', rescale_f=1.0,
                                agg_quantity='mutual_information', agg_quantity_label='Average Mutual Information',
                                rescale_f_agg=1.0, decimals=2, bold_best='min')
        make_result_table_latex(stats, 'classifier_accuracy_table.tex', quantity='instance_pred_accuracy', rescale_f=100.0,
                                agg_quantity='instance_pred_accuracy', agg_quantity_label='Average Classifier Accuracy',
                                rescale_f_agg=100.0, decimals=1, bold_best='min')

    if args.plot_train_eval_curves:
        plot_all_env_curves(log_readers, 'train_eval:mean_episode_return_mavg', 'trainset_curves_all_env', baseline='random', ylabel='Train Score', make_legend_fig=True)
        plot_all_env_curves(log_readers, 'test:mean_episode_return_mavg', 'testset_curves_all_env', baseline='random', ylabel='Test Score', make_legend_fig=False)
        plot_all_env_curves(log_readers, 'generalisation_gap_mavg', 'gengap_curves_all_env', baseline='random', ylabel='Generalisation Gap', make_legend_fig=False)

        #/ metric = train_eval:mean_episode_return_mavg, test:mean_episode_return_mavg
        plot_per_env_curves(log_readers, 'train_eval:mean_episode_return_mavg', 'trainset_curves_per_env', make_legend_fig=True)
        plot_per_env_curves(log_readers, 'test:mean_episode_return_mavg', 'testset_curves_per_env', make_legend_fig=False)

    # plot_rliable(log_readers)

    # Careful, no checks to verify number of seeds is the same across runs
    print("Done")
