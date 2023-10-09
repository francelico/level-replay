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

def update_baseline_scores(BASELINE_SCORES, logreaders, baseline='random'):
    for logr in logreaders:
        BASELINE_SCORES[logr.env_name][baseline] = (logr.final_test_eval_mean, logr.final_test_eval_std)

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
            ids = np.argsort(np.abs(logr.logs['total_student_grad_updates'] - update))[:len(logr.pid_dirs)]
            scores = logr.logs['train_eval:mean_episode_return_mavg'][ids].to_numpy()
            logr.normalised_scores_train = scores / BASELINE_SCORES[logr.env_name][baseline][0]
            mean_scores.append(scores.mean() / BASELINE_SCORES[logr.env_name][baseline][0])
            std_scores.append(scores.std() / BASELINE_SCORES[logr.env_name][baseline][0])
    mean_scores = np.array(mean_scores)
    std_scores = np.array(std_scores)
    std = 1/len(std_scores) * np.sqrt((std_scores**2).sum())
    return mean_scores.mean(), std

def compute_stats(log_readers, stat_keys=None, update=-1, **kwargs):

    if stat_keys is None:
        stat_keys = ['instance_pred_accuracy_train', 'instance_pred_prob_train', 'instance_pred_entropy_train',
                    'instance_pred_accuracy', 'instance_pred_prob', 'instance_pred_entropy', 'generalisation_gap',
                    'instance_pred_accuracy_stale', 'instance_pred_prob_stale', 'instance_pred_entropy_stale',
                    'mutual_information', 'mutual_information_stale', 'generalisation_bound', 'generalisation_bound_stale',
                     'mutual_information_u', 'mutual_information_u_stale', 'generalisation_bound_u', 'generalisation_bound_u_stale',]
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
            ids = np.argsort(np.abs(logr.logs['total_student_grad_updates'] - update))[:len(logr.pid_dirs)]
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

    labels = LABELS
    qck_lbl_getter = {
        'vl1-MI': LABELS['s1-value_l1_s2-instance_pred_log_prob_bf-0.25_l2-0.1_fs-0.5_fe-1.0'],
        'vl1-U': LABELS['s1-value_l1_s2-random_bf-0.25_l2-1.0_fs-0.5_fe-1.0'],
        'U': LABELS['s1-random_s2-off_bf-0.0_l2-1.0_fs-0.0_fe-1.0'],
        'vl1': LABELS['s1-value_l1_s2-off_bf-0.0_l2-1.0_fs-0.0_fe-1.0'],
        'MI': LABELS['s1-instance_pred_log_prob_s2-off_bf-0.0_l2-1.0_fs-0.0_fe-1.0']
    }
    colors = None

    metric_scores = {}
    gen_gaps = {}
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
    for key, log_readers in log_readers_dict.items():
        if key not in labels:
            continue
        norm_scores = [[] for _ in range(len(ENV_NAMES))]
        gen_gap = [[] for _ in range(len(ENV_NAMES))]
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
        for logr in log_readers:
            env_name = logr.env_name
            env_idx = ENV_NAMES.index(env_name)
            if hasattr(logr, 'normalised_scores'):
                norm_scores[env_idx] = logr.normalised_scores
            else:
                norm_scores[env_idx] = logr.final_test_eval_scores / BASELINE_SCORES[env_name][baseline][0]
            gen_gap[env_idx] = logr.normalised_scores_train - logr.normalised_scores
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
        metric_scores[labels[key]] = np.array(norm_scores).T
        gen_gaps[labels[key]] = np.array(gen_gap).T
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

    # MEASURE CORRELATION BETWEEN MUTUAL INFORMATION AND GENERALISATION GAP
    # kendall = 0.41129332479141845, kendall_p = 4.990833884155022e-28
    all_mutual_infos = np.array([mutual_infos_u_stale[key] for key in gen_gaps.keys()]).flatten()
    all_gen_gaps = np.array([gen_gaps[key] for key in gen_gaps.keys()]).flatten()
    lin_coef = np.corrcoef(all_mutual_infos, all_gen_gaps)
    dist_corr = calculate_dist_corr(all_mutual_infos, all_gen_gaps)
    kendall, kendall_p = calculate_Kendall(all_mutual_infos, all_gen_gaps)

    aggregate_func = lambda x: np.array([
        # metrics.aggregate_median(x),
        # metrics.aggregate_iqm(x),
        metrics.aggregate_mean(x),
        # metrics.aggregate_optimality_gap(x)
    ])

    aggregate_scores, aggregate_score_cis = rly.get_interval_estimates(
        metric_scores, aggregate_func, reps=50000)

    aggregate_gaps, aggregate_gaps_cis = rly.get_interval_estimates(
        gen_gaps, aggregate_func, reps=50000)

    join_agg_scores = lambda x,y: np.concatenate([x, y], axis=0)
    join_agg_gaps = lambda x,y: np.concatenate([x, y], axis=-1)
    make_algo_pairs = lambda id_pairs, scores: {f'{qck_lbl_getter[pair[0]]}~{qck_lbl_getter[pair[1]]}':
                                                (scores[qck_lbl_getter[pair[0]]], scores[qck_lbl_getter[pair[1]]])
                                                for pair in id_pairs}
    aggregate_scores_j = {key: join_agg_scores(aggregate_scores[key], aggregate_gaps[key]) for key in aggregate_scores.keys()}
    aggregate_score_cis_j = {key: join_agg_gaps(aggregate_score_cis[key], aggregate_gaps_cis[key]) for key in aggregate_score_cis.keys()}

    pairs = [
        ("vl1-U", "vl1"),
        ("vl1-MI", "vl1"),
        ("vl1", "U"),
        ("MI", "U"),
    ]

    algorithm_pairs = make_algo_pairs(pairs, metric_scores)
    algorithm_pairs_gap = make_algo_pairs(pairs, gen_gaps)

    average_probabilities, average_prob_cis = \
        rly.get_interval_estimates(algorithm_pairs, metrics.probability_of_improvement, reps=2000)

    average_probabilities_gap, average_prob_cis_gap = \
        rly.get_interval_estimates(algorithm_pairs_gap, metrics.probability_of_improvement, reps=2000)

    ### PLOTTING CODE

    # SCATTER PLOT OF MI TO GEN GAP
    for key, scores in gen_gaps.items():
        print(key, scores.mean(), scores.std())
        print(key, mutual_infos_u[key].mean(), mutual_infos_u[key].std())

    fig, ax = plt.subplots(figsize=(10, 10))
    ax.set_xlabel('Mutual Information')
    ax.set_ylabel('Normalised Generalisation Gap')
    for key, scores in gen_gaps.items():
        ax.scatter(mutual_infos_u_stale[key].mean(axis=0), scores.mean(axis=0), label=key)
    ax.legend()
    adjust_legend(fig, ax)
    if args.debug:
        plt.show()
    else:
        plt.savefig(os.path.join(args.output_path, f"Mutual_Information_stale.{PLOT_FILE_FORMAT}"))
        plt.close(fig)

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
                                metric_names=['Mean Norm. Score', 'Mean Norm. Gen. Gap'],
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
        plt.close(fig)

    save_probablity_of_improvement_plot("procgen_prob_of_improvement", average_probabilities, average_prob_cis,
                                        xlabel='P(X $>$ Y), Normalised Score')
    save_probablity_of_improvement_plot("procgen_prob_of_improvement_gap", average_probabilities_gap, average_prob_cis_gap,
                                        xlabel='P(X $>$ Y), Normalised Generalisation Gap')

    return aggregate_scores, aggregate_score_cis, average_probabilities, average_prob_cis

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
