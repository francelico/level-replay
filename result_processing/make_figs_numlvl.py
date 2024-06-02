import argparse
import copy
import os
import logging
from collections import defaultdict

import matplotlib.pyplot as plt
import matplotlib as mpl
import numpy as np
import seaborn as sns
import pandas as pd
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

ENV_NAMES = 'miner'.split(',')

PLOT_FILE_FORMAT = 'pdf'

METHOD_LABELS = {
    's1-value_l1' : '$S=S^V$',
    's1-random': '$P_S=\mathcal{U}$',
    }

LEVELSET_SIZES = {
    '200': '$|L|=200$',
    '500': '$|L|=500$',
    '1000' : '$|L|=1000$',
    '2000' : '$|L|=2000$'}

runs = []
plot_args = []
for env_name in ENV_NAMES:
    for method in METHOD_LABELS:
        for levelset_size in LEVELSET_SIZES:
            runs.append(f"e-{env_name}_{method}_lvl-{levelset_size}")
            plot_args.append({'label': METHOD_LABELS[method], 'levelset_size': LEVELSET_SIZES[levelset_size]})

# {batch_name: {kw: val}}
PLOTTING = {run: plot for run, plot in zip(runs, plot_args)}
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
        help='rolling window for smoother plots.')
    parser.add_argument(
        '--histogram_metrics',
        type=str,
        default='run_average_shift_gap_pids,final_shift_gap_pids,final_gen_gap_pids,final_test_eval_scores,final_train_eval_scores,delta_train_eval_scores',
    )

    return parser.parse_args()

def compute_stats(log_readers, stat_keys=None, update=-1, **kwargs):

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

def ax_format(ax):
    ax.grid(False)
    ax.spines[['right', 'top']].set_visible(False)
    ax.spines[['bottom', 'left']].set_color('black')
    ax.tick_params(bottom=True, left=True, )
    ax.yaxis.set_major_locator(plt.MaxNLocator(4))

def adjust_ax_fontsize(ax, fontsize=24):
    # Make room for legend in the figure

    for item in (
         [ax.title, ] +
         ax.get_xticklabels() +
         ax.get_yticklabels()
    ):
        item.set_fontsize(fontsize)
    ax.xaxis.label.set_fontsize(fontsize)
    ax.yaxis.label.set_fontsize(fontsize)

def adjust_legend(fig, ax, ncol=3, fontsize=20):

    fig.tight_layout(rect=(0, 0, 1, 0.93))

    handles, labels = ax.get_legend_handles_labels()
    labels, handles = dedupe_legend_labels_and_handles(labels, handles)

    fig.legend(handles, labels,
               ncol=ncol,
               bbox_to_anchor=(0.5, 0.95),
               bbox_transform=fig.transFigure,
               loc='center',
               frameon=False,
               fontsize=fontsize
               )

def separate_legend(ax_o, ncol=3):
    fig, ax = plt.subplots()
    fig.set_size_inches(ncol*2, 0.5)
    handles, labels = ax_o.get_legend_handles_labels()
    labels, handles = dedupe_legend_labels_and_handles(labels, handles)
    fig.legend(handles, labels,
               ncol=ncol,
               bbox_to_anchor=(0.5, 0.5),
               bbox_transform=fig.transFigure,
               loc='center',
               frameon=False,
               # fontsize=fontsize
               )
    ax.spines[['right', 'top','bottom', 'left']].set_visible(False)
    ax.set_xticks([])
    ax.set_yticks([])
    return fig

def plot_numlvl_histograms(log_readers, metrics, **kwargs):
    kwargs = {}
    kwargs['method_plot_args'] = PLOTTING
    kwargs['plot_args'] = {'legend_order': list(LEVELSET_SIZES.values())}
    save_to = args.output_path

    mean_train_scores_random = {}
    for logr_array in log_readers.values():
        assert len(logr_array) == 1
        logr = logr_array[0]
        if 'random' in logr.run_name:
            mean_train_scores_random[logr.run_name] = logr.final_train_eval_scores.mean()

    df = []
    for logr_array in log_readers.values():
        assert len(logr_array) == 1
        logr = logr_array[0]
        random_run_name = logr.run_name.replace('s1-value_l1', 's1-random')
        for i in range(len(logr.pid_dirs)):
            logr.run_average_shift_gap_var_pids = logr.run_average_shift_gap_pids / logr.final_train_eval_scores
            logr.delta_train_eval_scores = mean_train_scores_random[random_run_name] - logr.final_train_eval_scores
            row = {metric : getattr(logr, metric)[i] for metric in metrics}
            row['run_name'] = logr.run_name
            row['env_name'] = logr.env_name
            if 'method_plot_args' in kwargs:
                row.update(kwargs['method_plot_args'][logr.run_name])
            df.append(row)
    df = pd.DataFrame(df)

    shiftgap_df = df[df['run_name'].str.contains('s1-value_l1')]
    shiftgap_df['label'] = 'ShiftGap'
    shiftgap_df['metric'] = shiftgap_df['final_shift_gap_pids']
    diff_train_scores_df = df[df['run_name'].str.contains('s1-value_l1')]
    diff_train_scores_df['label'] = '$\Delta$ train score'
    diff_train_scores_df['metric'] = diff_train_scores_df['delta_train_eval_scores']
    df2 = pd.concat([shiftgap_df, diff_train_scores_df])

    plot_args = kwargs.get('plot_args', {})
    sns_args = {
        'hue_order': plot_args.get('legend_order', None),
        'alpha': 0.75,
    }

    fig, ax = plt.subplots()
    ax = sns.barplot(df2, x="label", y="metric", hue="levelset_size", ax=ax, **sns_args)
    ax.set_xlabel('')
    ax.set_ylabel('')
    adjust_ax_fontsize(ax)
    ax_format(ax)
    ax.legend_ = None
    fig.tight_layout()
    fig.savefig(os.path.join(save_to, f'numlvl_shiftgap_diffs.{PLOT_FILE_FORMAT}'))

    fig, ax = plt.subplots()
    ax = sns.barplot(df, x="label", y="final_gen_gap_pids", hue="levelset_size", ax=ax, **sns_args)
    ax.set_xlabel('')
    ax.set_ylabel('GenGap')
    adjust_ax_fontsize(ax)
    ax_format(ax)
    #Separate legend
    legend_fig = separate_legend(ax, ncol=len(plot_args.get('legend_order', None)))
    legend_fig.savefig(os.path.join(save_to, f'numlvl_legend.{PLOT_FILE_FORMAT}'))
    ax.legend_ = None
    fig.tight_layout()
    fig.savefig(os.path.join(save_to, f'numlvl_final_gen_gap.{PLOT_FILE_FORMAT}'))

    fig, ax = plt.subplots()
    ax = sns.barplot(df, x="label", y="final_test_eval_scores", hue="levelset_size", ax=ax, **sns_args)
    ax.set_xlabel('')
    ax.set_ylabel('Test scores')
    ax.set_ylim(0, 14)
    adjust_ax_fontsize(ax)
    ax_format(ax)
    ax.legend_ = None
    fig.tight_layout()
    fig.savefig(os.path.join(save_to, f'numlvl_final_test_scores.{PLOT_FILE_FORMAT}'))

    fig, ax = plt.subplots()
    ax = sns.barplot(df, x="label", y="final_train_eval_scores", hue="levelset_size", ax=ax, **sns_args)
    ax.set_xlabel('')
    ax.set_ylabel('Train scores')
    ax.set_ylim(0, 14)
    adjust_ax_fontsize(ax)
    ax_format(ax)
    ax.legend_ = None
    fig.tight_layout()
    fig.savefig(os.path.join(save_to, f'numlvl_final_train_scores.{PLOT_FILE_FORMAT}'))

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

    stats = {}
    for run_id in log_readers:
        st = compute_stats(log_readers[run_id])
        stats[run_id] = DotDict(st)

    plot_numlvl_histograms(log_readers, args.histogram_metrics.split(','))

    # Careful, no checks to verify number of seeds is the same across runs
    print("Done")
