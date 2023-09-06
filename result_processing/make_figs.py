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

EVAL_METRICS = ['test_returns', 'solved_rate']

PLOT_FILE_FORMAT = 'pdf'

# {batch_name: {kw: val}}
PLOTTING = {
    # 'latent_dataset': {'label': 'latent-$\mathcal{D}$'},
    'dr_dataset': {'label': 'iid-$\mathcal{D}$'},
    'dr_level_space': {'label': 'DR'},
    'plr_dataset': {'label': 'PLR-$\mathcal{D}$'},
    'plr_level_space': {'label': 'PLR'},
    'accel_dataset': {'label': 'ACCEL-$\mathcal{D}$'},
    'accel_level_space': {'label': 'ACCEL'},
    }
# Set plotting colors from a seaborn palette.
plotting_color_palette = sns.color_palette('colorblind', n_colors=len(PLOTTING))
PLOTTING = {
    method: {'color': plotting_color_palette[i], **kwargs}
    for i, (method, kwargs) in enumerate(PLOTTING.items())
}
# Use the order in the PLOTTING dict for the legend.
METHOD_ORDER = list([d['label'] for k, d in PLOTTING.items()])


def parse_args():
    parser = argparse.ArgumentParser(description='Fig')
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
        default='baserun,'
                # 's1-value_l1_s2-instance_pred_log_prob,'
                's1-value_l1_s2-off,'
                's1-random_s2-off,'
                's1-instance_pred_log_prob_s2-off',
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


def mean_normalised_scores(log_readers, baseline='random'):

    mean_scores = []
    std_scores = []
    for logr in log_readers:
        mean_scores.append(logr.final_test_eval_mean / BASELINE_SCORES[logr.env_name][baseline][0])
        std_scores.append(logr.final_test_eval_std / BASELINE_SCORES[logr.env_name][baseline][0])
    mean_scores = np.array(mean_scores)
    std_scores = np.array(std_scores)
    std = 1/len(std_scores) * np.sqrt((std_scores**2).sum())
    return mean_scores.mean(), std

def mean_precision(log_readers):
    raise NotImplementedError

def mean_generalisation_gap(log_readers):
    raise NotImplementedError


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
    log_readers = [reader.LogReader(run_name, args.base_path, args.output_path, seeds=seeds, ignore_extra=args.ignore_extra_pids) for run_name in run_names]
    log_readers = [log_reader for log_reader in log_readers if log_reader.completed]

    temp = defaultdict(list)
    for log_reader in log_readers:
        # group different envs together
        run_name_no_env = re.sub(r'e-\w+_', '', log_reader.run_name)
        temp[run_name_no_env].append(log_reader)
        # temp[log_reader.env_name][f's1-{log_reader.s1}_s2-{log_reader.s2}_bf-{log_reader.bf}'].append(log_reader)
        # temp[log_reader.env_name][log_reader.run_name].append(log_reader)

    log_readers = temp

    for key in list(log_readers.keys()):
        if len(log_readers[key]) < 16:
            del log_readers[key]
        for i, log_reader in enumerate(log_readers[key]):
            if not hasattr(log_reader, 'env_name'):
                print(key, i)
    stats = {}
    for run_id in log_readers:
        scn = mean_normalised_scores(log_readers[run_id])
        st = {'norm_test_sc_mean': scn[0],
              'norm_test_sc_std': scn[1], }
        stats[run_id] = DotDict(st)

    # Careful, number of seeds not homogeneous across runs

    print("Done")