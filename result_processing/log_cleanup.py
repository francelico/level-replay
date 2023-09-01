import itertools
import json
import sys
import os
import re
import shutil
import csv
from collections import defaultdict
from typing import Dict, Any
import copy
import shlex

import torch
import numpy as np

from level_replay.file_writer import FileWriter, overwrite_file
from level_replay.arguments import parser


def grep(pattern, file):
    with open(file, 'r') as f:
        lines = f.readlines()

    matched_lines = []
    matched_line_numbers = []
    for i, line in enumerate(lines):
        if re.search(pattern, line):
            matched_lines.append(line)
            matched_line_numbers.append(i)

    return matched_lines, matched_line_numbers


def add_update_num_to_model_tar(results_dir, slurm_dir):
    slurm_log_files = [os.path.join(slurm_dir, f) for f in os.listdir(slurm_dir)]
    for slurm_log_file in slurm_log_files:
        match, _ = grep("successfully", slurm_log_file)
        if not match:
            match, _ = grep("Saving arguments to", slurm_log_file)
            if match:
                assert len(match) == 1
                match = match[0]
                runpath = match.split('Saving arguments to ')[1].strip('\n')
                # convert to path and only get the last two directories
                rel_runpath = os.path.join(*runpath.split('/')[-3:-1])
            match, line_no = grep("Saving checkpoint to", slurm_log_file)
            if match:
                line_id = line_no[-1] + 2
                with open(slurm_log_file, 'r') as f:
                    lines = f.readlines()
                line_with_update = lines[line_id]
                update_num = line_with_update.split(' ')[1]
                print(f'Found unsuccessfull run with last saved update {update_num}: {rel_runpath}. '
                      f'Renaming model.tar to model_{update_num}.tar')
                # now rename the 'model.tar' file in the runpath to 'model_{update_num}.tar'
                model_path = os.path.join(results_dir, rel_runpath, 'model.tar')
                new_model_path = os.path.join(results_dir, rel_runpath, f'model_{update_num}.tar')
                os.rename(model_path, new_model_path)
            else:
                print(f'Found unsuccessfull run without checkpoint: {rel_runpath}')

    for slurm_log_file in slurm_log_files:
        match, _ = grep("successfully", slurm_log_file)
        if match:
            match, _ = grep("Saving arguments to", slurm_log_file)
            if match:
                assert len(match) == 1
                match = match[0]
                runpath = match.split('Saving arguments to ')[1].strip('\n')
                # convert to path and only get the last two directories
                rel_runpath = os.path.join(*runpath.split('/')[-3:-1])
                print(f'Found successfull run: {rel_runpath}')
                if os.path.exists(os.path.join(results_dir, rel_runpath)):
                    if not os.path.exists(os.path.join(results_dir, rel_runpath, 'model.tar')):
                        print("model.tar does not exist. Checking for model_*.tar")
                        checkpoint_filenames = []
                        for file in os.listdir(os.path.join(results_dir, rel_runpath)):
                            if file.endswith(".tar"):
                                checkpoint_filenames.append(file)
                                break
                        assert len(checkpoint_filenames) == 1, f"Found {len(checkpoint_filenames)} checkpoint files: {checkpoint_filenames}"
                        checkpoint_filename = checkpoint_filenames[0]
                        print(f"Found checkpoint file: {checkpoint_filename}. Renaming to model.tar")
                        os.rename(os.path.join(results_dir, rel_runpath, checkpoint_filename), os.path.join(results_dir, rel_runpath, 'model.tar'))


def cleanup_logfiles(results_dir):
    run_dirs = [os.path.join(results_dir, p) for p in os.listdir(results_dir)]
    for run_dir in run_dirs:
        pid_dirs = [os.path.join(run_dir, p) for p in os.listdir(run_dir) if p != 'latest']
        for pid_dir in pid_dirs:
            plogger = FileWriter(rootdir=pid_dir,
                                 seeds=[0], symlink_to_latest=False, no_setup=True
                                 )
            remove_duplicates(plogger)
            if os.path.exists(os.path.join(pid_dir, 'model.tar')):
                plogger.mark_completed()
            else:
                # check if there is a model_*.tar file
                checkpoint_filenames = []
                for file in os.listdir(pid_dir):
                    if file.endswith(".tar"):
                        checkpoint_filenames.append(file)
                assert len(
                    checkpoint_filenames) <= 1, f"Found {len(checkpoint_filenames)} checkpoint files: {checkpoint_filenames}"
                if checkpoint_filenames:
                    last_saved_update = int(checkpoint_filenames[0].split('_')[1].split('.')[0])
                    plogger.delete_after_update(last_saved_update)
                else:
                    print(f"Found unsuccessfull run without checkpoint: {pid_dir}. Deleting pid directory.")
                    # delete the pid directory
                    shutil.rmtree(pid_dir)


def update_checkpoint_states(results_dir):
    run_dirs = [os.path.join(results_dir, p) for p in os.listdir(results_dir)]

    for run_dir in run_dirs:
        pid_dirs = [os.path.join(run_dir, p) for p in os.listdir(run_dir) if p != 'latest']
        for pid_dir in pid_dirs:
            plogger = FileWriter(rootdir=pid_dir,
                                 seeds=[0], symlink_to_latest=False, no_setup=True
                                 )
            checkpoint_filenames = []
            for file in os.listdir(os.path.expandvars(os.path.expanduser(plogger.basepath))):
                if file.endswith(".tar"):
                    checkpoint_filenames.append(file)
                    break
            if not checkpoint_filenames:
                continue
            assert len(
                checkpoint_filenames) == 1, f"{plogger.basepath}: Found {len(checkpoint_filenames)} checkpoint files: {checkpoint_filenames}"
            file = checkpoint_filenames[0]
            checkpoint_path = os.path.expandvars(os.path.expanduser(plogger.basepath + '/' + file))
            if file == 'model.tar':
                num_updates = 0
            else:
                num_updates = int(file.split('_')[1].split('.')[0])
            print(
                f'{plogger.basepath}: Checkpoint found at update {num_updates}. Recovering level sampler state from logs\n')
            seeds, seed_scores, seed_staleness, unseen_seed_weights, next_seed_index = recover_level_sampler(plogger,
                                                                                                           num_updates - 1)
            print(f'Updating Checkpoint States\n')
            checkpoint = torch.load(checkpoint_path, map_location=lambda storage, loc: storage)
            checkpoint['seeds'] = seeds
            checkpoint['seed_scores'] = seed_scores
            checkpoint['seed_staleness'] = seed_staleness
            checkpoint['unseen_seed_weights'] = unseen_seed_weights
            checkpoint['next_seed_index'] = next_seed_index
            torch.save(
                checkpoint,
                checkpoint_path,
            )


def remove_duplicates(plogger) -> None:
    # remove duplicates from the log file, only used in an old version of the code where duplicates were an issue.
    if plogger.num_duplicates == 0:
        return

    for path in [plogger.paths["level_weights"], plogger.paths["level_value_loss"], plogger.paths["level_instance_value_loss"],
                    plogger.paths["level_returns"], plogger.paths["instance_pred_entropy"], plogger.paths["instance_pred_accuracy"],
                    plogger.paths["instance_pred_precision"]]:
        with open(path, "r") as level_file:
            reader = csv.reader(level_file)
            lines = list(reader)
            header = lines[0]
            new_lines = []
            for row_id in reversed(range(1, len(lines))):
                if lines[row_id] == header:
                    new_lines.extend(lines[row_id:])
                    break
        if len(new_lines) > 0:
            overwrite_file(path, new_lines)

    # in the log file, identify where step[row] < steps[row-1] and remove all rows above, starting from row - 1 but
    # not including the header
    with open(plogger.paths["logs"], "r") as logfile:
        reader = csv.DictReader(logfile)
        lines = list(reader)
        new_lines = []
        for row_id in reversed(range(2, len(lines))):
            if int(lines[row_id]['step']) < int(lines[row_id-1]['step']):
                new_lines.extend(lines[row_id:])
                break

        tick = 0
        for row_id in range(len(new_lines)):
            new_lines[row_id]['# _tick'] = str(int(tick))
            tick += 1

    if new_lines:
        overwrite_file(plogger.paths["logs"], new_lines)

    # in the final test files remove all duplicated header rows
    with open(plogger.paths["final_test_eval"], "r") as finaltestfile:
        reader = csv.reader(finaltestfile)
        lines = list(reader)
        header = lines[0]
    if lines[-1] != header:
        new_lines = [header, lines[-1]]
        overwrite_file(plogger.paths["final_test_eval"], new_lines)
    elif len(lines) > 1:
        new_lines = [header]
        overwrite_file(plogger.paths["final_test_eval"], new_lines)

def recover_level_sampler(plogger, num_updates):
    # recover the level sampler from the log file, only used in an old version of the code where the level sampler state was not saved
    level_replay_strategy = plogger.metadata["args"]["level_replay_strategy"]
    if level_replay_strategy not in ["value_l1", "random"]:
        raise NotImplementedError

    with open(plogger.paths["level_weights"], "r") as levelweightsfile:
        level_weights_reader = csv.reader(levelweightsfile)
        header = next(level_weights_reader)
        # convert header to numpy array:
        if header[0] == "#":
            header = header[1:]
        elif header[0].startswith("#"):
            header[0] = header[0].split("# ")[1]
        seeds = np.array(header, dtype=np.int32)
        level_weights = np.array(list(level_weights_reader)[num_updates], dtype=np.float64)
    # unseen_seed_weights should be a boolean array that is 1 at indices i where level_weigthts[i] == 0
    if level_replay_strategy == "value_l1": #only implemented for value_l1
        assert plogger.metadata["args"]["level_replay_score_transform"] == "rank"
        with open(plogger.paths["level_value_loss"], "r") as levelvaluelossfile:
            level_scores_reader = csv.reader(levelvaluelossfile)
            scores = list(level_scores_reader)[1:]
        seed_scores = np.full(len(seeds), np.nan, dtype=np.float)
        for row_id, score_row in enumerate(scores):
            if row_id > num_updates != -1:
                break
            for id, score in enumerate(score_row):
                if score != "nan":
                    seed_scores[id] = float(score)
        # replace nan with 0
        seed_scores = np.nan_to_num(seed_scores)
        score_temperature = plogger.metadata["args"]["level_replay_temperature"]
        temp = np.flip(seed_scores.argsort())
        ranks = np.empty_like(temp)
        ranks[temp] = np.arange(len(temp)) + 1
        score_weights = 1 / ranks ** (1. / score_temperature)
        staleness_coeff = plogger.metadata["args"]["staleness_coef"]
        staleness_weights = (level_weights - (1 - staleness_coeff) * score_weights)/staleness_coeff
        seed_staleness = staleness_weights.argsort()
        unseen_seed_weights = np.isclose(seed_scores, 0).astype(np.float64)
    elif level_replay_strategy == "random":
        seed_scores = np.zeros_like(level_weights)
        seed_staleness = np.zeros_like(level_weights)
        unseen_seed_weights = np.isclose(level_weights, 0).astype(np.float64)
    else:
        raise NotImplementedError

    if np.any(unseen_seed_weights):
        next_seed_index = np.where(unseen_seed_weights == 1)[0][0]
    else:
        next_seed_index = 0

    return seeds, seed_scores, seed_staleness, unseen_seed_weights, next_seed_index


def clone_dirs(src_dir, dest_dirs):
    for dest_dir in dest_dirs:
        shutil.copytree(src_dir, dest_dir)


def arg_product(arguments):
    return list(dict(zip(arguments.keys(), values)) for values in itertools.product(*arguments.values()))

def update_args(args):

    if 'level_replay_secondary_strategy_coef_end' not in args:
        args['level_replay_secondary_strategy_coef_end'] = 0.0
    if 'level_replay_secondary_temperature' not in args:
        args['level_replay_secondary_temperature'] = 1.0
    if 'level_replay_secondary_strategy' not in args:
        args['level_replay_secondary_strategy'] = 'off'
    if 'level_replay_secondary_strategy_fraction_end' not in args:
        if 'level_replay_secondary_strategy_coef_update_fraction' in args:
            args['level_replay_secondary_strategy_fraction_end'] = args['level_replay_secondary_strategy_coef_update_fraction']
        else:
            args['level_replay_secondary_strategy_fraction_end'] = 0.0
    if 'level_replay_secondary_strategy_fraction_start' not in args:
        args['level_replay_secondary_strategy_fraction_start'] = 0.0

def set_xpid(args):

    xpid = f"e-{args['env_name']}_" \
            f"s1-{args['level_replay_strategy']}_" \
            f"s2-{args['level_replay_secondary_strategy']}_" \
            f"bf-{args['level_replay_secondary_strategy_coef_end']}_" \
            f"l2-{args['level_replay_secondary_temperature']}_" \
            f"fs-{args['level_replay_secondary_strategy_fraction_start']}_" \
            f"fe-{args['level_replay_secondary_strategy_fraction_end']}_" \
            f"s-{args['seed']}"
    return xpid


def set_logdir(args, base_dir=None):

    if base_dir is None:
        base_dir = args['log_dir']
    logdir = os.path.join(base_dir,
                          f"e-{args['env_name']}_"
                          f"s1-{args['level_replay_strategy']}_"
                          f"s2-{args['level_replay_secondary_strategy']}_"
                          f"bf-{args['level_replay_secondary_strategy_coef_end']}_"
                          f"l2-{args['level_replay_secondary_temperature']}_"
                          f"fs-{args['level_replay_secondary_strategy_fraction_start']}_"
                          f"fe-{args['level_replay_secondary_strategy_fraction_end']}"
                          )
    return logdir


def set_bootstrap_dir(args):

    bootstrap_log_dir = f"e-{args['env_name']}_" \
                        f"s1-{args['level_replay_strategy']}_" \
                        f"fs-{args['level_replay_secondary_strategy_fraction_start']}_baserun"

    bootstrap_pid =     f"e-{args['env_name']}_" \
                        f"s1-{args['level_replay_strategy']}_" \
                        f"fs-{args['level_replay_secondary_strategy_fraction_start']}_" \
                        f"s-{args['seed']}"

    bootstrap_dir = os.path.join(bootstrap_log_dir, bootstrap_pid)
    return bootstrap_dir


def create_exp_list(sweep_args: Dict[str, str], args: Dict[str, Any]):

    for arg in sweep_args:
        assert isinstance(sweep_args[arg], str)
        assert arg in args
        sweep_args[arg] = sweep_args[arg].split(",")

    sweep_dictionaries = arg_product(sweep_args)

    exp_dictionaries = []
    for exp in sweep_dictionaries:
        exp_d = copy.deepcopy(args)
        exp_d.update(exp)
        exp_dictionaries.append(exp_d)

    return exp_dictionaries


def args2string(dict_args: Dict[str, Any]):
    string = []
    for key in dict_args:
        if key.endswith("_SWEEP"):
            continue
        if isinstance(dict_args[key], bool):
            if dict_args[key]:
                string.append("--" + key)
            else:
                continue
        elif dict_args[key] is None:
            continue
        else:
            string.append("--" + key + "=" + str(dict_args[key]))
    return " ".join(string)


##########################

# Rename the xpids and logdirs of the experiments in the result directory according to convention
def rename_pids(result_dir):
    run_dirs = [os.path.join(result_dir, p) for p in os.listdir(result_dir)]
    for run_dir in run_dirs:
        pid_dirs = [os.path.join(run_dir, p) for p in os.listdir(run_dir) if p != 'latest']
        for pid_dir in pid_dirs:
            plogger = FileWriter(rootdir=pid_dir,
                                 seeds=[0], symlink_to_latest=False, no_setup=True
                                 )
            meta = plogger.metadata
            args = meta['args']
            update_args(args)
            new_pid = set_xpid(args)
            new_logdir = set_logdir(args, base_dir=result_dir)
            if pid_dir.endswith('_bkup'):
                new_logdir += '_baserun'
            meta['xpid'] = new_pid
            args['xpid'] = new_pid
            args['log_dir'] = new_logdir
            plogger.metadata = meta
            plogger.close(successful=plogger.completed)
            if not os.path.exists(new_logdir):
                os.makedirs(new_logdir)
            if os.path.exists(os.path.join(new_logdir, new_pid)):
                print(f"WARNING - run_pid {pid_dir} : {new_pid} already exists in {new_logdir}. Adding _extra")
                new_pid += '_extra'
            shutil.move(pid_dir, os.path.join(new_logdir, new_pid))
        shutil.rmtree(run_dir)


def rename_baserun_pids(result_dir):
    run_dirs = [os.path.join(result_dir, p) for p in os.listdir(result_dir)]
    for run_dir in run_dirs:
        if not run_dir.endswith('_baserun'):
            continue
        pid_dirs = [os.path.join(run_dir, p) for p in os.listdir(run_dir) if p != 'latest']
        for pid_dir in pid_dirs:
            plogger = FileWriter(rootdir=pid_dir,
                                 seeds=[0], symlink_to_latest=False, no_setup=True
                                 )
            meta = plogger.metadata
            args = meta['args']
            update_args(args)

            new_logdir = f"e-{args['env_name']}_" \
                                f"s1-{args['level_replay_strategy']}_" \
                                f"fs-{args['level_replay_secondary_strategy_fraction_start']}_baserun"
            new_logdir = os.path.join(result_dir, new_logdir)
            new_pid = f"e-{args['env_name']}_" \
                            f"s1-{args['level_replay_strategy']}_" \
                            f"fs-{args['level_replay_secondary_strategy_fraction_start']}_" \
                            f"s-{args['seed']}"
            meta['xpid'] = new_pid
            args['xpid'] = new_pid
            args['log_dir'] = new_logdir
            plogger.metadata = meta
            plogger.close(successful=plogger.completed)
            if not os.path.exists(new_logdir):
                os.makedirs(new_logdir)
            if os.path.exists(os.path.join(new_logdir, new_pid)):
                print(f"WARNING - run_pid {pid_dir} : {new_pid} already exists in {new_logdir}. Adding _extra")
                new_pid += '_extra'
            shutil.move(pid_dir, os.path.join(new_logdir, new_pid))
        shutil.rmtree(run_dir)


def create_full_exp_file(exp_dir: str,
                    filename: str,
                    args: Dict[str, Any],
                    setup_xpid: bool = False,
                    setup_logdir: bool = False,
                    bootstrap: bool = False):
    # create a list of strings with all possible combinations of sweep_args and the shared fixed args
    # each string is a command line argument
    # write this list to a file in the experiment directory

    sweep_args = {}
    for arg in list(args.keys()):
        if arg.endswith("_SWEEP"):
            if args[arg] is not None:
                arg_sw = arg.replace("_SWEEP", "")
                sweep_args[arg_sw] = args[arg]
            del args[arg]

    exp_dictionaries = create_exp_list(sweep_args, args)

    if setup_logdir:
        for exp in exp_dictionaries:
            exp['log_dir'] = set_logdir(exp)

    if setup_xpid:
        for exp in exp_dictionaries:
            exp['xpid'] = set_xpid(exp)

    if bootstrap:
        for exp in exp_dictionaries:
            exp['bootstrap_from_dir'] = set_bootstrap_dir(exp)

    exp_strings = []
    for exp in exp_dictionaries:
        string = args2string(exp)
        exp_strings.append(string)

    with open(os.path.join(exp_dir, filename), "w") as exp_file:
        exp_file.write("\n".join(exp_strings))

    return exp_strings


def create_todo_exp_file(input_exp_file=None,
                         result_dir=None,
                         slurm_exp_dir=None,
                         server_config_file='server_config.json',
                         to_server=False,
                         keep_original_split=False,
                         verbose=True):

    repo_dir = os.path.dirname(os.path.dirname(os.path.abspath(sys.argv[0])))
    if slurm_exp_dir is None:
        slurm_exp_dir = os.path.join(repo_dir, 'slurm')
    if result_dir is None:
        result_dir = os.path.join(repo_dir, 'results')

    server_config = json.load(open(os.path.join(slurm_exp_dir, server_config_file), 'r'))
    server_config = [conf for conf in server_config if conf['use']]

    if to_server:
        if input_exp_file is None:
            input_exp_file = [conf['exp_file'] for conf in server_config]
        else:
            input_exp_file = input_exp_file.split(",")
        output_exp_file = [conf['exp_file'] for conf in server_config]
        result_dir_out = [conf['result_dir'] for conf in server_config]
        if keep_original_split:
            assert input_exp_file == output_exp_file, \
                "If keep_original_split is True, input_exp_file and output_exp_file must be the same"
    else:
        assert input_exp_file is not None
        input_exp_file = input_exp_file.split(",")
        assert len(input_exp_file) == 1
        assert not keep_original_split, "keep_original_split is only supported when to_server is False"
        output_exp_file = [input_exp_file[0].split('.txt')[0] + '_todo.txt']
        result_dir_out = result_dir.split(',')

    exp_strings = defaultdict(list)
    for i, exp_file in enumerate(input_exp_file):
        exp_strings[exp_file] += open(exp_file, 'r').readlines()
        if verbose:
            print(f"Added {len(exp_strings[exp_file])} experiments from {exp_file}")

    # remove completed runs
    todo_strings = defaultdict(list)
    for i, exp_file in enumerate(exp_strings):
        for string in exp_strings[exp_file]:
            args = parser.parse_args(shlex.split(string)).__dict__
            log_dirname = os.path.basename(args['log_dir'])
            pid_dir = os.path.join(result_dir, log_dirname, args['xpid'])
            new_logdir = os.path.join(result_dir_out[i], log_dirname)
            args['log_dir'] = new_logdir
            new_string = args2string(args)
            if os.path.exists(pid_dir):
                print(f"Found {pid_dir}")
                plogger = FileWriter(rootdir=pid_dir, seeds=[0], symlink_to_latest=False, no_setup=True)
                if not plogger.completed:
                    todo_strings[exp_file].append(new_string)
            else:
                todo_strings[exp_file].append(new_string)

    if not to_server:
        assert len(output_exp_file) == 1
        with open(os.path.join(slurm_exp_dir, output_exp_file[0]), 'w') as f:
            for exp_file in todo_strings:
                for line in todo_strings[exp_file]:
                    f.write(line + '\n')
    else:
        # split experiments across servers
        if keep_original_split:
            for i, exp_file in enumerate(input_exp_file):
                with open(os.path.join(slurm_exp_dir, output_exp_file[i]), 'w') as f:
                    for line in todo_strings[exp_file]:
                        f.write(line + '\n')
                if verbose:
                    print(f"Server {server_config[i]['server']} gets {len(todo_strings[exp_file])} experiments")
        else:
            all_exp_strings = []
            for exp_file in todo_strings:
                all_exp_strings += todo_strings[exp_file]
            num_exp = len(all_exp_strings)
            z = sum([conf['split_weight'] for conf in server_config])
            start = 0
            for i, conf in enumerate(server_config):
                if i == len(server_config) - 1:
                    end = num_exp
                else:
                    end = start + int(conf['split_weight'] / z * num_exp)
                with open(os.path.join(slurm_exp_dir, output_exp_file[i]), 'w') as f:
                    for line in all_exp_strings[start:end]:
                        f.write(line + '\n')
                if verbose:
                    print(f"Server {conf['server']} gets {end - start} experiments")
                start = end

    return todo_strings

if __name__ == "__main__":

    #args
    # --num_processes=64 --level_replay_strategy=value_l1
    # --level_replay_score_transform=rank --level_replay_temperature=0.1 --staleness_coef=0.1
    # --instance_predictor --instance_predictor_hidden_size=-1 --level_replay_secondary_strategy=instance_pred_log_prob
    # --level_replay_secondary_strategy_fraction_start=0.5 --checkpoint --log_dir=~/procgen/level-replay/results

    parser.add_argument(
        "--level_replay_secondary_temperature_SWEEP",
        type=str,
        default='0.1,0.5,1.0,2.0',
        help="SWEEP PARAM: Level replay scoring strategy")
    parser.add_argument(
        "--level_replay_secondary_strategy_coef_end_SWEEP",
        type=str,
        default='0.25,0.5,1.0',
        help="SWEEP PARAM: Level replay coefficient balancing primary and secondary strategies, end value")
    parser.add_argument(
        '--env_name_SWEEP',
        type=str,
        default='bigfish,heist,climber,caveflyer,jumper,fruitbot,plunder,coinrun,ninja,leaper,'
                'maze,miner,dodgeball,starpilot,chaser,bossfight',
        help='SWEEP PARAM: environment to train on')
    parser.add_argument(
        '--seed_SWEEP',
        type=str,
        default='8,88,888,8888',
        help='SWEEP PARAM: random seed')

    args = parser.parse_args()
    # rename_pids('/home/francelico/dev/PhD/procgen/results/results_cp')
    # rename_baserun_pids('/home/francelico/dev/PhD/procgen/results/results_cp')
    create_full_exp_file(os.path.expandvars(os.path.expanduser('~/dev/PhD/procgen/level-replay/slurm')),
                    'test_experiment.txt',
                    args.__dict__,
                    setup_xpid=True,
                    setup_logdir=True,
                    bootstrap=True)
    create_todo_exp_file(input_exp_file='test_experiment.txt',
                         to_server=False,
                         result_dir='/home/francelico/dev/PhD/procgen/results/results_cp',
                         keep_original_split=False)

    sys.exit(0)

