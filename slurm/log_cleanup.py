import sys
import os
import re
import shutil
import csv
import torch
import numpy as np

from level_replay.file_writer import FileWriter, overwrite_file


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

sys.exit(0)

