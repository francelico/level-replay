# Copyright (c) Facebook, Inc. and its affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

import copy
import csv
import datetime
import json
import logging
import os
import time
from typing import Dict, List, Any
import numpy as np


def gather_metadata() -> Dict:
    date_start = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S.%f")
    # Gathering git metadata.
    try:
        import git

        try:
            repo = git.Repo(search_parent_directories=True)
            git_sha = repo.commit().hexsha
            git_data = dict(
                commit=git_sha,
                branch=None if repo.head.is_detached else repo.active_branch.name,
                is_dirty=repo.is_dirty(),
                path=repo.git_dir,
            )
        except git.InvalidGitRepositoryError:
            git_data = None
    except ImportError:
        git_data = None
    # Gathering slurm metadata.
    if "SLURM_JOB_ID" in os.environ:
        slurm_env_keys = [k for k in os.environ if k.startswith("SLURM")]
        slurm_data = {}
        for k in slurm_env_keys:
            d_key = k.replace("SLURM_", "").replace("SLURMD_", "").lower()
            slurm_data[d_key] = os.environ[k]
    else:
        slurm_data = None
    return dict(
        date_start=date_start,
        date_end=None,
        successful=False,
        git=git_data,
        slurm=slurm_data,
        env=os.environ.copy(),
    )


def overwrite_file(path, new_lines):
    with open(path, "w") as file:
        if isinstance(new_lines[0], list):
            writer = csv.writer(file)
            writer.writerows(new_lines)
        elif isinstance(new_lines[0], dict):
            writer = csv.DictWriter(file, fieldnames=new_lines[0].keys())
            writer.writeheader()
            writer.writerows(new_lines)
        else:
            raise ValueError(f"Unsupported type {type(new_lines[0])} for new_lines")
        file.flush()


class FileWriter:
    def __init__(
        self,
        xpid: str = None,
        xp_args: dict = None,
        rootdir: str = "~/logs",
        symlink_to_latest: bool = True,
        seeds=None,
        no_setup: bool = False,
    ):
        if not xpid:
            # Make unique id.
            xpid = "{proc}_{unixtime}".format(
                proc=os.getpid(), unixtime=int(time.time())
            )
        self.xpid = xpid
        self._tick = 0
        self.no_setup = no_setup

        formatter = logging.Formatter("%(message)s")
        self._logger = logging.getLogger("logs/out")
        if self._logger.hasHandlers():
            self._logger.handlers.clear()

        # To stdout handler.logging.streamhandler()
        shandle = logging.StreamHandler()
        shandle.setFormatter(formatter)
        self._logger.addHandler(shandle)
        if self.no_setup:
            self._logger.setLevel(logging.WARNING)
        else:
            self._logger.setLevel(logging.INFO)

        rootdir = os.path.expandvars(os.path.expanduser(rootdir))
        # To file handler.
        if not self.no_setup:
            self.basepath = os.path.join(rootdir, self.xpid)
        else:
            self.basepath = os.path.join(rootdir)
        if not os.path.exists(self.basepath):
            if self.no_setup:
                raise FileNotFoundError("Log directory not found in read-only mode: %s" % self.basepath)
            else:
                self._logger.info("Creating log directory: %s", self.basepath)
                os.makedirs(self.basepath, exist_ok=True)
        else:
            self._logger.info("Found log directory: %s", self.basepath)

        if symlink_to_latest and not self.no_setup:
            # Add 'latest' as symlink unless it exists and is no symlink.
            symlink = os.path.join(rootdir, "latest")
            try:
                if os.path.islink(symlink):
                    os.remove(symlink)
                if not os.path.exists(symlink):
                    os.symlink(self.basepath, symlink)
                    self._logger.info("Symlinked log directory: %s", symlink)
            except OSError:
                # os.remove() or os.symlink() raced. Don't do anything.
                pass

        self.paths = dict(
            msg="{base}/out.log".format(base=self.basepath),
            logs="{base}/logs.csv".format(base=self.basepath),
            fields="{base}/fields.csv".format(base=self.basepath),
            meta="{base}/meta.json".format(base=self.basepath),
            level_weights="{base}/level_weights.csv".format(base=self.basepath),
            level_value_loss="{base}/level_value_loss.csv".format(base=self.basepath),
            level_instance_value_loss="{base}/level_instance_value_loss.csv".format(base=self.basepath),
            level_returns="{base}/level_returns.csv".format(base=self.basepath),
            instance_pred_entropy="{base}/instance_pred_entropy.csv".format(base=self.basepath),
            instance_pred_accuracy="{base}/instance_pred_accuracy.csv".format(base=self.basepath),
            instance_pred_log_prob="{base}/instance_pred_log_prob.csv".format(base=self.basepath),
            instance_pred_prob="{base}/instance_pred_prob.csv".format(base=self.basepath),
            instance_pred_precision="{base}/instance_pred_precision.csv".format(base=self.basepath),
            instance_pred_recall="{base}/instance_pred_recall.csv".format(base=self.basepath),
            instance_pred_f1="{base}/instance_pred_f1.csv".format(base=self.basepath),
            final_test_eval="{base}/final_test_eval.csv".format(base=self.basepath)
        )

        self._logger.info("Saving messages to %s", self.paths["msg"])
        if os.path.exists(self.paths["msg"]) and not self.no_setup:
            self._logger.warning(
                "Path to message file already exists. " "New data will be appended."
            )

        fhandle = logging.FileHandler(self.paths["msg"])
        fhandle.setFormatter(formatter)
        self._logger.addHandler(fhandle)

        if not self.no_setup:
            self._logger.info("Saving logs data to %s", self.paths["logs"])
            self._logger.info("Saving logs' fields to %s", self.paths["fields"])
        self.fieldnames = ["_tick", "_time"]
        self.final_test_eval_fieldnames = ['num_test_seeds', 'mean_episode_return', 'median_episode_return']
        if os.path.exists(self.paths["logs"]):
            if not self.no_setup:
                self._logger.warning(
                    "Path to log file already exists. " "New data will be appended."
                )
            # Override default fieldnames.
            with open(self.paths["fields"], "r") as csvfile:
                reader = csv.reader(csvfile)
                lines = list(reader)
                if len(lines) > 0:
                    self.fieldnames = lines[-1]
            # Override default tick: use the last tick from the logs file plus 1.
            with open(self.paths["logs"], "r") as csvfile:
                reader = csv.reader(csvfile)
                lines = list(reader)
                # Need at least two lines in order to read the last tick:
                # the first is the csv header and the second is the first line
                # of data.
                if len(lines) > 1:
                    self._tick = int(lines[-1][0]) + 1

        self._fieldfile = open(self.paths["fields"], "a")
        self._fieldwriter = csv.writer(self._fieldfile)
        self._logfile = open(self.paths["logs"], "a")
        self._logwriter = csv.DictWriter(self._logfile, fieldnames=self.fieldnames)
        self._levelweightsfile = open(self.paths["level_weights"], "a")
        self._levelweightswriter = csv.writer(self._levelweightsfile)
        self._finaltestfile = open(self.paths["final_test_eval"], "a")
        self._finaltestwriter = csv.DictWriter(self._finaltestfile, fieldnames=self.final_test_eval_fieldnames)
        self._levelvaluelossfile = open(self.paths["level_value_loss"], "a")
        self._levelvaluelosswriter = csv.writer(self._levelvaluelossfile)
        self._levelinstancevaluelossfile = open(self.paths["level_instance_value_loss"], "a")
        self._levelinstancevaluelosswriter = csv.writer(self._levelinstancevaluelossfile)
        self._levelreturnsfile = open(self.paths["level_returns"], "a")
        self._levelreturnswriter = csv.writer(self._levelreturnsfile)
        self._instancepredentropyfile = open(self.paths["instance_pred_entropy"], "a")
        self._instancepredentropywriter = csv.writer(self._instancepredentropyfile)
        self._instancepredaccuracyfile = open(self.paths["instance_pred_accuracy"], "a")
        self._instancepredaccuracywriter = csv.writer(self._instancepredaccuracyfile)
        self._instancepredlogprobfile = open(self.paths["instance_pred_log_prob"], "a")
        self._instancepredlogprobwriter = csv.writer(self._instancepredlogprobfile)
        self._instancepredprobfile = open(self.paths["instance_pred_prob"], "a")
        self._instancepredprobwriter = csv.writer(self._instancepredprobfile)
        self._instancepredprecisionfile = open(self.paths["instance_pred_precision"], "a")
        self._instancepredprecisionwriter = csv.writer(self._instancepredprecisionfile)
        self._instancepredrecallfile = open(self.paths["instance_pred_recall"], "a")
        self._instancepredrecallwriter = csv.writer(self._instancepredrecallfile)
        self._instancepredf1file = open(self.paths["instance_pred_f1"], "a")
        self._instancepredf1writer = csv.writer(self._instancepredf1file)

        if not self.no_setup:
            self.setup(xp_args, seeds)
        else:
            self.metadata = json.load(open(self.paths["meta"], "r"))
            self.xpid = self.metadata["xpid"]
            with open(self.paths["level_weights"], "r") as levelweightsfile:
                level_weights_reader = csv.reader(levelweightsfile)
                header = next(level_weights_reader)
                # convert header to numpy array:
                if header[0] == "#":
                    header = header[1:]
                elif header[0].startswith("#"):
                    header[0] = header[0].split("# ")[1]
                seeds = np.array(header, dtype=np.int32)
            self.seeds = [str(seed) for seed in seeds]
            self.read_data()

    def setup(self, xp_args, seeds):

        self.seeds = [str(seed) for seed in seeds]

        # Metadata gathering.
        if xp_args is None:
            xp_args = {}
        self.metadata = gather_metadata()
        # We need to copy the args, otherwise when we close the file writer
        # (and rewrite the args) we might have non-serializable objects (or
        # other unwanted side-effects).
        self.metadata["args"] = copy.deepcopy(xp_args)
        self.metadata["xpid"] = self.xpid

        self._logger.info("Saving arguments to %s", self.paths["meta"])
        if os.path.exists(self.paths["meta"]):
            self._logger.warning(
                "Path to meta file already exists. " "Will check if arguments match and if they do will overwritte."
            )
            with open(self.paths["meta"], "r") as f:
                old_meta = json.load(f)
                if not xp_args["override_previous_args"]:
                    for arg in old_meta["args"]:
                        if arg not in ["log_dir", "verbose"]:
                            assert old_meta["args"][arg] == self.metadata["args"][arg], (
                                "Argument {} changed from {} to {}".format(
                                    arg, old_meta["args"][arg], self.metadata["args"][arg]
                                )
                            )
                self.metadata["previous_slurm"] = []
                if "previous_slurm" in old_meta and isinstance(old_meta["previous_slurm"], list):
                    self.metadata["previous_slurm"].extend(old_meta["previous_slurm"])
                if "slurm" in old_meta and old_meta["slurm"] is not None:
                    self.metadata["previous_slurm"].append(old_meta["slurm"])
                if "successful" in old_meta:
                    self.metadata["successful"] = old_meta["successful"]

        self._save_metadata()

        # only write the header if the file is empty
        if self._levelweightsfile.tell() == 0:
            self._levelweightsfile.write("# %s\n" % ",".join(self.seeds))
            self._levelweightsfile.flush()
        if self._levelvaluelossfile.tell() == 0:
            self._levelvaluelossfile.write("# %s\n" % ",".join(self.seeds))
            self._levelvaluelossfile.flush()
        if self._levelinstancevaluelossfile.tell() == 0:
            self._levelinstancevaluelossfile.write("# %s\n" % ",".join(self.seeds))
            self._levelinstancevaluelossfile.flush()
        if self._levelreturnsfile.tell() == 0:
            self._levelreturnsfile.write("# %s\n" % ",".join(self.seeds))
            self._levelreturnsfile.flush()
        if self._instancepredentropyfile.tell() == 0:
            self._instancepredentropyfile.write("# %s\n" % ",".join(self.seeds))
            self._instancepredentropyfile.flush()
        if self._instancepredaccuracyfile.tell() == 0:
            self._instancepredaccuracyfile.write("# %s\n" % ",".join(self.seeds))
            self._instancepredaccuracyfile.flush()
        if self._instancepredlogprobfile.tell() == 0:
            self._instancepredprecisionfile.write("# %s\n" % ",".join(self.seeds))
            self._instancepredlogprobfile.flush()
        if self._instancepredprobfile.tell() == 0:
            self._instancepredprobfile.write("# %s\n" % ",".join(self.seeds))
            self._instancepredprobfile.flush()
        if self._instancepredprecisionfile.tell() == 0:
            self._instancepredprecisionfile.write("# %s\n" % ",".join(self.seeds))
            self._instancepredprecisionfile.flush()
        if self._instancepredrecallfile.tell() == 0:
            self._instancepredrecallfile.write("# %s\n" % ",".join(self.seeds))
            self._instancepredrecallfile.flush()
        if self._instancepredf1file.tell() == 0:
            self._instancepredf1file.write("# %s\n" % ",".join(self.seeds))
            self._instancepredf1file.flush()

        if self._finaltestfile.tell() == 0:
            self._finaltestwriter.writeheader()
            self._finaltestfile.flush()

    def log(self, to_log: Dict, tick: int = None, verbose: bool = False) -> None:
        if tick is not None:
            raise NotImplementedError
        else:
            to_log["_tick"] = self._tick
            self._tick += 1
        to_log["_time"] = time.time()

        old_len = len(self.fieldnames)
        for k in to_log:
            if k not in self.fieldnames:
                self.fieldnames.append(k)
        if old_len != len(self.fieldnames):
            self._fieldwriter.writerow(self.fieldnames)
            self._logger.info("Updated log fields: %s", self.fieldnames)

        if to_log["_tick"] == 0:
            self._logfile.write("# %s\n" % ",".join(self.fieldnames))

        if verbose:
            self._logger.info(
                "LOG | %s",
                ", ".join(["{}: {}".format(k, to_log[k]) for k in sorted(to_log)]),
            )

        self._logwriter.writerow(to_log)
        self._logfile.flush()

    def log_level_weights(self, weights):
        self._levelweightswriter.writerow(weights)
        self._levelweightsfile.flush()

    def log_level_value_loss(self, value_loss):
        self._levelvaluelosswriter.writerow(value_loss)
        self._levelvaluelossfile.flush()

    def log_level_instance_value_loss(self, instance_value_loss):
        self._levelinstancevaluelosswriter.writerow(instance_value_loss)
        self._levelinstancevaluelossfile.flush()

    def log_level_returns(self, returns):
        self._levelreturnswriter.writerow(returns)
        self._levelreturnsfile.flush()

    def log_instance_pred_log_prob(self, instance_pred_log_prob):
        self._instancepredlogprobwriter.writerow(instance_pred_log_prob)
        self._instancepredlogprobfile.flush()

    def log_instance_pred_prob(self, instance_pred_prob):
        self._instancepredprobwriter.writerow(instance_pred_prob)
        self._instancepredprobfile.flush()

    def log_instance_pred_accuracy(self, instance_pred_accuracy):
        self._instancepredaccuracywriter.writerow(instance_pred_accuracy)
        self._instancepredaccuracyfile.flush()

    def log_instance_pred_entropy(self, instance_pred_entropy):
        self._instancepredentropywriter.writerow(instance_pred_entropy)
        self._instancepredentropyfile.flush()

    def log_instance_pred_precision(self, instance_pred_precision):
        self._instancepredprecisionwriter.writerow(instance_pred_precision)
        self._instancepredprecisionfile.flush()

    def log_instance_pred_recall(self, instance_pred_recall):
        self._instancepredrecallwriter.writerow(instance_pred_recall)
        self._instancepredrecallfile.flush()

    def log_instance_pred_f1(self, instance_pred_f1):
        self._instancepredf1writer.writerow(instance_pred_f1)
        self._instancepredf1file.flush()

    def log_final_test_eval(self, to_log):
        self._finaltestwriter.writerow(to_log)
        self._finaltestfile.flush()

    def close(self, successful: bool = True) -> None:
        self.metadata["date_end"] = datetime.datetime.now().strftime(
            "%Y-%m-%d %H:%M:%S.%f"
        )
        self.metadata["successful"] = successful
        self._save_metadata()

        for f in [self._logfile, self._fieldfile]:
            f.close()

    def _save_metadata(self) -> None:
        with open(self.paths["meta"], "w") as jsonfile:
            json.dump(self.metadata, jsonfile, indent=4, sort_keys=True)

    def mark_completed(self) -> bool:
        with open(self.paths["final_test_eval"], "r") as final_test_eval_file:
            reader = csv.reader(final_test_eval_file)
            lines = list(reader)
        header = lines[0]
        assert header == self.final_test_eval_fieldnames
        if len(lines) > 1 and lines[-1] != header:
            self.close(successful=True)
            return True
        else:
            self.close(successful=False)
            return False

    def delete_after_update(self, num_update):
        "Erase all logs that were written after the given update number"
        with open(self.paths["logs"], "r+") as logfile:
            reader = csv.DictReader(logfile)
            lines = list(reader)
            new_lines = []
            assert int(lines[-1]["# _tick"]) >= num_update
            for row in lines:
                if int(row["# _tick"]) <= num_update:
                    new_lines.append(row)
                else:
                    break
            logfile.seek(0)
            logfile.truncate()
            writer = csv.DictWriter(logfile, fieldnames=lines[0].keys())
            writer.writeheader()
            writer.writerows(new_lines)

        assert int(new_lines[-1]["# _tick"]) == len(new_lines) - 1
        self._tick = num_update + 1

        for path in [self.paths["level_weights"],
                     self.paths["level_value_loss"],
                     self.paths["level_instance_value_loss"],
                     self.paths["level_returns"],
                     self.paths["instance_pred_entropy"],
                     self.paths["instance_pred_accuracy"],
                     self.paths["instance_pred_log_prob"],
                     self.paths["instance_pred_prob"],
                     self.paths["instance_pred_precision"],
                     self.paths["instance_pred_recall"],
                     self.paths["instance_pred_f1"]]:
            with open(path, "r+") as level_file:
                reader = csv.reader(level_file)
                lines = list(reader)
                new_lines = lines[:num_update+2]
                level_file.seek(0)
                level_file.truncate()
                writer = csv.writer(level_file)
                writer.writerows(new_lines)

    @property
    def completed(self) -> bool:
        return self.metadata["successful"]

    @property
    def num_duplicates(self) -> int:
        with open(self.paths["final_test_eval"], "r") as finaltestfile:
            reader = csv.reader(finaltestfile)
            lines = list(reader)
        header = lines[0]
        num_dups = 0
        for row in lines:
            if row == header:
                num_dups += 1
        num_dups -= 1
        return num_dups

    def read_data(self):
        # Returns a dictionary of numpy arrays of dim [num_updates]
        with open(self.paths["logs"], "r") as logfile:
            reader = csv.DictReader(logfile)
            lines = list(reader)
        header = lines[0].keys()
        logs = {k: [] for k in header}
        for row in lines:
            for k in header:
                logs[k].append(float(row[k]))
        self._logs = {k: np.array(v) for k, v in logs.items()}

        # Returns a numpy array of weights of dim [num_updates, len(self.seeds)]
        with open(self.paths["level_weights"], "r") as levelweightsfile:
            reader = csv.reader(levelweightsfile)
            lines = list(reader)
        header = lines[0]
        self._level_weights = np.array([[float(val) for val in row] for row in lines[1:]])

        # Returns a numpy array of value loss of dim [num_updates, len(self.seeds)]
        with open(self.paths["level_value_loss"], "r") as levelvaluelossfile:
            reader = csv.reader(levelvaluelossfile)
            lines = list(reader)
        header = lines[0]
        self._level_value_loss = np.array([[float(val) for val in row] for row in lines[1:]])

        # Returns a numpy array of instance value loss of dim [num_updates, len(self.seeds)]
        with open(self.paths["level_instance_value_loss"], "r") as levelinstancevaluelossfile:
            reader = csv.reader(levelinstancevaluelossfile)
            lines = list(reader)
        header = lines[0]
        self._level_instance_value_loss = np.array([[float(val) for val in row] for row in lines[1:]])

        # Returns a numpy array of returns of dim [num_updates, len(self.seeds)]
        with open(self.paths["level_returns"], "r") as levelreturnsfile:
            reader = csv.reader(levelreturnsfile)
            lines = list(reader)
        header = lines[0]
        self._level_train_returns = np.array([[float(val) for val in row] for row in lines[1:]])

        # Returns a numpy array of instance pred entropy of dim [num_updates, len(self.seeds)]
        with open(self.paths["instance_pred_entropy"], "r") as instancepredentropyfile:
            reader = csv.reader(instancepredentropyfile)
            lines = list(reader)
        header = lines[0]
        self._instance_pred_entropy = np.array([[float(val) for val in row] for row in lines[1:]])

        # Returns a numpy array of instance pred accuracy of dim [num_updates, len(self.seeds)]
        with open(self.paths["instance_pred_accuracy"], "r") as instancepredaccuracyfile:
            reader = csv.reader(instancepredaccuracyfile)
            lines = list(reader)
        header = lines[0]
        self._instance_pred_accuracy = np.array([[float(val) for val in row] for row in lines[1:]])

        # Returns a numpy array of instance pred precision of dim [num_updates, len(self.seeds)]
        with open(self.paths["instance_pred_prob"], "r") as instancepredprobfile:
            reader = csv.reader(instancepredprobfile)
            lines = list(reader)
        header = lines[0]
        self._instance_pred_prob = np.array([[float(val) for val in row] for row in lines[1:]])

        # Returns a numpy array of instance pred precision of dim [num_updates, len(self.seeds)]
        with open(self.paths["instance_pred_log_prob"], "r") as instancepredlogprobfile:
            reader = csv.reader(instancepredlogprobfile)
            lines = list(reader)
        #NO HEADER DUE TO BUG
        if len(lines) == len(self._instance_pred_prob) + 1:
            lines = lines[1:]
        self._instance_pred_log_prob = np.array([[float(val) for val in row] for row in lines])

        # Returns a numpy array of instance pred precision of dim [num_updates, len(self.seeds)]
        with open(self.paths["instance_pred_precision"], "r") as instancepredprecisionfile:
            reader = csv.reader(instancepredprecisionfile)
            lines = list(reader)
        header = lines[0]
        self._instance_pred_precision = np.array([[float(val) for val in row] for row in lines[1:]])

        # Returns a numpy array of instance pred precision of dim [num_updates, len(self.seeds)]
        with open(self.paths["instance_pred_recall"], "r") as instancepredrecallfile:
            reader = csv.reader(instancepredrecallfile)
            lines = list(reader)
        header = lines[0]
        self._instance_pred_recall = np.array([[float(val) for val in row] for row in lines[1:]])

        # Returns a numpy array of instance pred precision of dim [num_updates, len(self.seeds)]
        with open(self.paths["instance_pred_f1"], "r") as instancepredf1file:
            reader = csv.reader(instancepredf1file)
            lines = list(reader)
        header = lines[0]
        self._instance_pred_f1 = np.array([[float(val) for val in row] for row in lines[1:]])

        with open(self.paths["final_test_eval"], "r") as finaltestfile:
            reader = csv.DictReader(finaltestfile)
            lines = list(reader)
        self._final_test_eval = {key: float(val) for key, val in lines[0].items()}

    @property
    def logs(self) -> Dict[str, np.ndarray]:
        return self._logs

    @property
    def level_weights(self) -> np.ndarray:
        return self._level_weights

    @property
    def level_value_loss(self) -> np.ndarray:
        return self._level_value_loss

    @property
    def level_instance_value_loss(self) -> np.ndarray:
        return self._level_instance_value_loss

    @property
    def level_train_returns(self) -> np.ndarray:
        return self._level_train_returns

    @property
    def instance_pred_entropy(self) -> np.ndarray:
        return self._instance_pred_entropy

    @property
    def instance_pred_accuracy(self) -> np.ndarray:
        return self._instance_pred_accuracy

    @property
    def instance_pred_log_prob(self) -> np.ndarray:
        return self._instance_pred_log_prob

    @property
    def instance_pred_prob(self) -> np.ndarray:
        return self._instance_pred_prob

    @property
    def instance_pred_precision(self) -> np.ndarray:
        return self._instance_pred_precision

    @property
    def instance_pred_recall(self) -> np.ndarray:
        return self._instance_pred_recall

    @property
    def instance_pred_f1(self) -> np.ndarray:
        return self._instance_pred_f1

    @property
    def final_test_eval(self) -> List[Dict[str, Any]]:
        return self._final_test_eval