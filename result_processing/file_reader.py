import numpy as np

from level_replay.file_writer import FileWriter
import os
import pandas as pd
import logging


class LogReader:

    def __init__(self, run_name, root_dir, output_dir, rolling_window=10):

        self.run_name = run_name
        self.root_dir = root_dir
        self.run_dir = os.path.join(root_dir, run_name)
        self.output_dir = output_dir
        self.pid_dirs = [os.path.join(self.run_dir, p) for p in os.listdir(self.run_dir) if p != 'latest']
        self.pid_filewriters = [FileWriter(rootdir=str(p), no_setup=True, symlink_to_latest=False) for p in self.pid_dirs]

        self._logger = logging.getLogger(self.__class__.__name__)
        if self._logger.hasHandlers():
            self._logger.handlers.clear()

        self._get_logs(rolling_window=10)

    def _get_logs(self, rolling_window=10):

        labels_to_smooth = ['train_eval:mean_episode_return', 'test:mean_episode_return', 'train:mean_episode_return',
                            'instance_pred_accuracy', 'instance_pred_precision', 'instance_pred_entropy']
        logs = [self._fix_logging_inconsistencies(fw) for fw in self.pid_filewriters]

        for i, log in enumerate(logs):
            for label in labels_to_smooth:
                data = pd.Series(log[label])
                log[f'{label}_mavg'] = data.rolling(rolling_window).mean().to_numpy()

        self.env_name = self.pid_filewriters[0].metadata['args']['env_name']
        self.level_replay_strategy = self.pid_filewriters[0].metadata['args']['level_replay_strategy']
        self.num_updates = int(logs[0]['# _tick'][-1])
        self.env_steps = logs[0]['step']

        for i, pid_filewriter in enumerate(self.pid_filewriters):
            assert self.env_name == pid_filewriter.metadata['args']['env_name'], "All pids must have the same env_name"
            assert self.level_replay_strategy == pid_filewriter.metadata['args']['level_replay_strategy'], \
                "All pids must have the same level_replay_strategy"
            if self.num_updates != logs[i]['# _tick'][-1]:
                self._logger.warning(f"Num updare mismatch: {logs[i]['# _tick'][-1]} != {self.num_updates} "
                f" in {pid_filewriter.basepath}")
                if np.abs(self.num_updates - logs[i]['# _tick'][-1]) > 1:
                    raise ValueError(f"Number of update mismatch in {pid_filewriter.basepath} in too high "
                                     f"({np.abs(self.num_updates - logs[i]['# _tick'][-1])})")

            # assert np.all(self.env_steps == logs[i]['step']), "All pids must have the same env_steps"

        self._logs = pd.concat([pd.DataFrame(d) for d in logs]).reset_index(drop=True)

    def _fix_logging_inconsistencies(self, filewriter):
        logs = filewriter.logs
        ticks = logs['# _tick']
        for i, tick in enumerate(ticks):
            if i != tick:
                self._logger.warning("Fixing logging inconsistency in %s at tick %d", filewriter.basepath, tick)
                logs['# _tick'] = np.arange(len(ticks))
                break

        return logs

    @property
    def logs(self)->pd.DataFrame:
        return self._logs

    @property
    def pg_loss(self)->pd.DataFrame:
        return pd.DataFrame([fw.logs['pg_loss'] for fw in self.pid_filewriters]).T

    @property
    def value_loss(self)->pd.DataFrame:
        return pd.DataFrame([fw.logs['value_loss'] for fw in self.pid_filewriters]).T

    @property
    def dist_entropy(self)->pd.DataFrame:
        return pd.DataFrame([fw.logs['dist_entropy'] for fw in self.pid_filewriters]).T

    @property
    def instance_pred_loss(self)->pd.DataFrame:
        return pd.DataFrame([fw.logs['instance_pred_loss'] for fw in self.pid_filewriters]).T

    @property
    def instance_pred_entropy(self)->pd.DataFrame:
        return pd.DataFrame([fw.logs['instance_pred_entropy'] for fw in self.pid_filewriters]).T

    @property
    def instance_pred_accuracy(self)->pd.DataFrame:
        return pd.DataFrame([fw.logs['instance_pred_accuracy'] for fw in self.pid_filewriters]).T

    @property
    def instance_pred_precision(self)->pd.DataFrame:
        return pd.DataFrame([fw.logs['instance_pred_precision'] for fw in self.pid_filewriters]).T

    @property
    def train_mean_return(self)->pd.DataFrame:
        return pd.DataFrame([fw.logs['train:mean_episode_return'] for fw in self.pid_filewriters]).T

    @property
    def train_median_return(self)->pd.DataFrame:
        return pd.DataFrame([fw.logs['train:median_episode_return'] for fw in self.pid_filewriters]).T

    @property
    def test_mean_return(self)->pd.DataFrame:
        return pd.DataFrame([fw.logs['test:mean_episode_return'] for fw in self.pid_filewriters]).T

    @property
    def test_median_return(self)->pd.DataFrame:
        return pd.DataFrame([fw.logs['test:median_episode_return'] for fw in self.pid_filewriters]).T

    @property
    def train_eval_mean_return(self)->pd.DataFrame:
        return pd.DataFrame([fw.logs['train_eval:mean_episode_return'] for fw in self.pid_filewriters]).T

    @property
    def train_eval_median_return(self)->pd.DataFrame:
        return pd.DataFrame([fw.logs['train_eval:median_episode_return'] for fw in self.pid_filewriters]).T

    @property
    def instance_predictor_hidden_size(self)->int:
        value = self.pid_filewriters[0].metadata['args']['instance_predictor_hidden_size']
        assert all([value == fw.metadata['args']['instance_predictor_hidden_size'] for fw in self.pid_filewriters]), \
            "All pids must have the same instance_predictor_hidden_size"
        return value


if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--run_name', type=str, default='exp-1')
    parser.add_argument('--root_dir', type=str, default='~/logs/ppo')
    parser.add_argument('--output_dir', type=str, default='~/logs/ppo')
    args = parser.parse_args()

    log_reader = LogReader(run_name=args.run_name, root_dir=args.root_dir, output_dir=args.output_dir)