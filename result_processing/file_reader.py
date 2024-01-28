import numpy as np

from level_replay.file_writer import FileWriter
import os
import pandas as pd
import logging

from level_replay.utils import DotDict


class LogReader:

    def __init__(self, run_name, root_dir, output_dir, rolling_window=10, seeds=None, ignore_extra=False):

        self.run_name = run_name
        self.root_dir = root_dir
        self.run_dir = os.path.join(root_dir, run_name)
        self.output_dir = output_dir
        self.pid_dirs = [os.path.join(self.run_dir, p) for p in os.listdir(self.run_dir) if p != 'latest']
        if ignore_extra:
            self.pid_dirs = [p for p in self.pid_dirs if not p.endswith('_extra')]
        self.pid_filewriters = [FileWriter(rootdir=str(p), no_setup=True, symlink_to_latest=False) for p in self.pid_dirs]
        self.pid_filewriters = [fw for fw in self.pid_filewriters if fw.completed]
        self.num_updates = int(self.args.num_env_steps) // self.args.num_steps // self.args.num_processes

        self._logger = logging.getLogger(self.__class__.__name__)
        self._logger.setLevel(logging.WARNING)
        if self._logger.hasHandlers():
            self._logger.handlers.clear()

        self._get_logs(rolling_window=rolling_window, seeds=seeds)

    def _get_logs(self, rolling_window=10, seeds=None):

        if seeds is not None:
            self.pid_filewriters = [fw for fw in self.pid_filewriters if str(fw.metadata['args']['seed']) in seeds]
            if len(self.pid_filewriters) != len(seeds):
                self._logger.warning(f'Not all seeds were found in the run {self.run_name}.')
                self._logs = None
                return

        labels_to_smooth = ['train_eval:mean_episode_return', 'test:mean_episode_return', 'train:mean_episode_return',
                            'instance_pred_accuracy_train', 'instance_pred_prob_train', 'instance_pred_entropy_train',
                            'instance_pred_accuracy', 'instance_pred_prob', 'instance_pred_entropy', 'level_value_loss',
                            'instance_pred_accuracy_stale', 'instance_pred_prob_stale', 'instance_pred_entropy_stale',
                            'mutual_information', 'mutual_information_stale', 'generalisation_gap', 'mean_agent_return']
        logs = [self._fix_logging_inconsistencies(fw) for fw in self.pid_filewriters]

        for i, log in enumerate(logs):
            for label in labels_to_smooth:
                data = pd.Series(log[label])
                log[f'{label}_mavg'] = data.rolling(rolling_window).mean().to_numpy()
                if label == 'generalisation_gap':
                    log[f'{label}_mavg'][-1] = log['train_eval:mean_episode_return_mavg'][-1] - self.pid_filewriters[i].final_test_eval['mean_episode_return']

        self.env_name = self.args['env_name']
        self.level_replay_strategy = self.args['level_replay_strategy']
        self.level_replay_secondary_strategy = self.args['level_replay_secondary_strategy']
        self.secondary_strategy_fraction_end = self.args['level_replay_secondary_strategy_coef_update_fraction'] if \
            'level_replay_secondary_strategy_coef_update_fraction' in self.args else \
            self.args['level_replay_secondary_strategy_fraction_end']
        self.env_steps = logs[0]['step']

        for i, pid_filewriter in enumerate(self.pid_filewriters):
            assert self.env_name == pid_filewriter.metadata['args']['env_name'], "All pids must have the same env_name"
            assert self.level_replay_strategy == pid_filewriter.metadata['args']['level_replay_strategy'], \
                "All pids must have the same level_replay_strategy"

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
        if self.num_updates != logs['# _tick'][-1] + 1:
            self._logger.warning(f"Num updare mismatch: {logs['# _tick'][-1]} != {self.num_updates} "
                                 f" in {filewriter.basepath}")
            if np.abs(self.num_updates - (logs['# _tick'][-1] + 1)) > 1:
                raise ValueError(f"Number of update mismatch in {filewriter.basepath} in too high "
                                 f"({np.abs(self.num_updates - logs['# _tick'][-1]) + 1})")
            else:
                for key in logs.keys():
                    new_array = np.zeros(self.num_updates).astype(logs[key].dtype)
                    new_array[:len(logs[key])] = logs[key]
                    logs[key] = new_array
                    logs[key][-1] = logs[key][-2]
                logs['# _tick'][-1] = logs['# _tick'][-1] + 1
        logs['total_student_grad_updates'] = logs['# _tick']
        logs['generalisation_gap'] = logs['train_eval:mean_episode_return'] - logs['test:mean_episode_return'] #TODO: at minus -1 should use the final eval value.
        logs['generalisation_gap'][-1] = logs['train_eval:mean_episode_return'][-1] - filewriter.final_test_eval['mean_episode_return']
        logs['mean_agent_return'] = logs['train:mean_episode_return']
        logs['mutual_information'] = self.compute_mutual_information(filewriter, False)
        logs['mutual_information_stale'] = self.compute_mutual_information(filewriter, True)
        logs['mutual_information_u'] = self.compute_mutual_information(filewriter, False, uniform=True)
        logs['mutual_information_u_stale'] = self.compute_mutual_information(filewriter, True, uniform=True)
        logs['overgen_gap'] = self.compute_overgen_gap(filewriter, False)
        logs['overgen_gap_stale'] = self.compute_overgen_gap(filewriter, True)
        logs['instance_pred_accuracy_stale'] = np.nanmean(self.get_stat_with_stale_updates(filewriter.instance_pred_accuracy), axis=-1)
        logs['instance_pred_prob_stale'] = np.nanmean(self.get_stat_with_stale_updates(filewriter.instance_pred_prob), axis=-1)
        logs['instance_pred_entropy_stale'] = np.nanmean(self.get_stat_with_stale_updates(filewriter.instance_pred_entropy), axis=-1)
        logs['instance_pred_accuracy'] = np.nanmean(self._fix_missing_stats(filewriter.instance_pred_accuracy), axis=-1)
        logs['instance_pred_prob'] = np.nanmean(self._fix_missing_stats(filewriter.instance_pred_prob), axis=-1)
        logs['instance_pred_entropy'] = np.nanmean(self._fix_missing_stats(filewriter.instance_pred_entropy), axis=-1)
        logs['level_value_loss'] = np.nanmean(self._fix_missing_stats(filewriter.level_value_loss), axis=-1)
        logs['num_seeds_buffer'] = np.ones_like(logs['# _tick']) * self.args.num_train_seeds #Not sure if needed
        logs['total_dgps'] = logs['num_seeds_buffer']
        logs['generalisation_bound'] = np.sqrt(2 * logs['mutual_information'] / logs['total_dgps'])
        logs['generalisation_bound_stale'] = np.sqrt(2 * logs['mutual_information_stale'] / logs['total_dgps'])
        logs['generalisation_bound_u'] = np.sqrt(2 * logs['mutual_information_u'] / logs['total_dgps'])
        logs['generalisation_bound_u_stale'] = np.sqrt(2 * logs['mutual_information_u_stale'] / logs['total_dgps'])
        return logs

    def _fix_missing_stats(self, data):
        for i, s in enumerate(data):
            if np.isnan(s).all():
                self._logger.warning("Fixing nan at tick %d", i)
                if i > 0:
                    data[i] = data[i - 1]
                else:
                    data[i] = data[i + 1]
        return data

    # A more accurate but slower approach would be to fit a linear predictor to recover weights, but this is good enough
    # low bias but high variance
    def get_rollout_weightings(self, fw, use_stale_updates=False, normalize=True):
        sampled = ~np.isnan(fw.level_train_returns)
        for i, s in enumerate(sampled):
            if not s.any():
                self._logger.warning("Found missing log in %s at update %d", fw.basepath, i)
                if i > 0:
                    sampled[i] = sampled[i - 1]
                else:
                    sampled[i] = sampled[i + 1]
        shifted_w = fw.level_weights.copy()
        shifted_w[1:] = shifted_w[:-1]
        shifted_w[0] = 1 / shifted_w.shape[1]
        ws = np.zeros_like(shifted_w)
        if use_stale_updates:
            for i, s in enumerate(sampled):
                if i > 0:
                    ws[i] = ws[i-1]

                ws[i][s] = shifted_w[i][s]
        else:
            ws[sampled] = shifted_w[sampled]

        if normalize:
            ws = ws / ws.sum(axis=-1).reshape(-1, 1)
        return ws

    def get_stat_with_stale_updates(self, stat):
        new_stat = stat.copy()
        sampled = ~np.isnan(stat)
        for i, s in enumerate(sampled):
            if not s.any():
                self._logger.warning("Found missing log at update %d", i)
                if i > 0:
                    s = sampled[i - 1]
                else:
                    s = sampled[i + 1]
            if i > 0:
                new_stat[i] = new_stat[i-1]
            new_stat[i][s] = stat[i][s]
        return new_stat

    def compute_mutual_information(self, fw, use_stale_updates=False, uniform=False):
        if uniform:
            pi = np.ones_like(fw.level_train_returns) / fw.level_train_returns.shape[-1]
        else:
            pi = self.get_rollout_weightings(fw, use_stale_updates=use_stale_updates, normalize=True)
        with np.errstate(divide='ignore'):
            hpi = -(pi * np.nan_to_num(np.log(pi), neginf=0)).sum(axis=-1)
        logpred = fw.instance_pred_log_prob
        if use_stale_updates:
            logpred = self.get_stat_with_stale_updates(logpred)
        rollout_buffer_size = fw.metadata['args']['num_processes'] * fw.metadata['args']['num_steps']
        mi = hpi + np.nansum(pi * logpred, axis=-1) / rollout_buffer_size
        return mi

    def compute_overgen_gap(self, fw, use_stale_updates=False):
        pi = self.get_rollout_weightings(fw, use_stale_updates=use_stale_updates, normalize=True)
        if use_stale_updates:
            sampled = ~np.isnan(fw.level_train_returns)
            scores = np.zeros_like(fw.level_train_returns)
            for i, s in enumerate(sampled):
                if i > 0:
                    scores[i] = scores[i-1]
                scores[i][s] = fw.level_train_returns[i][s]
        else:
            scores = fw.level_train_returns
        return np.nansum(pi * scores, axis=-1) - np.nanmean(scores, axis=-1)

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

    @property
    def args(self)->DotDict:
        return DotDict(self.pid_filewriters[0].metadata['args'])

    @property
    def final_test_eval_scores(self)->np.ndarray:
        return np.array([float(fw.final_test_eval['mean_episode_return']) for fw in self.pid_filewriters])

    @property
    def final_test_eval_mean(self):
        return self.final_test_eval_scores.mean()

    @property
    def final_test_eval_std(self):
        return self.final_test_eval_scores.std()

    @property
    def completed(self):
        if not self.pid_filewriters:
            return False
        return all([fw.completed for fw in self.pid_filewriters])

    @property
    def s1(self):
        return self.args['level_replay_strategy']

    @property
    def s2(self):
        return self.args['level_replay_secondary_strategy']

    @property
    def bf(self):
        return self.args['level_replay_secondary_strategy_coef_end']

    @property
    def l2(self):
        return self.args['level_replay_secondary_temperature']

    @property
    def fs(self):
        return self.args['level_replay_strategy_fraction_start']

    @property
    def fe(self):
        return self.secondary_strategy_fraction_end


if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--run_name', type=str, default='exp-1')
    parser.add_argument('--root_dir', type=str, default='~/logs/ppo')
    parser.add_argument('--output_dir', type=str, default='~/logs/ppo')
    args = parser.parse_args()

    log_reader = LogReader(run_name=args.run_name, root_dir=args.root_dir, output_dir=args.output_dir)