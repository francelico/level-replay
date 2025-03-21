# Copyright (c) Facebook, Inc. and its affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
#
# Implementation for mixed sampling strategies and instance_pred_log_prob scoring fn by Samuel Garcin.

from collections import namedtuple
import numpy as np
import torch

class LevelSampler():
    def __init__(
        self, seeds, obs_space, action_space, num_actors=1, 
        strategy='random', replay_schedule='fixed', score_transform='power',
        temperature=1.0, eps=0.05,
        rho=0.2, nu=0.5, alpha=1.0, 
        staleness_coef=0, staleness_transform='power', staleness_temperature=1.0,
        secondary_strategy='off', secondary_strategy_coef=0.0,
        secondary_score_transform='rank', secondary_temperature=1.0,):
        self.obs_space = obs_space
        self.action_space = action_space
        self.strategy = strategy
        self.replay_schedule = replay_schedule
        self.score_transform = score_transform
        self.temperature = temperature
        self.eps = eps
        self.rho = rho
        self.nu = nu
        self.alpha = alpha
        self.staleness_coef = staleness_coef
        self.staleness_transform = staleness_transform
        self.staleness_temperature = staleness_temperature
        self.secondary_strategy = secondary_strategy if secondary_strategy != 'off' else None
        self.secondary_strategy_coef = secondary_strategy_coef
        self.secondary_score_transform = secondary_score_transform
        self.secondary_temperature = secondary_temperature

        # Track seeds and scores as in np arrays backed by shared memory
        self._init_seed_index(seeds)

        self.unseen_seed_weights = np.array([1.]*len(seeds))
        self.seed_scores = np.array([0.]*len(seeds), dtype=np.float)
        self.seed_secondary_scores = np.array([0.]*len(seeds), dtype=np.float)
        self.buffer_logs = dict(level_return=np.full(len(seeds), np.nan, dtype=np.float),
                                level_value_loss=np.full(len(seeds), np.nan, dtype=np.float),
                                level_instance_value_loss=np.full(len(seeds), np.nan, dtype=np.float),
                                instance_pred_entropy=np.full(len(seeds), np.nan, dtype=np.float),
                                instance_pred_accuracy=np.full(len(seeds), np.nan, dtype=np.float),
                                instance_pred_log_prob=np.full(len(seeds), np.nan, dtype=np.float),
                                instance_pred_prob=np.full(len(seeds), np.nan, dtype=np.float),
                                instance_pred_precision=np.full(len(seeds), np.nan, dtype=np.float),
                                instance_pred_recall=np.full(len(seeds), np.nan, dtype=np.float),
                                instance_pred_f1=np.full(len(seeds), np.nan, dtype=np.float))
        self.partial_seed_scores = np.zeros((num_actors, len(seeds)), dtype=np.float)
        self.partial_seed_secondary_scores = np.zeros((num_actors, len(seeds)), dtype=np.float)
        self.partial_seed_steps = np.zeros((num_actors, len(seeds)), dtype=np.int64)
        self.partial_buffer_logs = dict(level_return=np.full((num_actors, len(seeds)), np.nan, dtype=np.float),
                                        level_value_loss=np.full((num_actors, len(seeds)), np.nan, dtype=np.float),
                                        level_instance_value_loss=np.full((num_actors, len(seeds)), np.nan, dtype=np.float),
                                        instance_pred_entropy=np.full((num_actors, len(seeds)), np.nan, dtype=np.float),
                                        instance_pred_accuracy=np.full((num_actors, len(seeds)), np.nan, dtype=np.float),
                                        instance_pred_log_prob=np.full((num_actors, len(seeds)), np.nan, dtype=np.float),
                                        instance_pred_prob=np.full((num_actors, len(seeds)), np.nan, dtype=np.float),
                                        instance_pred_precision=np.full((num_actors, len(seeds)), np.nan, dtype=np.float),
                                        instance_pred_recall=np.full((num_actors, len(seeds)), np.nan, dtype=np.float),
                                        instance_pred_f1=np.full((num_actors, len(seeds)), np.nan, dtype=np.float))
        self.seed_staleness = np.array([0.]*len(seeds), dtype=np.float)

        self.next_seed_index = 0 # Only used for sequential strategy

    def seed_range(self):
        return (int(min(self.seeds)), int(max(self.seeds)))

    def _init_seed_index(self, seeds):
        self.seeds = np.array(seeds, dtype=np.int64)
        self.seed2index = {seed: i for i, seed in enumerate(seeds)}

    def update_with_rollouts(self, rollouts, instance_prediction_stats=None):

        score_functions = []
        for strategy in [self.strategy, self.secondary_strategy]:
            if strategy is not None:
                # Update with a RolloutStorage object
                if strategy == 'policy_entropy':
                    score_function = self._average_entropy
                elif strategy == 'least_confidence':
                    score_function = self._average_least_confidence
                elif strategy == 'min_margin':
                    score_function = self._average_min_margin
                elif strategy == 'gae':
                    score_function = self._average_gae
                elif strategy == 'value_l1':
                    score_function = self._average_value_l1
                elif strategy == 'one_step_td_error':
                    score_function = self._one_step_td_error
                elif strategy == 'positive_value_loss':
                    score_function = self._average_positive_value_loss
                elif strategy == 'clipped_value_loss':
                    score_function = self._average_clipped_value_loss
                elif strategy == 'weighted_value_loss':
                    score_function = self._average_weighted_value_loss
                elif strategy == 'random':
                    score_function = self._always_zero
                elif strategy== 'instance_pred_log_prob':
                    score_function = self._neg_sum_instance_pred_log_prob
                else:
                    raise ValueError(f'Unsupported strategy, {strategy}')
                score_functions.append(score_function)
            else:
                score_functions.append(None)

        self._update_with_rollouts(rollouts, score_functions[0], instance_prediction_stats=instance_prediction_stats,
                                   secondary_score_function=score_functions[1])

    def update_seed_scores(self, actor_index, seed_idx, score, num_steps, secondary_score=None):
        score, secondary_score = self._partial_update_seed_scores(actor_index, seed_idx, score, num_steps, secondary_score, done=True)

        self.unseen_seed_weights[seed_idx] = 0. # No longer unseen

        old_score = self.seed_scores[seed_idx]
        self.seed_scores[seed_idx] = (1 - self.alpha)*old_score + self.alpha*score

        if secondary_score is not None:
            old_secondary_score = self.seed_secondary_scores[seed_idx]
            self.seed_secondary_scores[seed_idx] = (1 - self.alpha)*old_secondary_score + self.alpha*secondary_score

    def update_buffer_logs(self, actor_index, seed_idx, score_function_kwargs, num_steps):
        logs = self._partial_update_buffer_logs(actor_index, seed_idx, score_function_kwargs, num_steps, done=True)
        for k, v in logs.items():
            self.buffer_logs[k][seed_idx] = v

    def _partial_update_seed_scores(self, actor_index, seed_idx, score, num_steps, secondary_scores=None, done=False):
        partial_score = self.partial_seed_scores[actor_index][seed_idx]
        partial_num_steps = self.partial_seed_steps[actor_index][seed_idx]

        running_num_steps = partial_num_steps + num_steps
        merged_score = partial_score + (score - partial_score)*num_steps/float(running_num_steps)

        if secondary_scores is not None:
            partial_secondary_score = self.partial_seed_secondary_scores[actor_index][seed_idx]
            merged_secondary_score = partial_secondary_score + \
                                     (secondary_scores - partial_secondary_score)*num_steps/float(running_num_steps)
        else:
            merged_secondary_score = None

        if done:
            self.partial_seed_scores[actor_index][seed_idx] = 0. # zero partial score, partial num_steps
            self.partial_seed_steps[actor_index][seed_idx] = 0
            self.partial_seed_secondary_scores[actor_index][seed_idx] = 0.
        else:
            self.partial_seed_scores[actor_index][seed_idx] = merged_score
            self.partial_seed_steps[actor_index][seed_idx] = running_num_steps
            if merged_secondary_score is not None:
                self.partial_seed_secondary_scores[actor_index][seed_idx] = merged_secondary_score

        return merged_score, merged_secondary_score

    def _partial_update_buffer_logs(self, actor_index, seed_idx, score_function_kwargs, num_steps, done=False):

        merged_buffer_logs = {}
        buffer_logs = self.compute_buffer_logs(**score_function_kwargs)

        for key in buffer_logs.keys():
            partial_buffer_logs = self.partial_buffer_logs[key][actor_index][seed_idx]
            if np.isnan(partial_buffer_logs):
                partial_buffer_logs = 0
            running_num_steps = self.partial_seed_steps[actor_index][seed_idx] + num_steps
            merged_buffer_logs[key] = partial_buffer_logs + (buffer_logs[key] - partial_buffer_logs)*num_steps/float(running_num_steps)

            if done:
                self.partial_buffer_logs[key][actor_index][seed_idx] = np.nan
            else:
                self.partial_buffer_logs[key][actor_index][seed_idx] = merged_buffer_logs[key]

        return merged_buffer_logs

    def compute_buffer_logs(self, **kwargs):
        value_loss = kwargs['returns'] - kwargs['value_preds']
        value_loss = value_loss.abs().mean().item()

        instance_value_loss = kwargs['returns'] - kwargs['instance_value_preds']
        instance_value_loss = instance_value_loss.abs().mean().item()

        if kwargs['done']:
            ep_returns = kwargs['rewards'].sum().item()
        else:
            ep_returns = np.nan

        instance_pred_entropy = kwargs['instance_pred_entropy'].mean().item()
        instance_pred_accuracy = kwargs['instance_pred_accuracy'].mean().item()
        instance_pred_log_prob = kwargs['instance_pred_log_prob'].sum().item()
        instance_pred_prob = kwargs['instance_pred_log_prob'].exp().mean().item()

        return {'level_value_loss': value_loss,
                'level_instance_value_loss': instance_value_loss,
                'level_return': ep_returns,
                'instance_pred_log_prob': instance_pred_log_prob,
                'instance_pred_prob': instance_pred_prob,
                'instance_pred_accuracy': instance_pred_accuracy,
                'instance_pred_entropy': instance_pred_entropy,
                'instance_pred_precision': kwargs['precision'] if 'precision' in kwargs else np.nan,
                'instance_pred_recall': kwargs['recall'] if 'recall' in kwargs else np.nan,
                'instance_pred_f1': kwargs['f1-score'] if 'f1-score' in kwargs else np.nan,
                }

    def _average_entropy(self, **kwargs):
        episode_logits = kwargs['episode_logits']
        num_actions = self.action_space.n
        max_entropy = -(1./num_actions)*np.log(1./num_actions)*num_actions

        return (-torch.exp(episode_logits)*episode_logits).sum(-1).mean().item()/max_entropy

    def _average_least_confidence(self, **kwargs):
        episode_logits = kwargs['episode_logits']
        return (1 - torch.exp(episode_logits.max(-1, keepdim=True)[0])).mean().item()

    def _average_min_margin(self, **kwargs):
        episode_logits = kwargs['episode_logits']
        top2_confidence = torch.exp(episode_logits.topk(2, dim=-1)[0])
        return 1 - (top2_confidence[:,0] - top2_confidence[:,1]).mean().item()

    def _average_gae(self, **kwargs):
        returns = kwargs['returns']
        value_preds = kwargs['value_preds']

        advantages = returns - value_preds

        return advantages.mean().item()

    def _average_value_l1(self, **kwargs):
        returns = kwargs['returns']
        value_preds = kwargs['value_preds']

        advantages = returns - value_preds

        return advantages.abs().mean().item()

    def _average_positive_value_loss(self, **kwargs):
        returns = kwargs['returns']
        value_preds = kwargs['value_preds']

        clipped_advantages = (returns - value_preds).clamp(0)

        return clipped_advantages.mean().item()

    def _average_clipped_value_loss(self, **kwargs):
        # outputs (returns - value_preds) if value_preds < returns < instance_value_preds, else 0
        returns = kwargs['returns']
        value_preds = kwargs['value_preds']
        instance_value_preds = kwargs['instance_value_preds']

        clipped_returns = torch.where(returns >= instance_value_preds, 0, returns)
        clipped_advantages = (clipped_returns - value_preds).clamp(0)

        return clipped_advantages.mean().item()

    def _average_weighted_value_loss(self, **kwargs):
        # outputs w * (returns - value_preds) if value_preds < returns < instance_value_preds, else 0
        # where w = 1 - (returns - value_preds) / (instance_value_preds - value_preds)
        # i.e. w linearly decays as returns approaches instance_value_preds
        returns = kwargs['returns']
        value_preds = kwargs['value_preds']
        instance_value_preds = kwargs['instance_value_preds']

        clipped_returns = torch.where(returns >= instance_value_preds, 0, returns)
        clipped_advantages = (clipped_returns - value_preds).clamp(0)

        value_diff = (instance_value_preds - value_preds).clamp(0)
        weighted_advantages = torch.where(value_diff > 0, clipped_advantages * (1 - clipped_advantages / value_diff), 0)

        return weighted_advantages.mean().item()

    def _neg_sum_instance_pred_log_prob(self, **kwargs):
        return - kwargs['instance_pred_log_prob'].sum().item()

    def _always_zero(self, **kwargs):
        return 0

    def _one_step_td_error(self, **kwargs):
        rewards = kwargs['rewards']
        value_preds = kwargs['value_preds']

        max_t = len(rewards)
        td_errors = (rewards[:-1] + value_preds[:max_t-1] - value_preds[1:max_t]).abs()

        return td_errors.abs().mean().item()

    @property
    def requires_value_buffers(self):
        return self.strategy in ['gae', 'value_l1', 'one_step_td_error', 'positive_value_loss', 'clipped_value_loss',
                                 'weighted_value_loss', 'random', 'instance_pred_log_prob']

    def _update_with_rollouts(self, rollouts, score_function, instance_prediction_stats=None,
                              secondary_score_function=None):
        level_seeds = rollouts.level_seeds.detach()
        policy_logits = rollouts.action_log_dist.detach()
        done = ~(rollouts.masks > 0)
        total_steps, num_actors = policy_logits.shape[:2]
        num_decisions = len(policy_logits)

        for actor_index in range(num_actors):
            done_steps = done[:,actor_index].nonzero()[:total_steps,0]
            start_t = 0

            for t in done_steps:
                if not start_t < total_steps: break

                if t == 0: # if t is 0, then this done step caused a full update of previous seed last cycle
                    continue 

                seed_t = level_seeds[start_t,actor_index].item()
                seed_idx_t = self.seed2index[seed_t]

                score_function_kwargs = {}
                episode_logits = policy_logits[start_t:t,actor_index]
                score_function_kwargs['episode_logits'] = torch.log_softmax(episode_logits, -1)
                num_steps = len(episode_logits)

                if self.requires_value_buffers:
                    score_function_kwargs['returns'] = rollouts.returns[start_t:t,actor_index].detach()
                    score_function_kwargs['rewards'] = rollouts.rewards[start_t:t,actor_index].detach()
                    score_function_kwargs['value_preds'] = rollouts.value_preds[start_t:t,actor_index].detach()
                    score_function_kwargs['instance_value_preds'] = rollouts.instance_value_preds[start_t:t,actor_index].detach()
                    score_function_kwargs['instance_pred_log_prob'] = rollouts.instance_pred_log_prob[start_t:t,actor_index].detach()
                    score_function_kwargs['instance_pred_accuracy'] = rollouts.instance_pred_accuracy[start_t:t,actor_index].detach()
                    score_function_kwargs['instance_pred_entropy'] = rollouts.instance_pred_entropy[start_t:t,actor_index].detach()
                    score_function_kwargs['done'] = True
                    if instance_prediction_stats is not None:
                        stats = instance_prediction_stats['classification_report'][str(seed_idx_t)]
                        for key in stats:
                            score_function_kwargs[key] = stats[key]
                    self.update_buffer_logs(actor_index, seed_idx_t, score_function_kwargs, num_steps)

                score = score_function(**score_function_kwargs)
                if secondary_score_function is not None:
                    secondary_score = secondary_score_function(**score_function_kwargs)
                else:
                    secondary_score = None
                self.update_seed_scores(actor_index, seed_idx_t, score, num_steps, secondary_score)

                start_t = t.item()

            if start_t < total_steps:
                seed_t = level_seeds[start_t,actor_index].item()
                seed_idx_t = self.seed2index[seed_t]

                score_function_kwargs = {}
                episode_logits = policy_logits[start_t:,actor_index]
                score_function_kwargs['episode_logits'] = torch.log_softmax(episode_logits, -1)
                num_steps = len(episode_logits)

                if self.requires_value_buffers:
                    score_function_kwargs['returns'] = rollouts.returns[start_t:,actor_index].detach()
                    score_function_kwargs['rewards'] = rollouts.rewards[start_t:,actor_index].detach()
                    score_function_kwargs['value_preds'] = rollouts.value_preds[start_t:,actor_index].detach()
                    score_function_kwargs['instance_value_preds'] = rollouts.instance_value_preds[start_t:,actor_index].detach()
                    score_function_kwargs['instance_pred_log_prob'] = rollouts.instance_pred_log_prob[start_t:,actor_index].detach()
                    score_function_kwargs['instance_pred_accuracy'] = rollouts.instance_pred_accuracy[start_t:,actor_index].detach()
                    score_function_kwargs['instance_pred_entropy'] = rollouts.instance_pred_entropy[start_t:,actor_index].detach()
                    score_function_kwargs['done'] = False
                    if instance_prediction_stats is not None:
                        stats = instance_prediction_stats['classification_report'][str(seed_idx_t)]
                        for key in stats:
                            score_function_kwargs[key] = stats[key]
                    self._partial_update_buffer_logs(actor_index, seed_idx_t, score_function_kwargs, num_steps)

                score = score_function(**score_function_kwargs)
                if secondary_score_function is not None:
                    secondary_score = secondary_score_function(**score_function_kwargs)
                else:
                    secondary_score = None
                self._partial_update_seed_scores(actor_index, seed_idx_t, score, num_steps, secondary_score)

    def after_update(self):
        # Reset partial updates, since weights have changed, and thus logits are now stale
        for actor_index in range(self.partial_seed_scores.shape[0]):
            for seed_idx in range(self.partial_seed_scores.shape[1]):
                if self.partial_seed_scores[actor_index][seed_idx] != 0:
                    self.update_seed_scores(actor_index, seed_idx, 0, 0, 0)
        self.partial_seed_scores.fill(0)
        self.partial_seed_steps.fill(0)
        self.partial_seed_secondary_scores.fill(0)

    def after_logging(self):
        # Reset buffer log and partial buffer log
        for key in self.partial_buffer_logs:
            self.partial_buffer_logs[key].fill(np.nan)
            self.buffer_logs[key].fill(np.nan)

    def _update_staleness(self, selected_idx):
        if self.staleness_coef > 0:
            self.seed_staleness = self.seed_staleness + 1
            self.seed_staleness[selected_idx] = 0

    def _sample_replay_level(self):
        sample_weights = self.sample_weights()

        if np.isclose(np.sum(sample_weights), 0):
            sample_weights = np.ones_like(sample_weights, dtype=np.float)/len(sample_weights)

        seed_idx = np.random.choice(range(len(self.seeds)), 1, p=sample_weights)[0]
        seed = self.seeds[seed_idx]

        self._update_staleness(seed_idx)

        return int(seed)

    def _sample_unseen_level(self):
        sample_weights = self.unseen_seed_weights/self.unseen_seed_weights.sum()
        seed_idx = np.random.choice(range(len(self.seeds)), 1, p=sample_weights)[0]
        seed = self.seeds[seed_idx]

        self._update_staleness(seed_idx)

        return int(seed)

    def sample(self, strategy=None):
        if not strategy:
            strategy = self.strategy

        if strategy == 'random':
            seed_idx = np.random.choice(range((len(self.seeds))))
            seed = self.seeds[seed_idx]
            return int(seed)

        if strategy == 'sequential':
            seed_idx = self.next_seed_index
            self.next_seed_index = (self.next_seed_index + 1) % len(self.seeds)
            seed = self.seeds[seed_idx]
            return int(seed)

        num_unseen = (self.unseen_seed_weights > 0).sum()
        proportion_seen = (len(self.seeds) - num_unseen)/len(self.seeds)

        if self.replay_schedule == 'fixed':
            if proportion_seen >= self.rho: 
                # Sample replay level with fixed prob = 1 - nu OR if all levels seen
                if np.random.rand() > self.nu or not proportion_seen < 1.0:
                    return self._sample_replay_level()

            # Otherwise, sample a new level
            return self._sample_unseen_level()

        else: # Default to proportionate schedule
            if proportion_seen >= self.rho and np.random.rand() < proportion_seen:
                return self._sample_replay_level()
            else:
                return self._sample_unseen_level()

    def sample_weights(self):
        weights = self._score_transform(self.score_transform, self.temperature, self.seed_scores)
        weights = weights * (1-self.unseen_seed_weights) # zero out unseen levels

        z = np.sum(weights)
        if z > 0:
            weights /= z

        if self.secondary_strategy is not None:
            secondary_weights = self._score_transform(self.secondary_score_transform, self.secondary_temperature,
                                                      self.seed_secondary_scores)
            secondary_weights = secondary_weights * (1-self.unseen_seed_weights)
            z = np.sum(secondary_weights)
            if z > 0:
                secondary_weights /= z
            weights = (1 - self.secondary_strategy_coef)*weights + self.secondary_strategy_coef*secondary_weights

        staleness_weights = 0
        if self.staleness_coef > 0:
            staleness_weights = self._score_transform(self.staleness_transform, self.staleness_temperature, self.seed_staleness)
            staleness_weights = staleness_weights * (1-self.unseen_seed_weights)
            z = np.sum(staleness_weights)
            if z > 0: 
                staleness_weights /= z

            weights = (1 - self.staleness_coef)*weights + self.staleness_coef*staleness_weights

        return weights

    def sample_level_returns(self):
        return self.buffer_logs['level_return']

    def sample_level_value_loss(self):
        return self.buffer_logs['level_value_loss']

    def sample_level_instance_value_loss(self):
        return self.buffer_logs['level_instance_value_loss']

    def sample_instance_pred_log_prob(self):
        return self.buffer_logs['instance_pred_log_prob']

    def sample_instance_pred_prob(self):
        return self.buffer_logs['instance_pred_prob']

    def sample_instance_pred_accuracy(self):
        return self.buffer_logs['instance_pred_accuracy']

    def sample_instance_pred_entropy(self):
        return self.buffer_logs['instance_pred_entropy']

    def sample_instance_pred_precision(self):
        return self.buffer_logs['instance_pred_precision']

    def sample_instance_pred_recall(self):
        return self.buffer_logs['instance_pred_recall']

    def sample_instance_pred_f1(self):
        return self.buffer_logs['instance_pred_f1']

    def _score_transform(self, transform, temperature, scores):
        if transform == 'constant':
            weights = np.ones_like(scores)
        if transform == 'max':
            weights = np.zeros_like(scores)
            scores = scores[:]
            scores[self.unseen_seed_weights > 0] = -float('inf') # only argmax over seen levels
            argmax = np.random.choice(np.flatnonzero(np.isclose(scores, scores.max())))
            weights[argmax] = 1.
        elif transform == 'eps_greedy':
            weights = np.zeros_like(scores)
            weights[scores.argmax()] = 1. - self.eps
            weights += self.eps/len(self.seeds)
        elif transform == 'rank':
            temp = np.flip(scores.argsort())
            ranks = np.empty_like(temp)
            ranks[temp] = np.arange(len(temp)) + 1
            weights = 1/ranks ** (1./temperature)
        elif transform == 'power':
            eps = 0 if self.staleness_coef > 0 else 1e-3
            weights = (np.array(scores) + eps) ** (1./temperature)
        elif transform == 'softmax':
            weights = np.exp(np.array(scores)/temperature)

        return weights