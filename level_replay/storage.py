# Copyright (c) 2017 Ilya Kostrikov
# 
# Licensed under the MIT License;
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     https://opensource.org/licenses/MIT
#
# This file is a modified version of:
# https://github.com/rraileanu/auto-drac/blob/master/ucb_rl2_meta/model.py
#
# Instance predictor implementation by Samuel Garcin.

import torch
from torch.utils.data.sampler import \
    BatchSampler, SubsetRandomSampler, SequentialSampler, WeightedRandomSampler


def _flatten_helper(T, N, _tensor):
    return _tensor.view(T * N, *_tensor.size()[2:])


class RolloutStorage(object):
    def __init__(self, num_steps, num_processes, obs_shape, action_space,
                 recurrent_hidden_state_size, actor_feature_size, split_ratio=0.05):
        self.obs = torch.zeros(num_steps + 1, num_processes, *obs_shape)
        self.recurrent_hidden_states = torch.zeros(
            num_steps + 1, num_processes, recurrent_hidden_state_size)
        self.actor_features = torch.zeros(num_steps + 1, num_processes, actor_feature_size)
        self.rewards = torch.zeros(num_steps, num_processes, 1)
        self.value_preds = torch.zeros(num_steps + 1, num_processes, 1)
        self.instance_value_preds = torch.zeros(num_steps + 1, num_processes, 1)
        self.returns = torch.zeros(num_steps + 1, num_processes, 1)
        self.action_log_probs = torch.zeros(num_steps, num_processes, 1)
        self.action_log_dist = torch.zeros(num_steps, num_processes, action_space.n)
        if action_space.__class__.__name__ == 'Discrete':
            action_shape = 1
        else:
            action_shape = action_space.shape[0]
        self.actions = torch.zeros(num_steps, num_processes, action_shape)
        if action_space.__class__.__name__ == 'Discrete':
            self.actions = self.actions.long()
        self.masks = torch.ones(num_steps + 1, num_processes, 1)

        # Masks that indicate whether it's a true terminal state
        # or time limit end state
        self.bad_masks = torch.ones(num_steps + 1, num_processes, 1)

        self.level_seeds = torch.zeros(num_steps, num_processes, 1, dtype=torch.int)
        self.instance_pred_entropy = torch.zeros(num_steps, num_processes, 1)
        self.instance_pred_accuracy = torch.zeros(num_steps, num_processes, 1)
        self.instance_pred_log_prob = torch.zeros(num_steps, num_processes, 1)

        self.num_steps = num_steps
        self.step = 0
        self.device = None
        
        self.split_ratio = split_ratio

    def to(self, device):
        self.obs = self.obs.to(device)
        self.recurrent_hidden_states = self.recurrent_hidden_states.to(device)
        self.actor_features = self.actor_features.to(device)
        self.rewards = self.rewards.to(device)
        self.value_preds = self.value_preds.to(device)
        self.instance_value_preds = self.instance_value_preds.to(device)
        self.returns = self.returns.to(device)
        self.action_log_probs = self.action_log_probs.to(device)
        self.action_log_dist = self.action_log_dist.to(device)
        self.actions = self.actions.to(device)
        self.masks = self.masks.to(device)
        self.bad_masks = self.bad_masks.to(device)
        self.level_seeds = self.level_seeds.to(device)
        self.instance_pred_entropy = self.instance_pred_entropy.to(device)
        self.instance_pred_accuracy = self.instance_pred_accuracy.to(device)
        self.instance_pred_log_prob = self.instance_pred_log_prob.to(device)
        self.device = device

    def insert(self, obs, recurrent_hidden_states, actor_features, actions, action_log_probs, action_log_dist,
               value_preds, instance_value_preds, rewards, masks, bad_masks, level_seeds):
        if len(rewards.shape) == 3: rewards = rewards.squeeze(2)
        self.obs[self.step + 1].copy_(obs)
        self.recurrent_hidden_states[self.step +
                                     1].copy_(recurrent_hidden_states)
        self.actor_features[self.step + 1].copy_(actor_features)
        self.actions[self.step].copy_(actions)
        self.action_log_probs[self.step].copy_(action_log_probs)
        self.action_log_dist[self.step].copy_(action_log_dist)
        self.value_preds[self.step].copy_(value_preds)
        self.instance_value_preds[self.step].copy_(instance_value_preds)
        self.rewards[self.step].copy_(rewards)
        self.masks[self.step + 1].copy_(masks)
        self.bad_masks[self.step + 1].copy_(bad_masks)

        self.level_seeds[self.step].copy_(level_seeds)

        self.step = (self.step + 1) % self.num_steps

    def insert_instance_pred(self, instance_pred_entropy, instance_pred_accuracy, instance_pred_log_prob):
        self.instance_pred_entropy.copy_(instance_pred_entropy)
        self.instance_pred_accuracy.copy_(instance_pred_accuracy)
        self.instance_pred_log_prob.copy_(instance_pred_log_prob)

    def after_update(self):
        self.obs[0].copy_(self.obs[-1])
        self.recurrent_hidden_states[0].copy_(self.recurrent_hidden_states[-1])
        self.actor_features[0].copy_(self.actor_features[-1])
        self.masks[0].copy_(self.masks[-1])
        self.bad_masks[0].copy_(self.bad_masks[-1])
        self.level_seeds[0].copy_(self.level_seeds[-1])

    def reset(self, obs=None):
        self.obs = torch.zeros_like(self.obs)
        if obs is not None:
            self.obs[0].copy_(obs)
        self.recurrent_hidden_states = torch.zeros_like(self.recurrent_hidden_states)
        self.actor_features = torch.zeros_like(self.actor_features)
        self.rewards = torch.zeros_like(self.rewards)
        self.value_preds = torch.zeros_like(self.value_preds)
        self.instance_value_preds = torch.zeros_like(self.instance_value_preds)
        self.returns = torch.zeros_like(self.returns)
        self.action_log_probs = torch.zeros_like(self.action_log_probs)
        self.action_log_dist = torch.zeros_like(self.action_log_dist)
        self.actions = torch.zeros_like(self.actions)
        self.masks = torch.ones_like(self.masks)
        self.bad_masks = torch.ones_like(self.bad_masks)
        self.level_seeds = torch.zeros_like(self.level_seeds)
        self.instance_pred_entropy = torch.zeros_like(self.instance_pred_entropy)
        self.instance_pred_accuracy = torch.zeros_like(self.instance_pred_accuracy)
        self.instance_pred_log_prob = torch.zeros_like(self.instance_pred_log_prob)
        self.step = 0

    def compute_returns(self,
                        next_value,
                        gamma,
                        gae_lambda):
        self.value_preds[-1] = next_value
        gae = 0
        for step in reversed(range(self.rewards.size(0))):
            delta = self.rewards[step] + gamma * self.value_preds[
                step + 1] * self.masks[step +
                                        1] - self.value_preds[step]
            gae = delta + gamma * gae_lambda * self.masks[step +
                                                            1] * gae
            self.returns[step] = gae + self.value_preds[step]

    def feed_forward_generator(self,
                               advantages,
                               num_mini_batch=None,
                               mini_batch_size=None,
                               balanced_sampling=False,):
        num_steps, num_processes = self.rewards.size()[0:2]
        batch_size = num_processes * num_steps

        if mini_batch_size is None:
            assert batch_size >= num_mini_batch, (
                "PPO requires the number of processes ({}) "
                "* number of steps ({}) = {} "
                "to be greater than or equal to the number of PPO mini batches ({})."
                "".format(num_processes, num_steps, num_processes * num_steps,
                          num_mini_batch))
            mini_batch_size = batch_size // num_mini_batch

        if balanced_sampling:
            seed_indices = self.level_seeds.flatten()
            class_counts = torch.bincount(seed_indices)
            class_weights = 1.0 / class_counts.float()
            weights = class_weights[seed_indices].flatten()
            weights = weights / weights.sum()
            sampler = BatchSampler(WeightedRandomSampler(weights, len(weights), replacement=True),
                                   mini_batch_size,
                                   drop_last=True)
        else:
            sampler = BatchSampler(
                SubsetRandomSampler(range(batch_size)),
                mini_batch_size,
                drop_last=True)
     
        for indices in sampler:
            obs_batch = self.obs[:-1].view(-1, *self.obs.size()[2:])[indices]
            recurrent_hidden_states_batch = self.recurrent_hidden_states[:-1].view(
                -1, self.recurrent_hidden_states.size(-1))[indices]
            actor_features_batch = self.actor_features[:-1].view(
                -1, self.actor_features.size(-1))[indices]
            actions_batch = self.actions.view(-1,
                                            self.actions.size(-1))[indices]
            value_preds_batch = self.value_preds[:-1].view(-1, 1)[indices]
            instance_value_preds_batch = self.instance_value_preds[:-1].view(-1, 1)[indices]
            return_batch = self.returns[:-1].view(-1, 1)[indices]
            masks_batch = self.masks[:-1].view(-1, 1)[indices]
            old_action_log_probs_batch = self.action_log_probs.view(-1,
                                                                    1)[indices]
            level_seeds_batch = self.level_seeds.view(-1, 1)[indices]
            if advantages is None:
                adv_targ = None
            else:
                adv_targ = advantages.view(-1, 1)[indices]

            yield obs_batch, recurrent_hidden_states_batch, actor_features_batch, actions_batch, value_preds_batch, \
                instance_value_preds_batch, return_batch, masks_batch, old_action_log_probs_batch, adv_targ, \
                level_seeds_batch

    def recurrent_generator(self, advantages, num_mini_batch):
        num_processes = self.rewards.size(1)
        assert num_processes >= num_mini_batch, (
            "PPO requires the number of processes ({}) "
            "to be greater than or equal to the number of "
            "PPO mini batches ({}).".format(num_processes, num_mini_batch))
        num_envs_per_batch = num_processes // num_mini_batch
        perm = torch.randperm(num_processes)
        for start_ind in range(0, num_processes, num_envs_per_batch):
            obs_batch = []
            recurrent_hidden_states_batch = []
            actor_features_batch = []
            actions_batch = []
            value_preds_batch = []
            instance_value_preds_batch = []
            return_batch = []
            masks_batch = []
            old_action_log_probs_batch = []
            adv_targ = []
            level_seeds_batch = []

            for offset in range(num_envs_per_batch):
                ind = perm[start_ind + offset]
                obs_batch.append(self.obs[:-1, ind])
                recurrent_hidden_states_batch.append(
                    self.recurrent_hidden_states[0:1, ind])
                actor_features_batch.append(self.actor_features[:-1, ind])
                actions_batch.append(self.actions[:, ind])
                value_preds_batch.append(self.value_preds[:-1, ind])
                instance_value_preds_batch.append(self.instance_value_preds[:-1, ind])
                return_batch.append(self.returns[:-1, ind])
                masks_batch.append(self.masks[:-1, ind])
                old_action_log_probs_batch.append(
                    self.action_log_probs[:, ind])
                adv_targ.append(advantages[:, ind])
                level_seeds_batch.append(self.level_seeds[:, ind])

            T, N = self.num_steps, num_envs_per_batch
            # These are all tensors of size (T, N, -1)
            obs_batch = torch.stack(obs_batch, 1)
            actor_features_batch = torch.stack(actor_features_batch, 1)
            actions_batch = torch.stack(actions_batch, 1)
            value_preds_batch = torch.stack(value_preds_batch, 1)
            instance_value_preds_batch = torch.stack(instance_value_preds_batch, 1)
            return_batch = torch.stack(return_batch, 1)
            masks_batch = torch.stack(masks_batch, 1)
            old_action_log_probs_batch = torch.stack(
                old_action_log_probs_batch, 1)
            adv_targ = torch.stack(adv_targ, 1)
            level_seeds_batch = torch.stack(level_seeds_batch, 1)

            # States is just a (N, -1) tensor
            recurrent_hidden_states_batch = torch.stack(
                recurrent_hidden_states_batch, 1).view(N, -1)

            # Flatten the (T, N, ...) tensors to (T * N, ...)
            obs_batch = _flatten_helper(T, N, obs_batch)
            actor_features_batch = _flatten_helper(T, N, actor_features_batch)
            actions_batch = _flatten_helper(T, N, actions_batch)
            value_preds_batch = _flatten_helper(T, N, value_preds_batch)
            instance_value_preds_batch = _flatten_helper(T, N, instance_value_preds_batch)
            return_batch = _flatten_helper(T, N, return_batch)
            masks_batch = _flatten_helper(T, N, masks_batch)
            old_action_log_probs_batch = _flatten_helper(T, N, \
                    old_action_log_probs_batch)
            adv_targ = _flatten_helper(T, N, adv_targ)
            level_seeds_batch = _flatten_helper(T, N, level_seeds_batch)

            yield obs_batch, recurrent_hidden_states_batch, actor_features_batch, actions_batch, value_preds_batch, \
                instance_value_preds_batch, return_batch, masks_batch, old_action_log_probs_batch, adv_targ, \
                level_seeds_batch
