# Copyright (c) 2017 Ilya Kostrikov
# 
# Licensed under the MIT License;
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     https://opensource.org/licenses/MIT
#
# This file is a modified version of:
# https://github.com/ikostrikov/pytorch-a2c-ppo-acktr-gail/blob/master/a2c_ppo_acktr/algo/ppo.py

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import random 

class PPO():
    """
    Vanilla PPO
    """
    def __init__(self,
                 actor_critic,
                 clip_param,
                 ppo_epoch,
                 num_mini_batch,
                 value_loss_coef,
                 entropy_coef,
                 lr=None,
                 eps=None,
                 max_grad_norm=None,
                 env_name=None,
                 auxiliary_head=None):

        self.actor_critic = actor_critic
        self.auxiliary_head = auxiliary_head if auxiliary_head is not None else None

        self.clip_param = clip_param
        self.ppo_epoch = ppo_epoch
        self.num_mini_batch = num_mini_batch

        self.value_loss_coef = value_loss_coef
        self.entropy_coef = entropy_coef

        self.max_grad_norm = max_grad_norm

        self.optimizer = optim.Adam(actor_critic.parameters(), lr=lr, eps=eps)
        self.optimizer_aux = optim.Adam(auxiliary_head.parameters(), lr=lr, eps=eps) if auxiliary_head is not None else \
            None

        self.env_name = env_name

    def update(self, rollouts, reset_predictor=False):
        advantages = rollouts.returns[:-1] - rollouts.value_preds[:-1]
        advantages = (advantages - advantages.mean()) / (
            advantages.std() + 1e-5)

        value_loss_epoch = 0
        action_loss_epoch = 0
        dist_entropy_epoch = 0
        instance_pred_loss_epoch = 0
        instance_pred_entropy_epoch = 0
        instance_pred_accuracy_epoch = 0
        instance_pred_precision_epoch = 0

        for e in range(self.ppo_epoch):
            if self.actor_critic.is_recurrent:
                data_generator = rollouts.recurrent_generator(
                    advantages, self.num_mini_batch)
            else:
                data_generator = rollouts.feed_forward_generator(
                    advantages, self.num_mini_batch)

            for sample in data_generator:
                obs_batch, recurrent_hidden_states_batch, actions_batch, value_preds_batch, \
                instance_value_preds_batch, return_batch, masks_batch, old_action_log_probs_batch, \
                        adv_targ, level_seeds = sample
                
                values, instance_values, action_log_probs, dist_entropy, _ = self.actor_critic.evaluate_actions(
                    obs_batch, recurrent_hidden_states_batch, masks_batch,
                    actions_batch, level_seeds)
                    
                ratio = torch.exp(action_log_probs -
                                  old_action_log_probs_batch)
                surr1 = ratio * adv_targ
                surr2 = torch.clamp(ratio, 1.0 - self.clip_param,
                                    1.0 + self.clip_param) * adv_targ
                action_loss = -torch.min(surr1, surr2).mean()

                value_pred_clipped = value_preds_batch + \
                    (values - value_preds_batch).clamp(-self.clip_param, self.clip_param)
                value_losses = (values - return_batch).pow(2)
                value_losses_clipped = (
                    value_pred_clipped - return_batch).pow(2)
                value_loss = 0.5 * torch.max(value_losses,
                                                value_losses_clipped).mean()

                # instance_value_pred_clipped = instance_value_preds_batch + \
                #     (instance_values - instance_value_preds_batch).clamp(-self.clip_param, self.clip_param)
                # instance_value_losses = (instance_values - return_batch).pow(2)
                # instance_value_losses_clipped = (
                #     instance_value_pred_clipped - return_batch).pow(2)
                # instance_value_loss = 0.5 * torch.max(instance_value_losses,
                #                                 instance_value_losses_clipped).mean()

                self.optimizer.zero_grad()
                # TODO: add instance_value_loss?
                loss = (value_loss*self.value_loss_coef +
                        action_loss - dist_entropy*self.entropy_coef)
                loss.backward()
                nn.utils.clip_grad_norm_(self.actor_critic.parameters(),
                                        self.max_grad_norm)
                self.optimizer.step()


        if self.auxiliary_head is not None:

            # trains auxiliary head
            if reset_predictor:
                seed_indices = rollouts.level_seeds.flatten()
                class_counts = torch.bincount(seed_indices)
                class_indices = torch.nonzero(class_counts).flatten()
                self.auxiliary_head.reset(num_instances=len(class_indices))
                self.optimizer_aux = optim.Adam(self.auxiliary_head.parameters(), lr=self.optimizer_aux.defaults['lr'],
                                                eps=self.optimizer_aux.defaults['eps'])

            if self.actor_critic.is_recurrent:
                data_generator_aux = rollouts.recurrent_generator(
                    advantages, self.num_mini_batch_predictor, balanced_sampling=True)
            else:
                data_generator_aux = rollouts.feed_forward_generator(
                    advantages, self.num_mini_batch_predictor, balanced_sampling=True)
            for e in range(self.predictor_epoch):
                for sample in data_generator_aux:
                    obs_batch, recurrent_hidden_states_batch, _, _, _, _, masks_batch, _, _, level_seeds = sample
                    #TODO: this is inefficient, best to store hidden features in rollouts at the last epoch.
                    hidden_features = self.actor_critic.get_hidden_features(obs_batch, recurrent_hidden_states_batch,
                                                                            masks_batch)

                    self.optimizer_aux.zero_grad()
                    instance_pred_dist = self.auxiliary_head(hidden_features.detach())
                    instance_logits = instance_pred_dist.logits
                    instance_entropy = instance_pred_dist.entropy().mean()
                    instance_pred_accuracy = self.auxiliary_head.accuracy(instance_logits, level_seeds).mean()
                    instance_pred_precision = self.auxiliary_head.precision(instance_logits, level_seeds).mean()
                    instance_pred_loss = F.cross_entropy(instance_logits, level_seeds.flatten().to(torch.int64))
                    instance_pred_loss.backward()
                    nn.utils.clip_grad_norm_(self.auxiliary_head.parameters(),
                                            self.max_grad_norm)
                    self.optimizer_aux.step()

        value_loss_epoch += value_loss.item()
        action_loss_epoch += action_loss.item()
        dist_entropy_epoch += dist_entropy.item()
        if self.auxiliary_head is not None:
            instance_pred_loss_epoch += instance_pred_loss.item()
            instance_pred_entropy_epoch += instance_entropy.item()
            instance_pred_accuracy_epoch += instance_pred_accuracy.item()
            instance_pred_precision_epoch += instance_pred_precision.item()

        num_updates = self.ppo_epoch * self.num_mini_batch

        value_loss_epoch /= num_updates
        action_loss_epoch /= num_updates
        dist_entropy_epoch /= num_updates
        instance_pred_loss_epoch /= num_updates
        instance_pred_entropy_epoch /= num_updates
        instance_pred_accuracy_epoch /= num_updates
        instance_pred_precision_epoch /= num_updates

        return value_loss_epoch, action_loss_epoch, dist_entropy_epoch, instance_pred_loss_epoch, \
               instance_pred_entropy_epoch, instance_pred_accuracy_epoch, instance_pred_precision_epoch
