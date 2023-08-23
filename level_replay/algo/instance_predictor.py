import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

from level_replay.model import InstancePredictor


class InstancePredictorModel():

    def __init__(self,
                 instance_predictor: InstancePredictor,
                 num_mini_batch,
                 epoch,
                 lr=None,
                 eps=None,
                 max_grad_norm=None,
                 env_name=None):

        self.instance_predictor = instance_predictor
        self.num_mini_batch = num_mini_batch
        self.epoch = epoch
        self.max_grad_norm = max_grad_norm
        self.optimizer = optim.Adam(instance_predictor.parameters(), lr=lr, eps=eps)
        self.env_name = env_name

    def update(self, rollouts, is_recurrent=False):

        self.instance_predictor.train()

        advantages = torch.zeros_like(rollouts.returns[:-1])

        instance_pred_loss_epoch = 0
        instance_pred_entropy_epoch = 0
        instance_pred_accuracy_epoch = 0
        instance_pred_prob_epoch = 0

        for e in range(self.epoch):

            if is_recurrent:
                data_generator = rollouts.recurrent_generator(
                    advantages, self.num_mini_batch, balanced_sampling=False)
            else:
                data_generator = rollouts.feed_forward_generator(
                    advantages, self.num_mini_batch, balanced_sampling=False)

            for sample in data_generator:
                obs_batch, recurrent_hidden_states_batch, hidden_features, _, _, _, _, masks_batch, _, _, level_seeds \
                    = sample

                self.optimizer.zero_grad()
                instance_pred_dist = self.instance_predictor(hidden_features.detach())
                instance_logits = instance_pred_dist.logits
                instance_entropy = instance_pred_dist.entropy().mean()
                instance_prob = instance_pred_dist.log_prob(level_seeds.flatten().to(torch.int64)).exp().mean()
                instance_pred_accuracy = self.instance_predictor.accuracy(instance_logits, level_seeds).mean()
                instance_pred_loss = F.cross_entropy(instance_logits, level_seeds.flatten().to(torch.int64))
                instance_pred_loss.backward()
                nn.utils.clip_grad_norm_(self.instance_predictor.parameters(), self.max_grad_norm)
                self.optimizer.step()

                instance_pred_loss_epoch += instance_pred_loss.item()
                instance_pred_entropy_epoch += instance_entropy.item()
                instance_pred_accuracy_epoch += instance_pred_accuracy.item()
                instance_pred_prob_epoch += instance_prob.item()

        num_updates = self.epoch * self.num_mini_batch

        instance_pred_loss_epoch /= num_updates
        instance_pred_entropy_epoch /= num_updates
        instance_pred_accuracy_epoch /= num_updates
        instance_pred_prob_epoch /= num_updates

        stats = {
            'instance_pred_loss': instance_pred_loss_epoch,
            'instance_pred_entropy': instance_pred_entropy_epoch,
            'instance_pred_accuracy': instance_pred_accuracy_epoch,
            'instance_pred_prob': instance_pred_prob_epoch,
        }

        return stats

    def predict(self, hidden_features, labels):
        original_shape = hidden_features.shape
        hidden_features = hidden_features.reshape(-1, hidden_features.shape[-1]).detach()
        labels = labels.flatten().to(torch.int64)
        instance_pred_dist = self.instance_predictor(hidden_features)
        instance_logits = instance_pred_dist.logits
        instance_logprob = instance_pred_dist.log_prob(labels)
        entropy = instance_pred_dist.entropy()
        accuracy = self.instance_predictor.correct_prediction(instance_logits, labels)
        report = self.instance_predictor.classification_report(instance_logits, labels)

        instance_logprob = instance_logprob.reshape(original_shape[:-1]).unsqueeze(-1)
        entropy = entropy.reshape(original_shape[:-1]).unsqueeze(-1)
        accuracy = accuracy.reshape(original_shape[:-1]).unsqueeze(-1)

        stats = {
            'instance_pred_log_prob': instance_logprob,
            'instance_pred_entropy': entropy,
            'instance_pred_accuracy': accuracy,
            'classification_report': report,
        }

        return stats

    def reset(self):

        # seed_indices = rollouts.level_seeds.flatten()
        # class_counts = torch.bincount(seed_indices)
        # class_indices = torch.nonzero(class_counts).flatten()
        # self.auxiliary_head.reset(num_instances=len(class_indices))
        # self.optimizer_aux = optim.Adam(self.auxiliary_head.parameters(), lr=self.optimizer_aux.defaults['lr'],
        #                                 eps=self.optimizer_aux.defaults['eps'])

        raise NotImplementedError
