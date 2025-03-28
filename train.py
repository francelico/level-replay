# Copyright (c) Facebook, Inc. and its affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
#
# Implementation for mixed sampling strategies and instance prediction by Samuel Garcin.

import copy
import os
import shutil
import sys
import time
from collections import deque, defaultdict
import timeit
import logging

import numpy as np
import torch
from baselines.logger import HumanOutputFormat

from level_replay import algo, utils
from level_replay.model import model_for_env_name
from level_replay.storage import RolloutStorage
from level_replay.file_writer import FileWriter
from level_replay.envs import make_lr_venv
from level_replay.arguments import parser
from test import evaluate
from level_replay.model import InstancePredictor

os.environ["OMP_NUM_THREADS"] = "1"

last_checkpoint_time = None


def train(args, seeds):
    global last_checkpoint_time
    args.cuda = not args.no_cuda and torch.cuda.is_available()
    device = torch.device("cuda:0" if args.cuda else "cpu")
    if 'cuda' in device.type:
        print('Using CUDA\n')

    torch.set_num_threads(1)

    utils.seed(args.seed)

    # bootstrap from an existing run
    if args.bootstrap_from_dir is not None:
        basepath = os.path.expandvars(os.path.expanduser(os.path.dirname(args.log_dir)))
        bootstrap_dir = os.path.join(basepath, args.bootstrap_from_dir)

        assert args.xpid is not None
        run_dir = os.path.join(os.path.expandvars(os.path.expanduser(args.log_dir)), args.xpid)
        if os.path.exists(run_dir):
            print("Run dir already exists: {}".format(run_dir))
        else:
            shutil.copytree(bootstrap_dir, run_dir)

    # Configure logging
    if args.xpid is None:
        args.xpid = "lr-%s" % time.strftime("%Y%m%d-%H%M%S")
    log_dir = os.path.expandvars(os.path.expanduser(args.log_dir))
    plogger = FileWriter(
        xpid=args.xpid, xp_args=args.__dict__, rootdir=log_dir,
        seeds=seeds,
    )
    stdout_logger = HumanOutputFormat(sys.stdout)

    if plogger.completed:
        print("Experiment already completed ({}).".format(plogger.basepath))
        return

    backup_update = full_backup_update(args)

    # Configure actor envs
    start_level = 0
    if args.full_train_distribution:
        num_levels = 0
        level_sampler_args = None
        seeds = None
        eval_level_sampler_args = None
    else:
        num_levels = 1
        level_sampler_args = dict(
            num_actors=args.num_processes,
            strategy=args.level_replay_strategy,
            replay_schedule=args.level_replay_schedule,
            score_transform=args.level_replay_score_transform,
            temperature=args.level_replay_temperature,
            eps=args.level_replay_eps,
            rho=args.level_replay_rho,
            nu=args.level_replay_nu, 
            alpha=args.level_replay_alpha,
            staleness_coef=args.staleness_coef,
            staleness_transform=args.staleness_transform,
            staleness_temperature=args.staleness_temperature,
            secondary_strategy=args.level_replay_secondary_strategy,
            secondary_strategy_coef=args.level_replay_secondary_strategy_coef_start,
            secondary_score_transform=args.level_replay_secondary_score_transform,
            secondary_temperature=args.level_replay_secondary_temperature,
        )
        eval_level_sampler_args = dict(
            num_actors=args.num_train_eval_processes,
            strategy="random",
        )
    envs, level_sampler = make_lr_venv(
        num_envs=args.num_processes, env_name=args.env_name,
        seeds=seeds, device=device,
        num_levels=num_levels, start_level=start_level,
        no_ret_normalization=args.no_ret_normalization,
        distribution_mode=args.distribution_mode,
        paint_vel_info=args.paint_vel_info,
        level_sampler_args=level_sampler_args)

    eval_envs, eval_level_sampler = make_lr_venv(
        num_envs=args.num_train_eval_processes, env_name=args.env_name,
        seeds=seeds, device=device,
        num_levels=num_levels, start_level=start_level,
        no_ret_normalization=args.no_ret_normalization,
        distribution_mode=args.distribution_mode,
        paint_vel_info=args.paint_vel_info,
        level_sampler_args=eval_level_sampler_args)
    
    is_minigrid = args.env_name.startswith('MiniGrid')

    actor_critic = model_for_env_name(args, envs)       
    actor_critic.to(device)

    if args.instance_predictor:
        instance_predictor = InstancePredictor(actor_critic.base.output_size, args.instance_predictor_hidden_size, args.num_train_seeds)
        instance_predictor.to(device)
    else:
        instance_predictor = None

    rollouts = RolloutStorage(args.num_steps, args.num_processes,
                                envs.observation_space.shape, envs.action_space,
                                actor_critic.recurrent_hidden_state_size,
                                actor_critic.actor_feature_size)
    rollouts.to(device)
    eval_rollouts = RolloutStorage(args.num_train_eval_steps, args.num_train_eval_processes,
                                   envs.observation_space.shape, envs.action_space,
                                   actor_critic.recurrent_hidden_state_size,
                                   actor_critic.actor_feature_size)
    eval_rollouts.to(device)
    batch_size = int(args.num_processes * args.num_steps / args.num_mini_batch)
    eval_batch_size = int(args.num_train_eval_processes * args.num_train_eval_steps /
                          args.num_mini_batch_instance_predictor)

    def save_checkpoint(update_number: int = None):
        if args.disable_checkpoint:
            return

        if update_number is None:
            filename = "model.tar"
        else:
            filename = f"model_{update_number}.tar"


        checkpointpath = os.path.expandvars(
            os.path.expanduser("%s/%s/%s" % (log_dir, args.xpid, filename))
        )

        logging.info("Saving checkpoint to %s", checkpointpath)
        state_dict = {
                "model_state_dict": actor_critic.state_dict(),
                "optimizer_state_dict": agent.optimizer.state_dict(),
                "instance_predictor_state_dict": instance_predictor.state_dict(),
                "instance_predictor_optimizer_state_dict": instance_predictor_model.optimizer.state_dict(),
                "args": vars(args),
                "seeds": level_sampler.seeds,
                "seed_scores": level_sampler.seed_scores,
                "seed_staleness": level_sampler.seed_staleness,
                "unseen_seed_weights": level_sampler.unseen_seed_weights,
                "next_seed_index": level_sampler.next_seed_index,
            }
        utils.safe_checkpoint(state_dict, checkpointpath)

        # remove old checkpoints
        old_checkpoint_filenames = []
        for file in os.listdir(os.path.expandvars(os.path.expanduser(plogger.basepath))):
            if file.endswith(".tar") and file != filename:
                old_checkpoint_filenames.append(file)
        for file in old_checkpoint_filenames:
            os.remove(os.path.expandvars(os.path.expanduser(plogger.basepath + '/' + file)))

    def make_full_backup(num_update):
        save_checkpoint(update_number=num_update)
        base_path = os.path.expandvars(os.path.expanduser("%s/%s" % (log_dir, args.xpid)))
        shutil.copytree(base_path, base_path + '_bkup')

    def load_checkpoint(checkpoint, agent, level_sampler, instance_predictor_model):
        agent.actor_critic.load_state_dict(checkpoint["model_state_dict"])
        agent.actor_critic.to(device)
        agent.optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
        instance_predictor_model.instance_predictor.load_state_dict(checkpoint["instance_predictor_state_dict"])
        instance_predictor_model.instance_predictor.to(device)
        instance_predictor_model.optimizer.load_state_dict(checkpoint["instance_predictor_optimizer_state_dict"])
        level_sampler.seeds = checkpoint["seeds"]
        level_sampler.seed_scores = checkpoint["seed_scores"]
        level_sampler.seed_staleness = checkpoint["seed_staleness"]
        level_sampler.unseen_seed_weights = checkpoint["unseen_seed_weights"]
        level_sampler.next_seed_index = checkpoint["next_seed_index"]

    agent = algo.PPO(
        actor_critic,
        args.clip_param,
        args.ppo_epoch,
        args.num_mini_batch,
        args.value_loss_coef,
        args.entropy_coef,
        lr=args.lr,
        eps=args.eps,
        max_grad_norm=args.max_grad_norm,
        env_name=args.env_name)

    if instance_predictor is not None:
        instance_predictor_model = algo.InstancePredictorModel(
            instance_predictor,
            args.num_mini_batch_instance_predictor,
            epoch=args.instance_predictor_epoch,
            lr=args.lr_instance_predictor,
            eps=args.eps_instance_predictor,
            max_grad_norm=args.max_grad_norm_instance_predictor,
            env_name=args.env_name,
        )
    else:
        instance_predictor_model = None

    # === Load checkpoint ===
    if args.checkpoint:
        restart_count = int(os.environ.get("SLURM_RESTART_COUNT", 0))
        if restart_count:
            logging.info(f"This job has already been restarted {restart_count} times by SLURM.")
        # find all .tar files in the log directory
        checkpoint_filenames = []
        for file in os.listdir(os.path.expandvars(os.path.expanduser(plogger.basepath))):
            if file.endswith(".tar") and file.startswith("model_"):
                checkpoint_filenames.append(file)
                break
        if len(checkpoint_filenames) > 0:
            assert len(checkpoint_filenames) == 1, "More than one checkpoint found. Aborting."
            file = checkpoint_filenames[0]
            checkpoint_path = os.path.expandvars(os.path.expanduser(plogger.basepath + '/' + file))
            start_at_update = int(file.split('_')[1].split('.')[0])
            print(f'Checkpoint found at update {start_at_update}. Loading Checkpoint States\n')
            checkpoint = torch.load(checkpoint_path, map_location=lambda storage, loc: storage)
            load_checkpoint(checkpoint, agent, level_sampler, instance_predictor_model)
            logging.info(f"Resuming preempted job after {start_at_update} updates\n") # 0-indexed next update
            logging.info(f"Clearing log files after {start_at_update} updates\n")
            plogger.delete_after_update(start_at_update)
        else:
            start_at_update = -1
            logging.info("No checkpoint found. Starting from scratch\n")
            shutil.rmtree(plogger.basepath)
            plogger = FileWriter(
                xpid=args.xpid, xp_args=args.__dict__, rootdir=log_dir,
                seeds=seeds,
            )
    else:
        start_at_update = -1
    assert plogger.num_duplicates == 0, "Duplicate data detected within log directory. Aborting."

    episode_rewards = deque(maxlen=10)
    num_updates = int(
        args.num_env_steps) // args.num_steps // args.num_processes

    timer = timeit.default_timer
    update_start_time = timer()
    first_reset = True
    for j in range(start_at_update + 1, num_updates):
        level_sampler.secondary_strategy_coef = schedule_secondary_strategy_coef(args, j)
        actor_critic.train()
        if instance_predictor is not None:
            instance_predictor.eval()
        rollouts, episode_rewards, instance_prediction_stats = collect_rollouts(args,
                                                     rollouts,
                                                     envs,
                                                     actor_critic,
                                                     instance_predictor_model=instance_predictor_model,
                                                     episode_rewards=episode_rewards,
                                                     reset=first_reset)
        first_reset = False
        # Update level sampler
        if level_sampler:
            level_sampler.update_with_rollouts(rollouts, instance_prediction_stats)

        value_loss, action_loss, dist_entropy = agent.update(rollouts)
        rollouts.after_update()
        if level_sampler:
            level_sampler.after_update()

        logging.info(f"\nEvaluating on {args.num_train_eval_episodes} train levels...\n  ")
        eval_rollouts, train_eval_episode_rewards, _ = collect_rollouts(args,
                                                               eval_rollouts,
                                                               eval_envs,
                                                               actor_critic,
                                                               instance_predictor_model=None,
                                                               num_episodes=args.num_train_eval_episodes,
                                                               reset=True)
        if instance_predictor_model is not None:
            instance_pred_train_stats = instance_predictor_model.update(eval_rollouts)
        else:
            instance_pred_train_stats = defaultdict(int)

        # Log stats every log_interval updates or if it is the last update
        if (j % args.log_interval == 0 and len(episode_rewards) > 1) or j == num_updates - 1:
            total_num_steps = (j + 1) * args.num_processes * args.num_steps

            update_end_time = timer()
            num_interval_updates = 1 if j == 0 else args.log_interval
            sps = num_interval_updates*(args.num_processes * args.num_steps) / (update_end_time - update_start_time)
            update_start_time = update_end_time

            logging.info(f"\nUpdate {j} done, {total_num_steps} steps\n  ")
            logging.info(f"\nEvaluating on {args.num_test_seeds} test levels...\n  ")
            eval_episode_rewards, eval_seeds = evaluate(args, actor_critic, args.num_test_seeds, device,
                                            start_level=args.num_train_seeds)

            stats = { 
                "step": total_num_steps,
                "pg_loss": action_loss,
                "value_loss": value_loss,
                "dist_entropy": dist_entropy,
                "instance_pred_loss_train": instance_pred_train_stats["instance_pred_loss"],
                "instance_pred_entropy_train": instance_pred_train_stats["instance_pred_entropy"],
                "instance_pred_accuracy_train": instance_pred_train_stats["instance_pred_accuracy"],
                "instance_pred_prob_train": instance_pred_train_stats["instance_pred_prob"],
                "coef_secondary": level_sampler.secondary_strategy_coef,
                "train:mean_episode_return": np.mean(episode_rewards),
                "train:median_episode_return": np.median(episode_rewards),
                "test:mean_episode_return": np.mean(eval_episode_rewards),
                "test:median_episode_return": np.median(eval_episode_rewards),
                "train_eval:mean_episode_return": np.mean(train_eval_episode_rewards),
                "train_eval:median_episode_return": np.median(train_eval_episode_rewards),
                "sps": sps,
            }
            if is_minigrid:
                stats["train:success_rate"] = np.mean(np.array(episode_rewards) > 0)
                stats["train_eval:success_rate"] = np.mean(np.array(train_eval_episode_rewards) > 0)
                stats["test:success_rate"] = np.mean(np.array(eval_episode_rewards) > 0)

            if j == num_updates - 1:
                logging.info(f"\nLast update: Evaluating on {args.final_num_test_seeds} test levels...\n  ")
                final_eval_episode_rewards, final_eval_seeds = evaluate(args, actor_critic, args.final_num_test_seeds, device,
                                                                        start_level=args.num_train_seeds)
                final_train_eval_episode_rewards, final_train_eval_seeds = evaluate(args, actor_critic, args.num_final_train_eval_episodes, device, seeds=level_sampler.seeds)

                mean_final_eval_episode_rewards = np.mean(final_eval_episode_rewards)
                median_final_eval_episode_rewards = np.median(final_eval_episode_rewards)
                
                plogger.log_final_test_eval({
                    'num_test_seeds': args.final_num_test_seeds,
                    'mean_episode_return': mean_final_eval_episode_rewards,
                    'median_episode_return': median_final_eval_episode_rewards
                })

                plogger.log_final_train_eval(final_train_eval_episode_rewards, final_train_eval_seeds)

            plogger.log(stats)
            if args.verbose:
                stdout_logger.writekvs(stats)

        # Log level weights
        if level_sampler and j % args.weight_log_interval == 0:
            plogger.log_level_weights(level_sampler.sample_weights())
            plogger.log_level_returns(level_sampler.sample_level_returns())
            plogger.log_level_value_loss(level_sampler.sample_level_value_loss())
            plogger.log_level_instance_value_loss(level_sampler.sample_level_instance_value_loss())
            plogger.log_instance_pred_entropy(level_sampler.sample_instance_pred_entropy())
            plogger.log_instance_pred_accuracy(level_sampler.sample_instance_pred_accuracy())
            plogger.log_instance_pred_log_prob(level_sampler.sample_instance_pred_log_prob())
            plogger.log_instance_pred_prob(level_sampler.sample_instance_pred_prob())
            plogger.log_instance_pred_precision(level_sampler.sample_instance_pred_precision())
            plogger.log_instance_pred_recall(level_sampler.sample_instance_pred_recall())
            plogger.log_instance_pred_f1(level_sampler.sample_instance_pred_f1())
        # could also only clear buffers when we write logs however it may lead to stale data. this is more consistent.
        level_sampler.after_logging()

        # Checkpoint 
        timer = timeit.default_timer
        if last_checkpoint_time is None:
            last_checkpoint_time = timer()
        try:
            if j == num_updates - 1:
                save_checkpoint()
            elif args.save_interval > 0 and timer() - last_checkpoint_time > args.save_interval * 60: # Save every args.save_interval min.
                save_checkpoint(update_number=j)
                last_checkpoint_time = timer()

            if j == backup_update:
                make_full_backup(j)
        except KeyboardInterrupt:
            return
    plogger.close(successful=True)


def generate_seeds(num_seeds, base_seed=0):
    return [base_seed + i for i in range(num_seeds)]


def load_seeds(seed_path):
    seed_path = os.path.expandvars(os.path.expanduser(seed_path))
    seeds = open(seed_path).readlines()
    return [int(s) for s in seeds]


def schedule_secondary_strategy_coef(args, num_update):
    if args.level_replay_secondary_strategy == "off":
        return 0.0
    num_updates = int(args.num_env_steps) // args.num_steps // args.num_processes
    start_coef = args.level_replay_secondary_strategy_coef_start
    end_coef = args.level_replay_secondary_strategy_coef_end
    start_update = int(num_updates * args.level_replay_secondary_strategy_fraction_start)
    end_update = int(num_updates * args.level_replay_secondary_strategy_fraction_end)
    delta = (end_coef - start_coef) / (end_update - start_update)
    if num_update < start_update:
        return 0.0
    elif num_update >= end_update:
        return end_coef
    else:
        return start_coef + delta * (num_update - start_update)


def full_backup_update(args):
    num_updates = int(args.num_env_steps) // args.num_steps // args.num_processes
    full_backup_update = int(num_updates * args.backup_fraction) - 1
    return full_backup_update


def collect_rollouts(
        args,
        rollouts,
        envs,
        actor_critic,
        instance_predictor_model=None,
        num_episodes=10,
        episode_rewards=None,
        reset = False,
        deterministic=False,
):

    if reset:
        obs, level_seeds = envs.reset()
        rollouts.reset(obs)
        level_seeds = level_seeds.unsqueeze(-1)
    else:
        level_seeds = rollouts.level_seeds[0].clone()

    if episode_rewards is None:
        episode_rewards = deque(maxlen=num_episodes)
    else:
        num_episodes = None

    for step in range(rollouts.num_steps):
        with torch.no_grad():
            obs_id = rollouts.obs[step]
            value, instance_value, action, action_log_dist, recurrent_hidden_states, actor_features = actor_critic.act(
                obs_id,
                rollouts.recurrent_hidden_states[step],
                rollouts.masks[step],
                level_seeds=level_seeds,
                deterministic=deterministic)
            action_log_prob = action_log_dist.gather(-1, action)

        # Obser reward and next obs
        obs, reward, done, infos = envs.step(action)

        # Reset all done levels by sampling from level sampler
        for i, info in enumerate(infos):
            if 'episode' in info.keys():
                episode_rewards.append(info['episode']['r'])

            level_seeds[i][0] = info['level_seed']

        # If done then clean the history of observations.
        masks = torch.FloatTensor(
            [[0.0] if done_ else [1.0] for done_ in done])
        bad_masks = torch.FloatTensor(
            [[0.0] if 'bad_transition' in info.keys() else [1.0]
             for info in infos])

        rollouts.insert(
            obs, recurrent_hidden_states, actor_features.detach(),
            action, action_log_prob, action_log_dist,
            value, instance_value, reward, masks, bad_masks, level_seeds)

    with torch.no_grad():
        obs_id = rollouts.obs[-1]
        next_value = actor_critic.get_value(
            obs_id, rollouts.recurrent_hidden_states[-1],
            rollouts.masks[-1]).detach()

    rollouts.compute_returns(next_value, args.gamma, args.gae_lambda)

    with torch.no_grad():
        if instance_predictor_model is not None:
            instance_prediction_stats = instance_predictor_model.predict(rollouts.actor_features[:-1], rollouts.level_seeds)
            instance_pred_entropy_rollouts = instance_prediction_stats['instance_pred_entropy']
            instance_pred_accuracy_rollouts = instance_prediction_stats['instance_pred_accuracy']
            instance_pred_log_prob = instance_prediction_stats['instance_pred_log_prob']
        else:
            instance_prediction_stats = None
            instance_pred_entropy_rollouts = torch.zeros_like(value)
            instance_pred_accuracy_rollouts = torch.zeros_like(value)
            instance_pred_log_prob = torch.zeros_like(value)

    rollouts.insert_instance_pred(instance_pred_entropy_rollouts, instance_pred_accuracy_rollouts, instance_pred_log_prob)

    if num_episodes is not None:
        while len(episode_rewards) < num_episodes:
            with torch.no_grad():
                _, _, action, _, recurrent_hidden_states, _ = actor_critic.act(
                    obs,
                    recurrent_hidden_states,
                    masks,
                    level_seeds=torch.zeros_like(level_seeds),
                    deterministic=deterministic)

            obs, _, done, infos = envs.step(action)

            masks = torch.tensor(
                [[0.0] if done_ else [1.0] for done_ in done],
                dtype=torch.float32,
                device=rollouts.device)

            for i, info in enumerate(infos):
                if 'episode' in info.keys():
                    episode_rewards.append(info['episode']['r'])

    return rollouts, episode_rewards, instance_prediction_stats


if __name__ == "__main__":
    args = parser.parse_args()

    if args.verbose:
        logging.getLogger().setLevel(logging.INFO)
    else:
        logging.disable(logging.CRITICAL)

    if args.seed_path:
        train_seeds = load_seeds(args.seed_path)
    else:
        train_seeds = generate_seeds(args.num_train_seeds)

    train(args, train_seeds)
