# 本文件主要代码由 AI 助手完成。
# 多进程 PPO：各环境并行采样，按环境计算 GAE，再合并更新同一网络。
from pathlib import Path
from functools import partial
from datetime import datetime
import argparse
import math
import sys
import time

import gymnasium as gym
import numpy as np
import torch
import torch.nn.functional as F

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))
from twice_step.piper_reach.ppo.env import PiperReachEnv
from twice_step.ppo_pendulum.networks import Ppo_net
from twice_step.ppo_cartpole.advantages import compute_gae
from twice_step.ppo_cartpole.losses import computer_ppo_loss


if __name__ == "__main__":
    # 默认 4 个环境、每环境采集 128 步，每批合计 512 条转移。
    seed = 42
    num_envs = 4
    rollout_steps = 128
    num_batches = 200
    update_epochs = 10
    minibatch_size = 64
    hidden_dimension = [512, 256, 128]
    activation = "elu"
    gamma = 0.99
    gae_lambda = 0.95
    learning_rate = 3e-4
    clip_epsilon = 0.2
    value_coef = 0.5
    entropy_coef = 0.0  # 现有网络返回高斯熵，而不是 tanh 后分布的熵。
    max_grad_norm = 0.5
    real_time = True

    parser = argparse.ArgumentParser(description="Parallel PPO training for Piper Reach")
    parser.add_argument("--no-render", action="store_true", help="所有环境无窗口运行")
    parser.add_argument("--num-envs", type=int, default=num_envs)
    parser.add_argument("--batches", type=int, default=num_batches)
    parser.add_argument("--rollout-steps", type=int, default=rollout_steps, help="每个环境每批采样的步数")
    parser.add_argument("--epochs", type=int, default=update_epochs)
    parser.add_argument("--minibatch-size", type=int, default=minibatch_size)
    parser.add_argument("--checkpoint-dir", type=Path, default=Path(__file__).resolve().parent)
    args = parser.parse_args()
    if min(args.num_envs, args.batches, args.rollout_steps, args.epochs, args.minibatch_size) < 1:
        parser.error("num-envs、batches、rollout-steps、epochs 和 minibatch-size 必须大于 0")

    torch.manual_seed(seed)
    np.random.seed(seed)
    torch.set_num_threads(1)
    # 工作进程各自独立仿真；所有画面由主进程中的单一窗口展示。
    factories = [partial(PiperReachEnv, render_mode=None, capture_visual_state=not args.no_render)
                 for _ in range(args.num_envs)]
    envs = gym.vector.AsyncVectorEnv(
        factories, context="spawn", copy=True, autoreset_mode=gym.vector.AutoresetMode.SAME_STEP
    )
    envs.action_space.seed(seed)
    state_dimension = envs.single_observation_space.shape[0]
    action_dimension = envs.single_action_space.shape[0]
    environment_config = {
        key: envs.get_attr(key)[0]
        for key in ("frame_skip", "max_steps", "success_distance", "joint_step", "min_goal_distance",
                    "linear_speed_cost", "angular_speed_cost")
    }
    simulation_timestep = float(envs.get_attr("sim_timestep")[0])
    model = Ppo_net(state_dimension, action_dimension, hidden_dimension, activation=activation)
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    total_env_steps = 0
    total_updates = 0
    episodes = 0
    episode_reward = np.zeros(args.num_envs, dtype=np.float64)
    episode_rewards = []
    episode_distances = []
    episode_successes = []
    window_closed = False
    training_started = time.monotonic()
    viewer = None

    try:
        states_now, initial_infos = envs.reset(seed=seed)  # 各环境依次使用 seed、seed+1、...
        if not args.no_render:
            from twice_step.piper_reach.ppo.parallel_viewer import ParallelViewer
            viewer = ParallelViewer(args.num_envs)
            viewer.sync(initial_infos["visual_qpos"], initial_infos["visual_target"])

        for batch_index in range(1, args.batches + 1):
            transitions = []
            for _ in range(args.rollout_steps):
                if viewer is not None and not viewer.is_running():
                    window_closed = True
                    break
                started_at = time.monotonic()
                states_tensor = torch.as_tensor(states_now, dtype=torch.float32)
                with torch.no_grad():
                    raw_actions, raw_log_probs, values, _ = model(states_tensor)
                    actions = raw_actions.tanh()
                    log_jacobian = 2 * (math.log(2) - raw_actions - F.softplus(-2 * raw_actions))
                    old_log_probs = raw_log_probs - log_jacobian.sum(-1)

                next_states, rewards, terminated, truncated, infos = envs.step(actions.numpy())
                done = terminated | truncated
                # SAME_STEP 返回的是下一局初始状态；对截断自举必须保存真正的终局状态。
                next_for_value = next_states.copy()
                if done.any():
                    for i in np.flatnonzero(done):
                        next_for_value[i] = infos["final_obs"][i]
                transitions.append((states_now, raw_actions, rewards, next_for_value,
                                    terminated, truncated, old_log_probs, values))
                total_env_steps += args.num_envs
                episode_reward += rewards

                for i in np.flatnonzero(done):
                    final = infos["final_info"]
                    episodes += 1
                    episode_rewards.append(float(episode_reward[i]))
                    episode_distances.append(float(final["distance"][i]))
                    episode_successes.append(bool(final["is_success"][i]))
                    recent_successes = episode_successes[-10:]
                    recent_distances = episode_distances[-10:]
                    print(f"局数 {episodes} | 环境 {i} | 奖励 {episode_reward[i]:.3f} | "
                          f"最终距离 {final['distance'][i]:.4f} 米 | 成功 {final['is_success'][i]} | "
                          f"最近{len(recent_successes)}局成功率 {np.mean(recent_successes):.1%} | "
                          f"平均最终距离 {np.mean(recent_distances):.4f} 米 | "
                          f"环境步数 {total_env_steps} | 更新次数 {total_updates}", flush=True)
                    episode_reward[i] = 0.0
                states_now = next_states

                if viewer is not None:
                    # SAME_STEP 的顶层 info 对应重置后的新回合，与 next_states 一致。
                    viewer.sync(infos["visual_qpos"], infos["visual_target"])
                    if not viewer.is_running():
                        window_closed = True
                        break
                if viewer is not None and real_time:
                    elapsed = time.monotonic() - started_at
                    time.sleep(max(0.0, environment_config["frame_skip"] * simulation_timestep - elapsed))

            # 关窗时不更新不完整批次，仍保存上一次完成更新的模型。
            if window_closed:
                print("窗口已关闭，准备保存当前模型。", flush=True)
                break

            states, raw_actions, rewards, next_states, terminated, truncated, old_log_probs, values = zip(*transitions)
            states = torch.as_tensor(np.stack(states), dtype=torch.float32)
            raw_actions = torch.stack(raw_actions)
            rewards = torch.as_tensor(np.stack(rewards), dtype=torch.float32)
            next_states = torch.as_tensor(np.stack(next_states), dtype=torch.float32)
            terminated = torch.as_tensor(np.stack(terminated), dtype=torch.bool)
            truncated = torch.as_tensor(np.stack(truncated), dtype=torch.bool)
            old_log_probs = torch.stack(old_log_probs)
            values = torch.stack(values)
            with torch.no_grad():
                next_values = model.critic_net(next_states).squeeze(-1)
            # [T, N] 上逐环境递推：不把不同机械臂的轨迹连成一条。
            _, advantages, value_targets = compute_gae(
                rewards, values, next_values, terminated, truncated, gamma, gae_lambda
            )
            advantages = (advantages - advantages.mean()) / (advantages.std(unbiased=False) + 1e-8)
            states = states.reshape(-1, state_dimension)
            raw_actions = raw_actions.reshape(-1, action_dimension)
            old_log_probs = old_log_probs.reshape(-1)
            advantages = advantages.reshape(-1)
            value_targets = value_targets.reshape(-1)

            for _ in range(args.epochs):
                indices = torch.randperm(len(states))
                for start in range(0, len(states), args.minibatch_size):
                    indices_batch = indices[start:start + args.minibatch_size]
                    mb_raw = raw_actions[indices_batch]
                    _, new_raw_log_probs, new_values, entropy = model(states[indices_batch], mb_raw)
                    log_jacobian = 2 * (math.log(2) - mb_raw - F.softplus(-2 * mb_raw))
                    new_log_probs = new_raw_log_probs - log_jacobian.sum(-1)
                    loss = computer_ppo_loss(
                        new_log_probs, old_log_probs[indices_batch], advantages[indices_batch],
                        new_values, value_targets[indices_batch], entropy,
                        clip_epsilon, value_coef, entropy_coef,
                    )
                    optimizer.zero_grad()
                    loss.backward()
                    torch.nn.utils.clip_grad_norm_(model.parameters(), max_grad_norm)
                    optimizer.step()
                    with torch.no_grad():
                        model.log_std.clamp_(-5, 2)
                    total_updates += 1

            elapsed_total = time.monotonic() - training_started
            print(f"批次 {batch_index}/{args.batches} | 环境步数 {total_env_steps} | "
                  f"更新次数 {total_updates} | 已完成局数 {episodes} | "
                  f"平均采样及训练速度 {total_env_steps / elapsed_total:.1f} 环境步/秒", flush=True)
    except KeyboardInterrupt:
        print("\n训练已中断，准备保存当前模型。", flush=True)
    finally:
        try:
            if viewer is not None:
                viewer.close()
        finally:
            envs.close()

    args.checkpoint_dir.mkdir(parents=True, exist_ok=True)
    checkpoint_path = args.checkpoint_dir / f"ppo_piper_parallel_{datetime.now():%Y%m%d_%H%M%S_%f}.pt"
    torch.save({
        "model_state_dict": model.state_dict(),
        "network_config": {"state_dimension": state_dimension, "action_dimension": action_dimension,
                           "hidden_dimension": hidden_dimension, "activation": activation},
        "algorithm": "PPO with tanh-squashed Gaussian actions",
        "action_mapping": "tanh of Gaussian sample; deterministic evaluation uses tanh(mean)",
        "total_env_steps": total_env_steps,
        "total_updates": total_updates,
        "episode_rewards": episode_rewards,
        "episode_distances": episode_distances,
        "episode_successes": episode_successes,
        "environment_config": environment_config,
        "training_config": {"seed": seed, "num_envs": args.num_envs, "num_batches": args.batches,
                            "rollout_steps_per_env": args.rollout_steps, "update_epochs": args.epochs,
                            "minibatch_size": args.minibatch_size, "gamma": gamma, "gae_lambda": gae_lambda,
                            "learning_rate": learning_rate, "clip_epsilon": clip_epsilon,
                            "value_coef": value_coef, "entropy_coef": entropy_coef,
                            "max_grad_norm": max_grad_norm, "simulation_timestep": simulation_timestep},
    }, checkpoint_path)
    print(f"模型已保存：{checkpoint_path}", flush=True)
