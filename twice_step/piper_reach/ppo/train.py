# 本文件主要代码由 AI 助手完成。
# 连续动作 PPO 训练：复用已有的 Actor-Critic、GAE 和 PPO 损失。
from pathlib import Path
import argparse
import math
import sys
import time
from datetime import datetime

import numpy as np
import torch
import torch.nn.functional as F

# 支持直接运行本文件，同时复用其他练习里的 PPO 代码。
sys.path.insert(0, str(Path(__file__).resolve().parents[3]))
from twice_step.piper_reach.ppo.env import PiperReachEnv
from twice_step.ppo_pendulum.networks import Ppo_net
from twice_step.ppo_cartpole.advantages import compute_gae
from twice_step.ppo_cartpole.losses import computer_ppo_loss


if __name__ == "__main__":
    # 常用训练参数可以直接在这里调整。
    seed = 42
    num_batches = 200
    rollout_steps = 512
    update_epochs = 10
    minibatch_size = 64
    hidden_dimension = [512, 256, 128]
    activation = "elu"
    gamma = 0.99
    gae_lambda = 0.95
    learning_rate = 3e-4
    clip_epsilon = 0.2
    value_coef = 0.5
    entropy_coef = 0.0  # 现有网络返回高斯熵，不是 tanh 变换后的熵。
    max_grad_norm = 0.5
    show_window = True
    real_time = True

    parser = argparse.ArgumentParser(description="PPO training for Piper Reach")
    parser.add_argument("--no-render", action="store_true", help="不打开窗口，加快训练")
    parser.add_argument("--batches", type=int, default=num_batches)
    parser.add_argument("--rollout-steps", type=int, default=rollout_steps)
    parser.add_argument("--epochs", type=int, default=update_epochs)
    parser.add_argument("--minibatch-size", type=int, default=minibatch_size)
    parser.add_argument("--checkpoint-dir", type=Path, default=Path(__file__).resolve().parent)
    args = parser.parse_args()
    if min(args.batches, args.rollout_steps, args.epochs, args.minibatch_size) < 1:
        parser.error("batches、rollout-steps、epochs 和 minibatch-size 必须大于 0")

    show_window = show_window and not args.no_render
    torch.manual_seed(seed)
    np.random.seed(seed)
    torch.set_num_threads(1)
    env = PiperReachEnv(render_mode="human" if show_window else None)
    env.action_space.seed(seed)
    state_dimension = env.observation_space.shape[0]
    action_dimension = env.action_space.shape[0]
    model = Ppo_net(state_dimension, action_dimension, hidden_dimension, activation=activation)
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    state, _ = env.reset(seed=seed)

    total_env_steps = 0
    total_updates = 0
    episodes = 0
    episode_reward = 0.0
    episode_rewards = []
    episode_distances = []
    episode_successes = []
    window_closed = False

    try:
        for batch_index in range(1, args.batches + 1):
            transitions = []
            for _ in range(args.rollout_steps):
                if show_window and not env.viewer.is_running():
                    window_closed = True
                    break
                started_at = time.monotonic()
                state_tensor = torch.as_tensor(state, dtype=torch.float32)
                with torch.no_grad():
                    raw_action, raw_log_prob, value, _ = model(state_tensor)
                    action = raw_action.tanh()
                    # tanh 变换后的动作密度：原始高斯 log_prob 减去雅可比项。
                    log_jacobian = 2 * (math.log(2) - raw_action - F.softplus(-2 * raw_action))
                    old_log_prob = raw_log_prob - log_jacobian.sum(-1)

                next_state, reward, terminated, truncated, info = env.step(action.numpy())
                transitions.append((state, raw_action, reward, next_state, terminated,
                                    truncated, old_log_prob, value))
                total_env_steps += 1
                episode_reward += float(reward)
                if terminated or truncated:
                    episodes += 1
                    episode_rewards.append(episode_reward)
                    episode_distances.append(float(info["distance"]))
                    episode_successes.append(bool(info["is_success"]))
                    recent_successes = episode_successes[-10:]
                    recent_distances = episode_distances[-10:]
                    print(f"局数 {episodes} | 奖励 {episode_reward:.3f} | "
                          f"最终距离 {info['distance']:.4f} 米 | 成功 {info['is_success']} | "
                          f"最近{len(recent_successes)}局成功率 {np.mean(recent_successes):.1%} | "
                          f"平均最终距离 {np.mean(recent_distances):.4f} 米 | "
                          f"环境步数 {total_env_steps} | 更新次数 {total_updates}", flush=True)
                    episode_reward = 0.0
                    state, _ = env.reset()
                else:
                    state = next_state

                if show_window and real_time:
                    elapsed = time.monotonic() - started_at
                    time.sleep(max(0.0, env.frame_skip * env.model.opt.timestep - elapsed))

            # 用户关窗时不对不完整批次进行更新。
            if window_closed:
                break

            states, raw_actions, rewards, next_states, terminated, truncated, old_log_probs, values = zip(*transitions)
            states = torch.as_tensor(np.stack(states), dtype=torch.float32)
            raw_actions = torch.stack(raw_actions)
            rewards = torch.tensor(rewards, dtype=torch.float32)
            next_states = torch.as_tensor(np.stack(next_states), dtype=torch.float32)
            terminated = torch.tensor(terminated, dtype=torch.bool)
            truncated = torch.tensor(truncated, dtype=torch.bool)
            old_log_probs = torch.stack(old_log_probs)
            values = torch.stack(values)
            with torch.no_grad():
                next_values = model.critic_net(next_states).squeeze(-1)
            _, advantages, value_targets = compute_gae(
                rewards, values, next_values, terminated, truncated, gamma, gae_lambda
            )
            advantages = (advantages - advantages.mean()) / (advantages.std(unbiased=False) + 1e-8)

            for _ in range(args.epochs):
                indices = torch.randperm(len(transitions))
                for start in range(0, len(transitions), args.minibatch_size):
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

            print(f"批次 {batch_index}/{args.batches} | 环境步数 {total_env_steps} | "
                  f"更新次数 {total_updates} | 已完成局数 {episodes}", flush=True)
    except KeyboardInterrupt:
        print("\n训练已中断，准备保存当前模型。", flush=True)
    finally:
        env.close()

    args.checkpoint_dir.mkdir(parents=True, exist_ok=True)
    checkpoint_path = args.checkpoint_dir / f"ppo_piper_{datetime.now():%Y%m%d_%H%M%S_%f}.pt"
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
        "environment_config": {"frame_skip": env.frame_skip, "max_steps": env.max_steps,
                               "success_distance": env.success_distance, "joint_step": env.joint_step,
                               "min_goal_distance": env.min_goal_distance,
                               "linear_speed_cost": env.linear_speed_cost,
                               "angular_speed_cost": env.angular_speed_cost},
        "training_config": {"seed": seed, "num_batches": args.batches, "rollout_steps": args.rollout_steps,
                            "update_epochs": args.epochs, "minibatch_size": args.minibatch_size,
                            "gamma": gamma, "gae_lambda": gae_lambda, "learning_rate": learning_rate,
                            "clip_epsilon": clip_epsilon, "value_coef": value_coef,
                            "entropy_coef": entropy_coef, "max_grad_norm": max_grad_norm},
    }, checkpoint_path)
    print(f"模型已保存：{checkpoint_path}")
