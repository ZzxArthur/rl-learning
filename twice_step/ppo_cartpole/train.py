# PPO训练入口：持续采样新批次，计算GAE并进行多轮小批更新。
# 统计完整回合奖励、环境步数和更新次数，结束后保存权重和奖励曲线。
from pathlib import Path
import torch
import numpy as np
import gymnasium as gym
import matplotlib.pyplot as plt

from networks import Ppo_net  # Ppo_net(state_dimension, action_dimension, hidden_dimension) -> model
from rollout import collect_rollout  # collect_rollout(env, state, model, num_step) -> transitions, state
from rollout import random_split  # random_split(states, actions, old_log_probs, advantages, values_target, mb_indices) -> minibatch tensors
from advantages import compute_gae  # compute_gae(rewards, values, next_values, terminated, truncated, gamma, gae_lambda) -> td_error, advantages, values_target
from losses import computer_ppo_loss  # computer_ppo_loss(new_log_probs, old_log_probs, advantages, new_values, values_target, entropy, clip_epsilon, value_coef, entropy_coef) -> total_loss


if __name__ == "__main__":
    seed = 42
    env_id = "CartPole-v1"
    num_batches = 1000
    num_steps = 256
    update_epochs = 16
    minibatch_size = 64

    gamma = 0.99
    gae_lambda = 0.7
    lr = 0.01
    clip_epsilon = 0.2
    value_coef = 0.5
    entropy_coef = 0.01

    network_config = {
        "state_dimension": 4,
        "action_dimension": 2,
        "hidden_dimension": [64, 64],
    }

    torch.manual_seed(seed)
    env = gym.make(env_id)
    ppo_model = Ppo_net(network_config["state_dimension"], network_config["action_dimension"], network_config["hidden_dimension"])
    optimizer = torch.optim.SGD(ppo_model.parameters(), lr=lr)
    state, _ = env.reset(seed=seed)
    total_env_steps = 0
    total_updates = 0
    num_episodes = 0
    episode_reward = 0.0
    episode_rewards = []

    for batch_index in range(1, num_batches + 1):
        transitions, state = collect_rollout(env, state, ppo_model, num_steps)
        for _, _, reward, _, terminated, truncated, _, _ in transitions:
            total_env_steps += 1
            episode_reward += reward
            if terminated or truncated:
                num_episodes += 1
                episode_rewards.append(episode_reward)
                print(f"局数{num_episodes}, 总奖励:{episode_reward:.2f}, "
                      f"累计环境步数:{total_env_steps}")
                episode_reward = 0.0
        # 以上回合奖励与环境步数统计代码主要由 AI 助手完成。

        states, actions, rewards, next_states, \
                    terminated, truncated, old_log_probs, values = zip(*transitions)

        states = torch.as_tensor(np.stack(states), dtype=torch.float32)
        actions = torch.tensor(actions, dtype=torch.long)
        rewards = torch.tensor(rewards, dtype=torch.float32)
        next_states = torch.as_tensor(np.stack(next_states), dtype=torch.float32)
        terminated = torch.tensor(terminated, dtype=torch.bool)
        truncated = torch.tensor(truncated, dtype=torch.bool)
        old_log_probs = torch.stack(old_log_probs)
        values = torch.stack(values)
        with torch.no_grad():
            next_values = ppo_model.critic_net(next_states).squeeze(-1)
        td_error, advantages, values_target = compute_gae(rewards, values, next_values, terminated, truncated, gamma, gae_lambda)

        for _ in range(update_epochs):
            indices = torch.randperm(num_steps, device=states.device)
            for start in range(0, num_steps, minibatch_size):
                mb_indices = indices[start: start + minibatch_size]
                mb_states, mb_actions, mb_old_log_probs, \
                    mb_advantages, mb_values_target = random_split(states, actions, old_log_probs,
                                                                   advantages, values_target, mb_indices)
                _, mb_new_log_probs, mb_new_values, mb_entropy = ppo_model(
                    mb_states, mb_actions
                )
                ppo_loss = computer_ppo_loss(mb_new_log_probs, mb_old_log_probs, mb_advantages,
                                             mb_new_values, mb_values_target, mb_entropy,
                                             clip_epsilon, value_coef, entropy_coef)
                ppo_loss.backward()
                with torch.no_grad():
                    optimizer.step()
                    total_updates += 1
                    optimizer.zero_grad()

        print(f"批次{batch_index}/{num_batches}, 累计环境步数:{total_env_steps}, "
              f"累计更新次数:{total_updates}, 已完成局数:{num_episodes}")

    checkpoint_path = Path(__file__).with_name("ppo_checkpoint.pt")
    torch.save({
        "model_state_dict": ppo_model.state_dict(),
        "network_config": network_config,
        "env_id": env_id,
        "total_env_steps": total_env_steps,
        "total_updates": total_updates,
        "episode_rewards": episode_rewards,
    }, checkpoint_path)
    print(f"模型已保存至: {checkpoint_path}")
    # 以上权重保存代码主要由 AI 助手完成。
    env.close()

    reward_window = 10
    group_end_episodes = []
    mean_rewards = []
    for start in range(0, len(episode_rewards), reward_window):
        group = episode_rewards[start:start + reward_window]
        group_end_episodes.append(start + len(group))
        mean_rewards.append(sum(group) / len(group))

    curve_path = Path(__file__).with_name("training_rewards.png")
    fig, ax = plt.subplots(figsize=(8, 4.5), layout="constrained")
    ax.plot(group_end_episodes, mean_rewards,
            color="#0072B2", linewidth=1, marker="o", markersize=3)
    ax.set_xlabel("Training episode (group end)")
    ax.set_ylabel("Mean episode reward")
    ax.set_title(f"PPO training rewards: {env_id}\n"
                 "Mean per 10 episodes (last group may be shorter)")
    ax.grid(alpha=0.25)
    fig.savefig(curve_path, dpi=150)
    plt.close(fig)
    print(f"训练奖励折线图已保存至: {curve_path}")
    # 以上奖励绘图代码主要由 AI 助手完成。
