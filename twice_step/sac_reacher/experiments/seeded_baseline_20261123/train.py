from replaybuffer import ReplayBuffer
from updata import actor_update, critic_update, alpha_update, soft_update, compute_td_target
from critic import Critic_net
from actor import Actor_net

import gymnasium as gym
import numpy as np
import torch
import math
from pathlib import Path


seed = 42
torch.manual_seed(seed)
np.random.seed(seed)

env = gym.make("Reacher-v5")
env.action_space.seed(seed)

state_space = env.observation_space
action_space = env.action_space
state_dimension = state_space.shape[0]
action_dimension = action_space.shape[0]
actor_lr = 3e-4
critic_lr = 3e-4
alpha_lr = 3e-4
gamma = 0.99
num_steps = 40000
batch_size = 256
warmup_steps = 1000
capacity = 20000
alpha = 0.2
target_entropy = -0.1
tau = 0.005
log_alpha = torch.tensor(
        math.log(alpha),
        dtype=torch.float32,
        requires_grad=True,
    )


actor = Actor_net(state_dimension, action_dimension)
critic1, critic2 = Critic_net(state_dimension, action_dimension), Critic_net(state_dimension, action_dimension)
target_critic1, target_critic2 = Critic_net(state_dimension, action_dimension), Critic_net(state_dimension, action_dimension)
target_critic1.load_state_dict(critic1.state_dict())
target_critic2.load_state_dict(critic2.state_dict())
replaybuffer = ReplayBuffer(capacity)

actor_optimizer = torch.optim.Adam(actor.parameters(), lr=actor_lr)
critic_optimizer = torch.optim.Adam(list(critic1.parameters()) + list(critic2.parameters()), lr=critic_lr)
alpha_optimizer = torch.optim.Adam([log_alpha], lr=alpha_lr)

state, _ = env.reset(seed=seed)
episode_reward = 0.0
episode_rewards = []
update_count = 0
critic_loss = None
actor_loss = None
# 以上训练统计初始化由 AI 助手完成。
for step in range(1, 1 + num_steps):
    if step <= warmup_steps:
        action = env.action_space.sample()
    else:
        state_tensor = torch.as_tensor(state, dtype=torch.float32).unsqueeze(0)
        with torch.no_grad():
            action, log_prob = actor.sample(state_tensor)
            action = action.squeeze(0).cpu().numpy()
    next_state, reward, terminated, truncated, info = env.step(action)
    replaybuffer.push(state, action, reward, next_state, terminated)
    episode_reward += float(reward)

    if step > (warmup_steps) and len(replaybuffer.buffer) > batch_size:
        states, actions, rewards, next_states, done = replaybuffer.get_transitions(batch_size)
        td_target = compute_td_target(actor, target_critic1, target_critic2, rewards,
                                      next_states, done, gamma, alpha)

        critic_loss = critic_update(critic1, critic2, critic_optimizer, states, actions, td_target)
        actor_loss = actor_update(actor, critic1, critic2, actor_optimizer, states, alpha)
        with torch.no_grad():
            soft_update(critic1, critic2, target_critic1, target_critic2, tau)
        alpha = alpha_update(actor, alpha_optimizer, target_entropy, states, log_alpha)
        update_count += 1
    if terminated or truncated:
        episode_rewards.append(episode_reward)
        recent_rewards = episode_rewards[-10:]
        recent_mean = sum(recent_rewards) / len(recent_rewards)
        loss_info = "预热中" if critic_loss is None else (
            f"Critic loss {critic_loss:.4f} | Actor loss {actor_loss:.4f}"
        )
        print(f"第 {len(episode_rewards)} 局 | 奖励 {episode_reward:.2f} | "
              f"最近 {len(recent_rewards)} 局均值 {recent_mean:.2f} | "
              f"末步距离 {-info['reward_dist']:.4f} | alpha {float(alpha):.4f} | "
              f"环境步 {step}/{num_steps} | 更新 {update_count} | {loss_info}", flush=True)
        episode_reward = 0.0
        state, _ = env.reset()
    else:
        state = next_state
env.close()

checkpoint_path = Path(__file__).with_name("sac_checkpoint.pt")
torch.save({
    "model_state_dict": actor.state_dict(),
    "env_id": "Reacher-v5",
    "state_dimension": state_dimension,
    "action_dimension": action_dimension,
    "episode_rewards": episode_rewards,
    "total_env_steps": num_steps,
    "total_updates": update_count,
    "training_config": {
        "seed": seed, "actor_lr": actor_lr, "critic_lr": critic_lr,
        "alpha_lr": alpha_lr, "gamma": gamma, "num_steps": num_steps,
        "batch_size": batch_size, "warmup_steps": warmup_steps,
        "capacity": capacity, "target_entropy": target_entropy, "tau": tau,
    },
    "final_alpha": float(alpha),
}, checkpoint_path)
print(f"Actor 权重保存到 {checkpoint_path}")

if episode_rewards:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    group_ends = []
    mean_rewards = []
    for start in range(0, len(episode_rewards), 10):
        group = episode_rewards[start:start + 10]
        group_ends.append(start + len(group))
        mean_rewards.append(sum(group) / len(group))
    figure, axes = plt.subplots(figsize=(8, 4.5), layout="constrained")
    axes.plot(group_ends, mean_rewards, marker="o", markersize=3)
    axes.set(xlabel="Episode (group end)", ylabel="Mean reward",
             title="Reacher SAC: mean reward per 10 episodes")
    axes.grid(alpha=0.25)
    curve_path = Path(__file__).with_name("training_rewards.png")
    figure.savefig(curve_path, dpi=150)
    plt.close(figure)
    print(f"奖励曲线保存到 {curve_path}")
# 以上训练记录、权重保存和绘图代码主要由 AI 助手完成。
