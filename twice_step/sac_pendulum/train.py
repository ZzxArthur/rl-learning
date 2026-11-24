# 连续 SAC 训练入口：采集经验、更新网络、保存权重和奖励曲线。
from pathlib import Path
import math
import random

import gymnasium as gym
import torch

from updates import computer_td_target  # computer_td_target(actor, target_q1, target_q2, next_states, rewards, terminated, gamma, alpha) -> y
from updates import update_critics  # update_critics(q1, q2, optimizer, states, actions, td_target) -> critic_loss
from updates import update_actor  # update_actor(actor, q1, q2, optimizer, states, alpha) -> actor_loss
from updates import updata_alpha  # updata_alpha(alpha_optimizer, log_alpha, log_probs, target_entropy) -> alpha
from updates import soft_update  # soft_update(q1, q2, target_q1, target_q2, tau) -> None
from networks import Actornet, Criticnet  # Actornet(**actor_config) -> actor; Criticnet(**critic_config) -> Q-network
from replay_batch import ReplayBuffer  # ReplayBuffer(capacity) -> buffer
from checkpoint import save_checkpoint  # save_checkpoint(actor, path, *, env_id, env_kwargs, network_config, training_config, episode_rewards, total_updates) -> saved path


# evaluate.py 读取这些设置。
CHECKPOINT_PATH = Path(__file__).with_name("sac_checkpoint.pt")
EVAL_CONFIG = {
    "num_episodes": 100,
    "num_visualize": 10,
    "render_mode": "human",
}


if __name__ == "__main__":
    # 环境与训练参数。
    seed = 42
    env_id = "Pendulum-v1"
    env_kwargs = {}  # 例如 {"max_episode_steps": 200}
    device = "cpu"  # 可以改为 "cuda"
    total_steps = 40000
    warmup_steps = 1000
    batch_size = 256
    capacity = 40000
    train_frequency = 1
    gradient_steps = 1

    gamma = 0.99
    initial_alpha = 0.2
    tau = 0.005
    actor_lr = 3e-4
    critic_lr = 3e-4
    alpha_lr = 3e-4
    target_entropy = -1.0
    actor_adam_kwargs = {}  # 例如 {"eps": 1e-5}
    critic_adam_kwargs = {}

    actor_hidden_dimensions = (256,)
    critic_hidden_dimensions = (256,)
    actor_activation = "relu"
    critic_activation = "relu"
    log_std_min = -5.0
    log_std_max = 2.0
    log_prob_epsilon = 1e-6
    log_every = 1
    reward_window = 10

    if min(total_steps, batch_size, train_frequency, gradient_steps, reward_window) < 1:
        raise ValueError("训练步数、批量大小、更新频率、每轮更新批数和奖励窗口必须大于 0")
    if capacity < batch_size or warmup_steps < 0 or log_every < 0:
        raise ValueError("经验池容量不能小于批量大小，预热步数和日志间隔不能小于 0")

    torch.manual_seed(seed)
    random.seed(seed)
    env = gym.make(env_id, **env_kwargs)
    state_space = env.observation_space
    action_space = env.action_space
    if (not isinstance(state_space, gym.spaces.Box) or len(state_space.shape) != 1
            or not isinstance(action_space, gym.spaces.Box) or len(action_space.shape) != 1):
        env.close()
        raise ValueError("当前训练入口需要向量形式的 Box 状态和动作空间")

    state_dimension = state_space.shape[0]
    action_dimension = action_space.shape[0]
    network_config = {
        "actor": {
            "state_dimension": state_dimension,
            "action_dimension": action_dimension,
            "hidden_dimensions": actor_hidden_dimensions,
            "activation": actor_activation,
            "log_std_min": log_std_min,
            "log_std_max": log_std_max,
            "log_prob_epsilon": log_prob_epsilon,
            "action_low": action_space.low.tolist(),
            "action_high": action_space.high.tolist(),
        },
        "critic": {
            "state_dimension": state_dimension,
            "action_dimension": action_dimension,
            "hidden_dimensions": critic_hidden_dimensions,
            "activation": critic_activation,
        },
    }

    actor = Actornet(**network_config["actor"]).to(device)
    q1 = Criticnet(**network_config["critic"]).to(device)
    q2 = Criticnet(**network_config["critic"]).to(device)
    target_q1 = Criticnet(**network_config["critic"]).to(device)
    target_q2 = Criticnet(**network_config["critic"]).to(device)
    target_q1.load_state_dict(q1.state_dict())
    target_q2.load_state_dict(q2.state_dict())

    actor_optimizer = torch.optim.Adam(actor.parameters(), lr=actor_lr, **actor_adam_kwargs)
    critic_optimizer = torch.optim.Adam(
        list(q1.parameters()) + list(q2.parameters()), lr=critic_lr, **critic_adam_kwargs
    )
    log_alpha = torch.tensor(math.log(initial_alpha), dtype=torch.float32,
                             device=device, requires_grad=True)
    alpha_optimizer = torch.optim.Adam([log_alpha], lr=alpha_lr)
    alpha = initial_alpha
    replaybuffer = ReplayBuffer(capacity)

    env.action_space.seed(seed)
    state, _ = env.reset(seed=seed)
    update_count = 0
    episode_rewards = []
    episode_reward = 0.0

    for step in range(1, total_steps + 1):
        # 预热时随机探索，之后由 Actor 采样动作。
        if step <= warmup_steps:
            action = env.action_space.sample()
        else:
            state_tensor = torch.as_tensor(state, dtype=torch.float32, device=device).unsqueeze(0)
            with torch.no_grad():
                action_tensor, _ = actor.sample_action(state_tensor)
            action = action_tensor.squeeze(0).cpu().numpy()

        next_state, reward, terminated, truncated, _ = env.step(action)
        replaybuffer.push(state, action, reward, next_state, terminated)
        episode_reward += float(reward)

        # 从经验池采样，依次更新 Critic、Actor、目标网络和 alpha。
        if (step >= warmup_steps and step % train_frequency == 0
                and len(replaybuffer.buffer) >= batch_size):
            for _ in range(gradient_steps):
                batch = replaybuffer.sample(batch_size)
                states, actions, rewards, next_states, terminated_batch = (
                    value.to(device) for value in batch
                )
                td_target = computer_td_target(
                    actor, target_q1, target_q2, next_states,
                    rewards, terminated_batch, gamma, alpha
                )
                critic_loss = update_critics(
                    q1, q2, critic_optimizer, states, actions, td_target
                )
                actor_loss = update_actor(actor, q1, q2, actor_optimizer, states, alpha)
                soft_update(q1, q2, target_q1, target_q2, tau)

                with torch.no_grad():
                    _, log_probs = actor.sample_action(states)
                alpha = updata_alpha(alpha_optimizer, log_alpha, log_probs, target_entropy)
                update_count += 1

        if terminated or truncated:
            episode_rewards.append(episode_reward)
            episode = len(episode_rewards)
            if log_every and episode % log_every == 0:
                recent_rewards = episode_rewards[-reward_window:]
                recent_mean = sum(recent_rewards) / len(recent_rewards)
                delta = f"{episode_reward - episode_rewards[-2]:+.2f}" if episode > 1 else "—"
                print(f"第 {episode} 局 | 总奖励 {episode_reward:.2f} | 较上局 {delta} | "
                      f"alpha {float(alpha):.4f} | 最近 {len(recent_rewards)} 局均值 {recent_mean:.2f} | "
                      f"环境步 {step}/{total_steps} | 累计更新 {update_count}", flush=True)
            episode_reward = 0.0
            state, _ = env.reset()
        else:
            state = next_state

    env.close()

    # 保存训练配置和 Actor 权重，供评估脚本读取。
    training_config = {
        "seed": seed,
        "device": device,
        "total_steps": total_steps,
        "warmup_steps": warmup_steps,
        "batch_size": batch_size,
        "capacity": capacity,
        "gamma": gamma,
        "alpha": initial_alpha,
        "tau": tau,
        "actor_lr": actor_lr,
        "critic_lr": critic_lr,
        "alpha_lr": alpha_lr,
        "target_entropy": target_entropy,
        "actor_adam_kwargs": actor_adam_kwargs,
        "critic_adam_kwargs": critic_adam_kwargs,
        "train_frequency": train_frequency,
        "gradient_steps": gradient_steps,
        "log_every": log_every,
        "reward_window": reward_window,
    }
    saved_path = save_checkpoint(
        actor, CHECKPOINT_PATH, env_id=env_id, env_kwargs=env_kwargs,
        network_config=network_config, training_config=training_config,
        episode_rewards=episode_rewards, total_updates=update_count,
    )
    print(f"完成局数 {len(episode_rewards)}，参数更新 {update_count} 次")
    print(f"Actor 权重与配置保存到 {saved_path}")

    if episode_rewards:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        group_ends = []
        group_means = []
        for start in range(0, len(episode_rewards), reward_window):
            group = episode_rewards[start:start + reward_window]
            group_ends.append(start + len(group))
            group_means.append(sum(group) / len(group))

        curve_path = Path(__file__).with_name("training_rewards.png")
        figure, axes = plt.subplots(figsize=(8, 4.5), layout="constrained")
        axes.plot(group_ends, group_means, marker="o", markersize=3, color="#0072B2")
        axes.set(xlabel="Training episode (group end)", ylabel="Mean episode reward",
                 title=f"SAC: {env_id} | Mean per {reward_window} episodes (last group may be shorter)")
        axes.grid(alpha=0.25)
        figure.savefig(curve_path, dpi=150)
        plt.close(figure)
        print(f"奖励曲线保存到 {curve_path}")

# 本文件主要代码由 AI 助手完成。
