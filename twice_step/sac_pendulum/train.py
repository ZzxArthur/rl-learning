# SAC 训练入口：集中配置，采集经验并更新网络。
from pathlib import Path
import random
import math
import gymnasium as gym
import torch

from updates import updata_alpha
from updates import computer_td_target  # computer_td_target(actor, target_q1, target_q2, next_states, rewards, terminated, gamma, alpha) -> y
from updates import update_critics  # update_critics(q1, q2, optimizer, states, actions, td_target) -> critic_loss
from updates import update_actor  # update_actor(actor, q1, q2, optimizer, states, alpha) -> actor_loss
from updates import soft_update  # soft_update(q1, q2, target_q1, target_q2, tau) -> None
from networks import Actornet, Criticnet  # Actornet(**actor_config) -> actor; Criticnet(**critic_config) -> Q-network
from replay_batch import ReplayBuffer  # ReplayBuffer(capacity) -> buffer
from checkpoint import save_checkpoint  # save_checkpoint(actor, path, *, env_id, env_kwargs, network_config, training_config, episode_rewards, total_updates) -> saved path


ENV_ID = "Pendulum-v1"
ENV_KWARGS = {}  # 例如 {"max_episode_steps": 200}

ACTOR_CONFIG = {
    "hidden_dimensions": (256,),
    "activation": "relu",
    "log_std_min": -5.0,
    "log_std_max": 2.0,
    "log_prob_epsilon": 1e-6,
}
CRITIC_CONFIG = {
    "hidden_dimensions": (256,),
    "activation": "relu",
}

TRAIN_CONFIG = {
    "seed": 42,
    "device": "cpu",  # 可设为 "cuda"
    "total_steps": 40000,
    "warmup_steps": 1000,
    "batch_size": 256,
    "capacity": 40000,
    "gamma": 0.99,
    "alpha": 0.2,
    "tau": 0.005,
    "actor_lr": 3e-4,
    "critic_lr": 3e-4,
    "alpha_lr": 3e-4,
    "target_entropy": -1.0,
    "actor_adam_kwargs": {},  # 例如 {"eps": 1e-5}
    "critic_adam_kwargs": {},
    "train_frequency": 1,  # 每隔多少环境步训练
    "gradient_steps": 1,  # 每次训练抽几批经验
    "log_every": 1,  # 每隔多少局打印，0 表示静默
    "reward_window": 10,
}

CHECKPOINT_PATH = Path(__file__).with_name("sac_checkpoint.pt")
REWARD_CURVE_PATH = Path(__file__).with_name("training_rewards.png")
EVAL_CONFIG = {
    "num_episodes": 100,
    "num_visualize": 10,
    "render_mode": "human",  # None 表示只评分
}


if __name__ == "__main__":
    seed = TRAIN_CONFIG["seed"]
    device = torch.device(TRAIN_CONFIG["device"])
    total_steps = TRAIN_CONFIG["total_steps"]
    warmup_steps = TRAIN_CONFIG["warmup_steps"]
    batch_size = TRAIN_CONFIG["batch_size"]
    capacity = TRAIN_CONFIG["capacity"]
    gamma = TRAIN_CONFIG["gamma"]
    alpha = TRAIN_CONFIG["alpha"]
    tau = TRAIN_CONFIG["tau"]
    train_frequency = TRAIN_CONFIG["train_frequency"]
    gradient_steps = TRAIN_CONFIG["gradient_steps"]
    log_every = TRAIN_CONFIG["log_every"]
    reward_window = TRAIN_CONFIG["reward_window"]
    log_alpha = torch.tensor(
        math.log(alpha),
        dtype=torch.float32,
        device=device,
        requires_grad=True,
    )
    alpha_optimizer = torch.optim.Adam(
        [log_alpha],
        lr=TRAIN_CONFIG["alpha_lr"],
    )
    target_entropy = TRAIN_CONFIG["target_entropy"]
    if min(total_steps, batch_size, train_frequency, gradient_steps, reward_window) < 1:
        raise ValueError("训练步数、批量大小、更新频率、每轮更新批数和奖励窗口必须大于 0")
    if capacity < batch_size or warmup_steps < 0 or log_every < 0:
        raise ValueError("经验池容量不能小于批量大小，预热步数和日志间隔不能小于 0")

    torch.manual_seed(seed)
    random.seed(seed)
    with gym.make(ENV_ID, **ENV_KWARGS) as env:
        state_space, action_space = env.observation_space, env.action_space
        if (not isinstance(state_space, gym.spaces.Box) or len(state_space.shape) != 1
                or not isinstance(action_space, gym.spaces.Box) or len(action_space.shape) != 1):
            raise ValueError("当前训练入口需要向量形式的 Box 状态和动作空间")
        dimensions = dict(state_dimension=state_space.shape[0], action_dimension=action_space.shape[0])
        network_config = {
            "actor": dict(ACTOR_CONFIG, **dimensions,
                          action_low=action_space.low.tolist(), action_high=action_space.high.tolist()),
            "critic": dict(CRITIC_CONFIG, **dimensions),
        }
        env.action_space.seed(seed)
        state, _ = env.reset(seed=seed)

        actor = Actornet(**network_config["actor"]).to(device)
        q1, q2 = (Criticnet(**network_config["critic"]).to(device) for _ in range(2))
        target_q1, target_q2 = (Criticnet(**network_config["critic"]).to(device) for _ in range(2))
        target_q1.load_state_dict(q1.state_dict())
        target_q2.load_state_dict(q2.state_dict())
        actor_optimizer = torch.optim.Adam(actor.parameters(), lr=TRAIN_CONFIG["actor_lr"],
                                            **TRAIN_CONFIG["actor_adam_kwargs"])
        critic_optimizer = torch.optim.Adam(list(q1.parameters()) + list(q2.parameters()),
                                             lr=TRAIN_CONFIG["critic_lr"],
                                             **TRAIN_CONFIG["critic_adam_kwargs"])
        replaybuffer = ReplayBuffer(capacity)

        update_count = 0
        episode_rewards = []
        episode_reward = 0.0
        for step in range(1, total_steps + 1):
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

            if (step >= warmup_steps and step % train_frequency == 0
                    and len(replaybuffer.buffer) >= batch_size):
                for _ in range(gradient_steps):
                    states, actions, rewards, next_states, terminated_batch = (
                        value.to(device) for value in replaybuffer.sample(batch_size)
                    )
                    td_target = computer_td_target(actor, target_q1, target_q2, next_states,
                                                   rewards, terminated_batch, gamma, alpha)
                    critic_loss = update_critics(q1, q2, critic_optimizer, states, actions, td_target)
                    actor_loss = update_actor(actor, q1, q2, actor_optimizer, states, alpha)
                    soft_update(q1, q2, target_q1, target_q2, tau)
                    with torch.no_grad():
                        _, log_probs = actor.sample_action(states)
                    alpha = updata_alpha(
                        alpha_optimizer,
                        log_alpha,
                        log_probs,
                        target_entropy,
                    )
                    update_count += 1

            if terminated or truncated:
                episode_rewards.append(episode_reward)
                episode = len(episode_rewards)
                if log_every and episode % log_every == 0:
                    recent_rewards = episode_rewards[-reward_window:]
                    recent_mean = sum(recent_rewards) / len(recent_rewards)
                    delta = f"{episode_reward - episode_rewards[-2]:+.2f}" if episode > 1 else "—"
                    print(f"第 {episode} 局 | 总奖励 {episode_reward:.2f} | 较上局 {delta} | alpah{float(alpha):.4f}"
                          f"最近 {len(recent_rewards)} 局均值 {recent_mean:.2f} | "
                          f"环境步 {step}/{total_steps} | 累计更新 {update_count}", flush=True)
                episode_reward = 0.0
                state, _ = env.reset()
            else:
                state = next_state

    saved_path = save_checkpoint(
        actor, CHECKPOINT_PATH, env_id=ENV_ID, env_kwargs=ENV_KWARGS,
        network_config=network_config, training_config=TRAIN_CONFIG,
        episode_rewards=episode_rewards, total_updates=update_count,
    )
    print(f"完成局数 {len(episode_rewards)}，参数更新 {update_count} 次")
    print(f"Actor 权重与配置保存到 {saved_path}")

    if episode_rewards:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        group_ends, group_means = [], []
        for start in range(0, len(episode_rewards), reward_window):
            group = episode_rewards[start:start + reward_window]
            group_ends.append(start + len(group))
            group_means.append(sum(group) / len(group))
        figure, axes = plt.subplots(figsize=(8, 4.5), layout="constrained")
        axes.plot(group_ends, group_means, marker="o", markersize=3, color="#0072B2")
        axes.set(xlabel="Training episode (group end)", ylabel="Mean episode reward",
                 title=f"SAC: {ENV_ID} | Mean per {reward_window} episodes (last group may be shorter)")
        axes.grid(alpha=0.25)
        figure.savefig(REWARD_CURVE_PATH, dpi=150)
        plt.close(figure)
        print(f"奖励曲线保存到 {REWARD_CURVE_PATH}")

# 本文件主要代码由 AI 助手完成。
