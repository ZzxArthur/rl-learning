"""加载已保存的 DQN，用贪心动作评估 LunarLander，并显示环境运行画面。"""

from pathlib import Path
import random

import gymnasium as gym
import torch

from Q_network import Q_network  # Q_network(states_num, actions_num, hidden_width) -> Q network


@torch.inference_mode()
def evaluate(model_path=None, eval_seeds=None, render_mode="human"):
    if model_path is None:
        model_path = Path(__file__).with_name("dqn_checkpoint.pt")
    if eval_seeds is None:
        eval_seeds = random.sample(range(1_000_000), 5)
    if not eval_seeds:
        raise ValueError("至少需要一个评估种子")

    checkpoint = torch.load(model_path, map_location="cpu", weights_only=True)
    if any(not torch.isfinite(value).all() for value in checkpoint["model_state_dict"].values()):
        raise FloatingPointError("模型参数含 NaN 或无穷大，请先检查训练结果")
    model = Q_network(**checkpoint["network_config"])
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()

    episode_rewards = []
    # human 模式由 Gymnasium 自动显示画面，并按环境帧率播放。
    with gym.make("LunarLander-v3", render_mode=render_mode) as env:
        for episode, seed in enumerate(eval_seeds, start=1):
            state, _ = env.reset(seed=seed)
            total_reward = 0.0
            episode_steps = 0
            while True:
                state_tensor = torch.as_tensor(state, dtype=torch.float32)
                q_values = model(state_tensor)
                action = int(q_values.argmax().item())
                state, reward, terminated, truncated, _ = env.step(action)
                total_reward += reward
                episode_steps += 1
                if terminated or truncated:
                    break
            episode_rewards.append(total_reward)
            print(f"评估局数{episode}, 种子{seed}, 总奖励{total_reward:.2f}, 步数{episode_steps}")

    mean_reward = sum(episode_rewards) / len(episode_rewards)
    print(f"{len(episode_rewards)} 局平均奖励: {mean_reward:.2f}")
    return episode_rewards


if __name__ == "__main__":
    evaluate()

# 本文件主要代码由 AI 助手完成。
