"""快速评估连续PPO一百局，再随机重放其中十局的Pendulum画面。"""

from pathlib import Path
import random
import sys
from time import perf_counter

import gymnasium as gym
import numpy as np
import torch

# 直接运行文件时补入仓库根目录，同时支持包导入。
if not __package__:
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from twice_step.ppo_pendulum.networks import Ppo_net  # Ppo_net(state_dimension, action_dimension, hidden_dimension) -> model


@torch.inference_mode()
def evaluate(model_path=None, eval_seeds=None, render_mode="human",
             num_episodes=100, num_visualize=10):
    start_time = perf_counter()
    if num_episodes < 1 or num_visualize < 0:
        raise ValueError("评估局数必须大于0，可视化局数不能小于0")
    if model_path is None:
        model_path = Path(__file__).with_name("ppo_checkpoint.pt")
    if eval_seeds is None:
        eval_seeds = random.sample(range(1_000_000), num_episodes)
    if not eval_seeds:
        raise ValueError("至少需要一个评估种子")
    visual_indices = set(random.sample(
        range(1, len(eval_seeds) + 1), min(num_visualize, len(eval_seeds))
    )) if render_mode is not None else set()

    checkpoint = torch.load(model_path, map_location="cpu", weights_only=True)
    if any(not torch.isfinite(value).all() for value in checkpoint["model_state_dict"].values()):
        raise FloatingPointError("模型参数含NaN或无穷大，请先检查训练结果")
    model = Ppo_net(**checkpoint["network_config"])
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()

    episode_rewards = []
    replays = []
    # 评分阶段不渲染；连续策略用均值动作，裁剪后执行并保存重放。
    with gym.make(checkpoint["env_id"]) as env:
        for episode, seed in enumerate(eval_seeds, start=1):
            state, _ = env.reset(seed=seed)
            total_reward = 0.0
            episode_steps = 0
            episode_actions = []
            while True:
                state_tensor = torch.as_tensor(state, dtype=torch.float32)
                action_mean = model.actor_net(state_tensor)
                action = np.clip(action_mean.numpy(), env.action_space.low, env.action_space.high)
                if episode in visual_indices:
                    episode_actions.append(action.copy())
                state, reward, terminated, truncated, _ = env.step(action)
                total_reward += reward
                episode_steps += 1
                if terminated or truncated:
                    break
            episode_rewards.append(total_reward)
            if episode in visual_indices:
                replays.append((episode, seed, episode_actions))
            print(f"评估局数{episode}, 种子{seed}, 总奖励:{total_reward:.2f}, 步数:{episode_steps}")

    mean_reward = sum(episode_rewards) / len(episode_rewards)
    print(f"{len(episode_rewards)}局平均奖励: {mean_reward:.2f}")
    print(f"最低奖励:{min(episode_rewards):.2f}, 最高奖励:{max(episode_rewards):.2f}, "
          f"无窗口评估耗时:{perf_counter() - start_time:.2f}秒")

    if replays:
        print(f"随机展示其中{len(replays)}局，重放不重复计入上述统计。")
        # human模式按环境帧率播放，使用评分时的原种子与原执行动作。
        with gym.make(checkpoint["env_id"], render_mode=render_mode) as env:
            for episode, seed, episode_actions in replays:
                print(f"可视化原第{episode}局, 种子{seed}, "
                      f"总奖励:{episode_rewards[episode - 1]:.2f}")
                env.reset(seed=seed)
                for action in episode_actions:
                    _, _, terminated, truncated, _ = env.step(action)
                    if terminated or truncated:
                        break
    return episode_rewards


if __name__ == "__main__":
    evaluate()

# 本文件主要代码由 AI 助手完成。
