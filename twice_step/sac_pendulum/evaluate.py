"""用确定性动作评估 Pendulum SAC Actor，并随机回放部分评估局。"""

import argparse
from pathlib import Path
import random
from statistics import mean

import gymnasium as gym
import torch

from checkpoint import ACTOR_PATH, load_actor  # load_actor(path) -> actor, env_id, env_kwargs
from train import CHECKPOINT_PATH, EVAL_CONFIG


@torch.inference_mode()
def evaluate(model_path=None, eval_seeds=None, render_mode=EVAL_CONFIG["render_mode"],
             num_episodes=EVAL_CONFIG["num_episodes"], num_visualize=EVAL_CONFIG["num_visualize"]):
    if num_episodes < 1 or num_visualize < 0:
        raise ValueError("评估局数必须大于 0，可视化局数不能小于 0")
    seeds = (random.sample(range(1_000_000), num_episodes)
             if eval_seeds is None else list(eval_seeds))
    if not seeds:
        raise ValueError("至少需要一个评估种子")
    replay_episodes = set()
    if render_mode is not None:
        replay_episodes = set(random.sample(range(len(seeds)), min(num_visualize, len(seeds))))

    if model_path is None:
        model_path = CHECKPOINT_PATH if CHECKPOINT_PATH.is_file() else ACTOR_PATH
    actor, env_id, env_kwargs = load_actor(model_path)
    print(f"评估权重 {model_path}")

    rewards = []
    replays = []
    with gym.make(env_id, **dict(env_kwargs, render_mode=None)) as env:
        for episode, seed in enumerate(seeds):
            state, _ = env.reset(seed=seed)
            total_reward = 0.0
            actions = []
            while True:
                state_tensor = torch.as_tensor(state, dtype=torch.float32)
                action = actor.deterministic_action(state_tensor).numpy()
                if episode in replay_episodes:
                    actions.append(action.copy())
                state, reward, terminated, truncated, _ = env.step(action)
                total_reward += float(reward)
                if terminated or truncated:
                    break
            rewards.append(total_reward)
            if episode in replay_episodes:
                replays.append((episode, seed, actions))
            print(f"第 {episode + 1} 局 | 种子 {seed} | 总奖励 {total_reward:.2f}")

    print(f"{len(rewards)} 局平均奖励 {mean(rewards):.2f}，"
          f"最低 {min(rewards):.2f}，最高 {max(rewards):.2f}")

    if replays:
        with gym.make(env_id, **dict(env_kwargs, render_mode=render_mode)) as env:
            for episode, seed, actions in replays:
                print(f"回放第 {episode + 1} 局 | 种子 {seed} | 总奖励 {rewards[episode]:.2f}")
                env.reset(seed=seed)
                for action in actions:
                    _, _, terminated, truncated, _ = env.step(action)
                    if terminated or truncated:
                        break
    return rewards


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, help="默认使用 SAC 训练入口的保存路径")
    parser.add_argument("--no-render", action="store_true", help="只评分，跳过窗口回放")
    args = parser.parse_args()
    evaluate(args.checkpoint, render_mode=None if args.no_render else EVAL_CONFIG["render_mode"])


# 本文件主要代码由 AI 助手完成。
