"""评估 Reacher SAC，并随机回放部分评估回合。"""

import argparse
from pathlib import Path
import random
from statistics import mean

import gymnasium as gym
import torch

from actor import Actor_net  # Actor_net(state_space, action_space) -> actor


@torch.inference_mode()
def evaluate(model_path=None, eval_seeds=None, render_mode="human",
             num_episodes=100, num_visualize=10):
    if num_episodes < 1 or num_visualize < 0:
        raise ValueError("评估局数必须大于 0，可视化局数不能小于 0")
    seeds = (random.sample(range(1_000_000), num_episodes)
             if eval_seeds is None else list(eval_seeds))
    if not seeds:
        raise ValueError("至少需要一个评估种子")

    model_path = (Path(__file__).with_name("sac_checkpoint.pt")
                  if model_path is None else Path(model_path))
    checkpoint = torch.load(model_path, map_location="cpu", weights_only=True)
    actor = Actor_net(checkpoint["state_dimension"], checkpoint["action_dimension"])
    actor.load_state_dict(checkpoint["model_state_dict"])
    actor.eval()

    replay_indices = set()
    if render_mode is not None:
        replay_indices = set(random.sample(range(len(seeds)), min(num_visualize, len(seeds))))

    episode_rewards = []
    final_distances = []
    replays = []
    print(f"评估权重 {model_path}")

    with gym.make(checkpoint["env_id"], render_mode=None) as env:
        for episode, seed in enumerate(seeds):
            state, _ = env.reset(seed=seed)
            total_reward = 0.0
            actions = []
            while True:
                state_tensor = torch.as_tensor(state, dtype=torch.float32).unsqueeze(0)
                action_mean, _ = actor(state_tensor)
                action = torch.tanh(action_mean).squeeze(0).cpu().numpy()
                if episode in replay_indices:
                    actions.append(action.copy())
                state, reward, terminated, truncated, info = env.step(action)
                total_reward += float(reward)
                if terminated or truncated:
                    break
            # Reacher-v5 默认 reward_dist 是负的末端到目标距离。
            final_distance = -float(info["reward_dist"])
            episode_rewards.append(total_reward)
            final_distances.append(final_distance)
            if episode in replay_indices:
                replays.append((episode, seed, actions))
            print(f"第 {episode + 1} 局 | 种子 {seed} | 总奖励 {total_reward:.2f} | "
                  f"最终距离 {final_distance:.4f}")

    print(f"{len(episode_rewards)} 局平均奖励 {mean(episode_rewards):.2f}，"
          f"最低 {min(episode_rewards):.2f}，最高 {max(episode_rewards):.2f}")
    print(f"最终距离平均 {mean(final_distances):.4f}，"
          f"最小 {min(final_distances):.4f}，最大 {max(final_distances):.4f}")

    if replays:
        with gym.make(checkpoint["env_id"], render_mode=render_mode) as env:
            for episode, seed, actions in replays:
                print(f"回放第 {episode + 1} 局 | 种子 {seed} | "
                      f"总奖励 {episode_rewards[episode]:.2f}")
                env.reset(seed=seed)
                for action in actions:
                    _, _, terminated, truncated, _ = env.step(action)
                    if terminated or truncated:
                        break
    return episode_rewards, final_distances


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, help="默认使用本项目的 sac_checkpoint.pt")
    parser.add_argument("--no-render", action="store_true", help="只评分，跳过窗口回放")
    parser.add_argument("--num-episodes", type=int, default=100, help="评估局数")
    parser.add_argument("--num-visualize", type=int, default=10, help="随机回放的局数")
    args = parser.parse_args()
    evaluate(args.checkpoint, render_mode=None if args.no_render else "human",
             num_episodes=args.num_episodes, num_visualize=args.num_visualize)


# 本文件主要代码由 AI 助手完成。
