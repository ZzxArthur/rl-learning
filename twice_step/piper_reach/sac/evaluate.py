import argparse
from pathlib import Path
import sys

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))
from twice_step.piper_reach.ppo.env import PiperReachEnv
from twice_step.piper_reach.sac.networks import Actor
from twice_step.ppo_pendulum.networks import Ppo_net


@torch.inference_mode()
def evaluate(path, seeds, fixed_start=None, success_distance=None):
    checkpoint = torch.load(path, map_location="cpu", weights_only=True)
    config = checkpoint["network_config"]
    dimensions = (config["state_dimension"], config["action_dimension"], config["hidden_dimension"])
    if checkpoint.get("algorithm") == "SAC":
        actor = Actor(*dimensions)
        actor.load_state_dict(checkpoint["actor_state_dict"])
        policy = actor.eval()
    else:
        model = Ppo_net(*dimensions, activation=config.get("activation", "tanh"))
        model.load_state_dict(checkpoint["model_state_dict"])
        model.eval()

        def policy(state):
            return model.actor_net(state).tanh()

    env = PiperReachEnv(legacy_observation=config["state_dimension"] == 18)
    for key, value in checkpoint["environment_config"].items():
        setattr(env, key, value)
    # 用相同的原始奖励衡量使用不同训练奖励的权重。
    env.precision_weight = 0.0
    if fixed_start is not None:
        env.fixed_start = fixed_start
    if success_distance is not None:
        env.success_distance = success_distance
    results = []
    with env:
        for seed in seeds:
            state, info = env.reset(seed=seed)
            initial_distance = closest_distance = info["distance"]
            total_reward = 0.0
            terminated = info["is_success"]
            truncated = False
            while not (terminated or truncated):
                action = policy(torch.as_tensor(state)).numpy()
                state, reward, terminated, truncated, info = env.step(action)
                total_reward += reward
                closest_distance = min(closest_distance, info["distance"])
            results.append((terminated, initial_distance, info["distance"], closest_distance, total_reward))

    results = np.asarray(results)
    successes = int(results[:, 0].sum())
    print(f"{path.parent.name}/{path.name} | 训练 {checkpoint['total_env_steps']:,} 步 | 阈值 {env.success_distance * 1000:g} 毫米")
    print(f"成功 {successes}/{len(results)} ({successes / len(results):.1%}) | 平均原始回报 {results[:, 4].mean():.3f}")
    print(f"初始均距 {results[:, 1].mean():.4f} 米 | 最终均距 {results[:, 2].mean():.4f} 米 | "
          f"各局最近距离均值 {results[:, 3].mean():.4f} 米 | 全部轨迹最近 {results[:, 3].min():.4f} 米", flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Compare deterministic Piper policies in CPU MuJoCo")
    parser.add_argument("checkpoints", type=Path, nargs="+")
    parser.add_argument("--episodes", type=int, default=10)
    parser.add_argument("--seed", type=int, default=10000)
    parser.add_argument("--success-distance", type=float,
                        help="覆盖权重记录的成功距离，单位为米")
    parser.add_argument("--fixed-start", action=argparse.BooleanOptionalAction, default=None,
                        help="用同一起点设置比较多个权重，省略时采用各自训练设置")
    args = parser.parse_args()
    if args.episodes < 1:
        parser.error("评估局数必须大于 0")
    if args.success_distance is not None and (not np.isfinite(args.success_distance) or args.success_distance <= 0):
        parser.error("成功距离必须是有限正数")
    torch.set_num_threads(1)
    seeds = range(args.seed, args.seed + args.episodes)
    for checkpoint in args.checkpoints:
        evaluate(checkpoint, seeds, fixed_start=args.fixed_start, success_distance=args.success_distance)


# 本文件主要代码由 AI 助手完成。
