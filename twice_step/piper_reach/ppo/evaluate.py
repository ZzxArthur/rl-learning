# 本文件主要代码由 AI 助手完成。
# 在独立种子上评估 PPO 模型，可选回放随机挑选的局数。
from pathlib import Path
import argparse
import sys
import time

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))
from twice_step.piper_reach.ppo.env import PiperReachEnv
from twice_step.ppo_pendulum.networks import Ppo_net


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Evaluate PPO on Piper Reach")
    parser.add_argument("checkpoint", type=Path, help="Piper PPO 权重文件")
    parser.add_argument("--episodes", type=int, default=100, help="评估局数")
    parser.add_argument("--seed", type=int, default=10000, help="评估种子的起始值")
    parser.add_argument("--visualize", type=int, default=0, help="随机选择多少局回放")
    args = parser.parse_args()
    if args.episodes < 1 or not 0 <= args.visualize <= args.episodes:
        parser.error("episodes 必须大于 0，visualize 必须在 0 到 episodes 之间")

    torch.set_num_threads(1)
    checkpoint = torch.load(args.checkpoint, map_location="cpu", weights_only=True)
    config = checkpoint["network_config"]
    model = Ppo_net(config["state_dimension"], config["action_dimension"], config["hidden_dimension"],
                    activation=config.get("activation", "tanh"))
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()

    legacy_observation = config["state_dimension"] == 18
    env = PiperReachEnv(legacy_observation=legacy_observation)
    for name, value in checkpoint["environment_config"].items():
        setattr(env, name, value)
    if config["state_dimension"] != env.observation_space.shape[0] or config["action_dimension"] != env.action_space.shape[0]:
        env.close()
        raise ValueError("权重的状态或动作维度与 Piper Reach 环境不一致")

    results = []
    try:
        for episode in range(args.episodes):
            episode_seed = args.seed + episode
            state, info = env.reset(seed=episode_seed)
            initial_distance = float(info["distance"])
            total_reward = 0.0
            terminated = truncated = False
            while not (terminated or truncated):
                with torch.no_grad():
                    state_tensor = torch.as_tensor(state, dtype=torch.float32)
                    action = model.actor_net(state_tensor).tanh().numpy()
                state, reward, terminated, truncated, info = env.step(action)
                total_reward += float(reward)
            results.append((episode_seed, initial_distance, float(info["distance"]),
                            bool(info["is_success"]), total_reward))
    finally:
        env.close()

    initial_distances = np.array([result[1] for result in results])
    final_distances = np.array([result[2] for result in results])
    successes = sum(result[3] for result in results)
    print(f"评估 {args.episodes} 局 | 成功 {successes}/{args.episodes} "
          f"({successes / args.episodes:.1%}) | 平均奖励 {np.mean([result[4] for result in results]):.3f}")
    print(f"初始距离均值 {initial_distances.mean():.4f} 米 | "
          f"最终距离均值 {final_distances.mean():.4f} 米 | "
          f"最小 {final_distances.min():.4f} 米 | 最大 {final_distances.max():.4f} 米")

    if args.visualize:
        indices = np.random.default_rng(args.seed).choice(args.episodes, size=args.visualize, replace=False)
        env = PiperReachEnv(render_mode="human", legacy_observation=legacy_observation)
        for name, value in checkpoint["environment_config"].items():
            setattr(env, name, value)
        try:
            for index in sorted(indices):
                episode_seed, _, expected_distance, expected_success, _ = results[index]
                state, _ = env.reset(seed=episode_seed)
                print(f"回放第 {index + 1} 局 | 种子 {episode_seed} | "
                      f"评估距离 {expected_distance:.4f} 米 | 成功 {expected_success}", flush=True)
                terminated = truncated = False
                while env.viewer.is_running() and not (terminated or truncated):
                    started_at = time.monotonic()
                    with torch.no_grad():
                        state_tensor = torch.as_tensor(state, dtype=torch.float32)
                        action = model.actor_net(state_tensor).tanh().numpy()
                    state, _, terminated, truncated, _ = env.step(action)
                    elapsed = time.monotonic() - started_at
                    time.sleep(max(0.0, env.frame_skip * env.model.opt.timestep - elapsed))
                if not env.viewer.is_running():
                    break
        except KeyboardInterrupt:
            print("\n回放已中断。")
        finally:
            env.close()
