from pathlib import Path

import gymnasium as gym
import torch

from q_network import QNetwork  # QNetwork() -> Q network


if __name__ == "__main__":
    model_path = Path(__file__).resolve().parent / "dqn_checkpoint.pt"
    checkpoint = torch.load(model_path, map_location="cpu", weights_only=True)
    model = QNetwork(**checkpoint["network_config"])
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()

    eval_seeds = [100, 101, 102, 103, 104]
    episode_rewards = []
    env = gym.make("CartPole-v1")
    for seed in eval_seeds:
        state, _ = env.reset(seed=seed)
        total_reward = 0.0
        while True:
            state_tensor = torch.as_tensor(state, dtype=torch.float32)
            action = model.select_action(state_tensor, epsilon=0)
            state, reward, terminated, truncated, _ = env.step(action)
            total_reward += reward
            if terminated or truncated:
                break
        episode_rewards.append(total_reward)
        print(f"评估种子: {seed}, 本局总奖励: {total_reward:.1f}")
    env.close()
    mean_reward = sum(episode_rewards) / len(episode_rewards)
    print(f"{len(episode_rewards)} 局平均奖励: {mean_reward:.1f}")
