import torch
import gymnasium as gym
import random
from pathlib import Path

from checkpoint import load_model


def evalue_policy(env, policy_model, eval_seeds):
    num_reward = []
    for seed in eval_seeds:

        state, info = env.reset(seed=seed)
        state = torch.as_tensor(state, dtype=torch.float32).unsqueeze(0)
        rewards = 0
        while True:
            with torch.no_grad():
                logits = policy_model(state)
            action = torch.argmax(logits).item()
            next_state, reward, terminated, truncated, info = env.step(action)
            next_state = torch.tensor(next_state, dtype=torch.float32).unsqueeze(0)
            rewards += reward

            if terminated or truncated:
                break
            state = next_state
        num_reward.append(rewards)
        print(f"种子为{seed},这一局奖励为{rewards}")
    print(f"平均奖励为:{sum(num_reward) / len(eval_seeds)}")
    return num_reward


if __name__ == "__main__":
    eval_seeds = random.sample(range(1_000_000), 5)
    model_path = Path(__file__).resolve().parent / "policy_weights.pt"
    policy_model = load_model(model_path)
    env = gym.make('CartPole-v1', render_mode='human')
    try:
        evalue_policy(env, policy_model, eval_seeds)
    finally:
        env.close()
