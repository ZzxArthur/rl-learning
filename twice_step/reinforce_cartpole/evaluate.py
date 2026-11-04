import torch
import gymnasium as gym

from checkpoint import load_model

def evalue_policy(env, policy_model, eval_seeds):
    num_reward = []
    for seed in eval_seeds:
        torch.manual_seed(seed)
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
    env.close()
    return num_reward


def evaluate_random_policy(env, eval_seeds):
    num_reward = []
    for seed in eval_seeds:
        env.reset(seed=seed)
        env.action_space.seed(seed)
        rewards = 0
        while True:

            action = env.action_space.sample()
            _, reward, terminated, truncated, _ = env.step(action)
            rewards += reward

            if terminated or truncated:
                break
        num_reward.append(rewards)
        print(f"种子为{seed},随机策略一局奖励为{rewards}")
    print(f"随机策略平均奖励为:{sum(num_reward) / len(eval_seeds)}")
    env.close()
    return num_reward


if __name__ == "__main__":
    eval_seeds = [i for i in range(100, 105)]
    env = gym.make('CartPole-v1')
    model_path = "twice_step/reinforce_cartpole/policy_weights.pt"
    policy_model = load_model(model_path)
    evalue_policy(env, policy_model, eval_seeds)
    env = gym.make('CartPole-v1')
    evaluate_random_policy(env, eval_seeds)

