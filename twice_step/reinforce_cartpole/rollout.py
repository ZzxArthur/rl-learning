import torch
from policy import PolicyNetwork
import gymnasium as gym


def collect_trajectory(env, policy_model, seed):
    state, info = env.reset(seed=seed)  # state为numpy需要转换为张量,用torch.as_tensor转换并增加批次维度
    state = torch.as_tensor(state, dtype=torch.float32).unsqueeze(0)
    trajectory = []
    while True:
        with torch.no_grad():
            logits = policy_model(state)
        action_distributions = torch.distributions.Categorical(logits=logits)
        action = action_distributions.sample().item()
        next_state, reward, terminated, truncated, info = env.step(action)
        next_state = torch.as_tensor(next_state, dtype=torch.float32).unsqueeze(0)

        trajectory.append({'state': state, 'action': action,
                           'reward': reward, 'next_state': next_state,
                           'terminated': terminated, 'truncated': truncated})
        if terminated or truncated:
            break
        state = next_state
    return trajectory


if __name__ == "__main__":
    torch.manual_seed(42)
    env = gym.make('CartPole-v1')
    policy_model = PolicyNetwork()
    seed = 42
    trajectory = collect_trajectory(env, policy_model, seed)

    env.close()
    print(trajectory)
