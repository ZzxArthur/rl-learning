# 检查 CartPole 轨迹中状态、动作、奖励和回报的对齐。
import torch
import gymnasium as gym
import numpy as np
from lesson_10_cartpole_reinforce import collect_episode  # collect_episode(env, seed, model_sample) -> trajectory
from lesson_10_cartpole_reinforce import compute_returns  # compute_returns(rewards, gamma, trajectory, trail_value=0.0) -> g_t


def inspect_episodes(episodes, gamma):
    #  [obs, action, reward, next_obs, terminated, truncated]
    states_list = [step[0] for step in episodes]
    actions_list = [step[1] for step in episodes]
    rewards_list = [step[2] for step in episodes]
    states = torch.tensor(np.stack(states_list), dtype=torch.float32)
    actions = torch.tensor(actions_list, dtype=torch.long)
    rewards = torch.tensor(rewards_list, dtype=torch.float32)
    returns = compute_returns(rewards, gamma, episodes)
    return states, actions, rewards, returns



if __name__ == "__main__":
    torch.manual_seed(42)
    env = gym.make("CartPole-v1")
    model = torch.nn.Linear(4, 2)
    seed = 42
    gamma = 0.99

    episodes = collect_episode(env, seed, model)
    states, actions, rewards, returns = inspect_episodes(episodes, gamma)
    env.close()

    print(f"轨迹长度: {len(episodes)}")
    for name, data in (("states", states), ("actions", actions), ("rewards", rewards), ("returns", returns)):
        print(f"{name}: shape={tuple(data.shape)}, dtype={data.dtype}")
    print(f"最后一步 terminated, truncated: {episodes[-1][4:6]}")
    for i in range(min(3, len(episodes))):
        print(f"第 {i} 步: action={actions[i].item()}, reward={rewards[i].item():.1f}, return={returns[i].item():.4f}")
