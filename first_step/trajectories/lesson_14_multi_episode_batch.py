# 采集多局 CartPole 轨迹，逐局计算回报后拼成批次并更新一次策略。
import torch
import gymnasium as gym

from lesson_10_cartpole_reinforce import collect_episode  # collect_episode(env, seed, model) -> trajectory
from lesson_10_cartpole_reinforce import update  # update(model, optimizer, states, actions, returns) -> loss
from lesson_12_trajectory_inspection import inspect_episodes  # inspect_episodes(episodes, gamma) -> states, actions, rewards, returns


def prepare_episode_batch(episodes, gamma):
    # 每局先计算回报，再按局的顺序拼接训练数据。
    states_by_trajectory = []
    actions_by_trajectory = []
    rewards_by_trajectory = []
    returns_by_trajectory = []
    for trajectory in episodes:
        states, actions, rewards, returns = inspect_episodes(trajectory, gamma)
        states_by_trajectory.append(states)
        actions_by_trajectory.append(actions)
        rewards_by_trajectory.append(rewards)
        returns_by_trajectory.append(returns)
    states_by_trajectory = torch.cat(states_by_trajectory, dim=0)
    actions_by_trajectory = torch.cat(actions_by_trajectory, dim=0)
    rewards_by_trajectory = torch.cat(rewards_by_trajectory, dim=0)
    returns_by_trajectory = torch.cat(returns_by_trajectory, dim=0)
    return states_by_trajectory, actions_by_trajectory, rewards_by_trajectory, returns_by_trajectory



if __name__ == "__main__":
    torch.manual_seed(42)
    env = gym.make("CartPole-v1")
    policy_model = torch.nn.Linear(in_features=4, out_features=2)
    policy_optimizer = torch.optim.SGD(policy_model.parameters(), lr=0.01)
    gamma = 0.9
    num_batch = 10
    episodes = []
    for episode_index in range(num_batch):
        seed = 42 if episode_index == 0 else None
        trajectory = collect_episode(env, seed, policy_model)
        episodes.append(trajectory)
        print(f"第 {episode_index + 1} 局步数: {len(trajectory)}")
    env.close()
    states, actions, rewards, returns = prepare_episode_batch(episodes, gamma)
    print(f"批次形状: states={tuple(states.shape)}, actions={tuple(actions.shape)}, returns={tuple(returns.shape)}")
    loss = update(policy_model, policy_optimizer, states, actions, returns)
    print(f"更新前策略损失: {loss.item():.4f}")

