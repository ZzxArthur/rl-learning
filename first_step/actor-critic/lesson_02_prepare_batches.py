# 练习2：整理多局轨迹，将状态、动作和各步回报按相同顺序拼接。

import torch
from lesson_01_batch_returns import computer_batch_rewards


def prepare_batch(episodes, gamma):
    # 返回 states, next_states, actions, returns, terminated
    states_by_episodes = []
    actions_by_episodes = []
    rewards_by_episodes = []
    next_states_by_episodes = []
    terminated_by_episodes = []
    for num_episodes in episodes:
        states = num_episodes["states"]
        actions = num_episodes["actions"]
        rewards = num_episodes["rewards"]
        next_states = num_episodes['next_states']
        terminated = num_episodes['terminated']
        states_by_episodes.append(states)
        actions_by_episodes.append(actions)
        rewards_by_episodes.append(rewards)
        next_states_by_episodes.append(next_states)
        terminated_by_episodes.append(terminated)
    states_by_episodes = torch.cat(states_by_episodes, dim=0)
    actions_by_episodes = torch.cat(actions_by_episodes, dim=0)
    next_states_by_episodes = torch.cat(next_states_by_episodes, dim=0)
    terminated_by_episodes = torch.cat(terminated_by_episodes, dim=0)
    gt_by_episode, flat_returns = computer_batch_rewards(rewards_by_episodes, gamma)
    return states_by_episodes, next_states_by_episodes, actions_by_episodes, flat_returns, terminated_by_episodes


def prepare_td_batch(episodes):
    # 返回 states, next_states, actions, rewards, terminated
    states_by_episodes = []
    actions_by_episodes = []
    rewards_by_episodes = []
    next_states_by_episodes = []
    terminated_by_episodes = []
    for num_episodes in episodes:
        states = num_episodes["states"]
        actions = num_episodes["actions"]
        rewards = num_episodes["rewards"]
        next_states = num_episodes['next_states']
        terminated = num_episodes['terminated']
        states_by_episodes.append(states)
        actions_by_episodes.append(actions)
        rewards_by_episodes.append(rewards)
        next_states_by_episodes.append(next_states)
        terminated_by_episodes.append(terminated)
    states_by_episodes = torch.cat(states_by_episodes, dim=0)
    next_states_by_episodes = torch.cat(next_states_by_episodes, dim=0)
    actions_by_episodes = torch.cat(actions_by_episodes, dim=0)
    rewards_by_episodes = torch.cat(rewards_by_episodes, dim=0)
    terminated_by_episodes = torch.cat(terminated_by_episodes, dim=0)

    return states_by_episodes, next_states_by_episodes, actions_by_episodes, rewards_by_episodes, terminated_by_episodes


if __name__ == "__main__":
    episodes = [
        {
            "states": torch.tensor(
                [[1.0, 0.0], [0.0, 1.0]],
                dtype=torch.float32,
            ),
            "actions": torch.tensor([0, 1], dtype=torch.long),
            "rewards": torch.tensor([1.0, 1.0], dtype=torch.float32),
            "next_states": torch.tensor(
                [[0.0, 1.0], [0.0, 0.0]], dtype=torch.float32
            ),
            "terminated": torch.tensor([False, True], dtype=torch.bool),
        },
        {
            "states": torch.tensor(
                [[1.0, 1.0]],
                dtype=torch.float32,
            ),
            "actions": torch.tensor([1], dtype=torch.long),
            "rewards": torch.tensor([2.0], dtype=torch.float32),
            "next_states": torch.tensor([[0.0, 0.0]], dtype=torch.float32),
            "terminated": torch.tensor([True], dtype=torch.bool),
        },
    ]

    gamma = 0.9

    print(prepare_batch(episodes, gamma))
