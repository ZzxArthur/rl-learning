# 练习46：整理多局轨迹，将状态、动作和各步回报按相同顺序拼接。

import torch
from pra45 import computer_batch_rewards


def prepare_batch(episodes, gamma):
    # 返回 states, actions, returns
    states_by_episodes = []
    actions_by_episodes = []
    rewards_by_episodes = []
    for num_episodes in episodes:
        states = num_episodes["states"]
        actions = num_episodes["actions"]
        rewards = num_episodes["rewards"]
        states_by_episodes.append(states)
        actions_by_episodes.append(actions)
        rewards_by_episodes.append(rewards)
    states_by_episodes = torch.cat(states_by_episodes, dim=0)
    actions_by_episodes = torch.cat(actions_by_episodes, dim=0)
    gt_by_episode, flat_returns = computer_batch_rewards(rewards_by_episodes, gamma)
    return states_by_episodes, actions_by_episodes, flat_returns

if __name__ == "__main__":
    episodes = [
        {
            "states": torch.tensor(
                [[1.0, 0.0], [0.0, 1.0]],
                dtype=torch.float32,
            ),
            "actions": torch.tensor([0, 1], dtype=torch.long),
            "rewards": torch.tensor([1.0, 1.0], dtype=torch.float32),
        },
        {
            "states": torch.tensor(
                [[1.0, 1.0]],
                dtype=torch.float32,
            ),
            "actions": torch.tensor([1], dtype=torch.long),
            "rewards": torch.tensor([2.0], dtype=torch.float32),
        },
    ]

    gamma = 0.9

    print(prepare_batch(episodes, gamma))
