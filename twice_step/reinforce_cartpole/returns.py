# 计算单条已采集轨迹中每一步的折扣回报。
import torch


def compute_returns(trajectory, gamma):
    returns_list = []
    for one_trajectory in trajectory[::-1]:
        reward = one_trajectory["reward"]
        terminated = one_trajectory["terminated"]
        gt = 0
        if terminated or not returns_list:
            gt = reward
        else:
            gt = reward + gamma * returns_list[-1]
        returns_list.append(gt)
    returns_list.reverse()
    returns = torch.tensor(returns_list, dtype=torch.float32)
    return returns


if __name__ == '__main__':
    trajectory = [
        {"reward": 1.0, "terminated": False, "truncated": False},
        {"reward": 2.0, "terminated": False, "truncated": False},
        {"reward": 3.0, "terminated": True,  "truncated": False},
    ]
    gamma = 0.9
    returns = compute_returns(trajectory, gamma)
    print(returns)
