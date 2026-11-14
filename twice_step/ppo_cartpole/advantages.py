# PPO优势估计：计算TD误差、带终止/截断边界的GAE和价值回归目标。
# 批次末尾的后续优势从0开始；未真正终止的转移仍使用next_values自举。
import torch


def compute_gae(rewards, values, next_values, terminated, truncated, gamma, gae_lambda):
    lenth = len(rewards)
    td_error = torch.zeros_like(values)
    advantage = torch.zeros_like(values)
    for i in range(lenth):
        td_target = rewards[i] + gamma * next_values[i] * (~terminated[i])
        td_error[i] = td_target - values[i]

    for j in range(-1, -lenth - 1, -1):
        advantage[j] = td_error[j] + gamma * gae_lambda * advantage[j+1] * (~terminated[j]) * (~truncated[j])

    values_target = advantage + values
    return td_error, advantage, values_target


if __name__ == "__main__":
    rewards = torch.tensor([1.0, 2.0, 3.0])
    values = torch.tensor([0.5, 0.6, 0.7])
    next_values = torch.tensor([0.6, 4.0, 0.0])
    terminated = torch.tensor([False, False, True])
    truncated = torch.tensor([False, True, False])
    gamma = 0.9
    gae_lambda = 0.8
    td_error, advantage, values_target = compute_gae(
        rewards, values, next_values, terminated, truncated, gamma, gae_lambda)
    print("TD误差:", td_error)
    print("GAE:", advantage)
    print("价值回归目标:", values_target)
