# 练习23：从轨迹提取即时奖励，封装反向递推函数计算各时间步的折扣回报。
# 核对不同长度的已结束轨迹，并对非一维奖励张量抛出异常。
import torch

def compute_returns(rewards, gamma):
    if rewards.ndim != 1:
        raise ValueError("奖励函数维度不匹配")

    g_t = torch.zeros_like(rewards)
    for i in range(g_t.numel()-1, -1, -1):
        if i + 1 < g_t.numel():
            g_t[i] = rewards[i] + gamma * g_t[i+1]
        else:
            g_t[i] = rewards[i]
    return g_t


trajectory_a = [
    ("S0", 0, 1.0, "S1", False),
    ("S1", 1, 2.0, "T", True),
]
trajectory_b = [
    ("S0", 1, 0.0, "T", True),
]

rewards_a = torch.tensor([step[2] for step in trajectory_a], dtype=torch.float32)
rewards_b = torch.tensor([step[2] for step in trajectory_b], dtype=torch.float32)
gamma = 0.9

g_t_a = compute_returns(rewards_a, gamma)
g_t_b = compute_returns(rewards_b, gamma)

print(rewards_a)
print(rewards_b)
print(g_t_a)
print(g_t_b)

expected_returns_a = torch.tensor([2.8, 2.0], dtype=torch.float32)
expected_returns_b = torch.tensor([0.0], dtype=torch.float32)

print(torch.allclose(g_t_a, expected_returns_a))
print(torch.allclose(g_t_b, expected_returns_b))
