# 练习27：以传入的末尾值为递推起点，倒序计算各时间步的折扣回报。
# 末尾值为零可表示真正终止，非零值用于练习截断后的后续回报估计。
import torch


def compute_returns(rewards, gamma, trail_value):
    if rewards.ndim != 1:
        raise ValueError("奖励函数维度不匹配")
    g_t = torch.zeros_like(rewards)
    terminated = False
    truncated = True

    for t in range(g_t.numel()-1, -1, -1):
        print(t)
        if t + 1 < g_t.numel():
            g_t[t] = rewards[t] + gamma * g_t[t+1] 
        elif terminated:
            g_t[t] = rewards[t]
        elif truncated:
            g_t[t] = rewards[t] + gamma * trail_value
    return g_t


rewards = torch.tensor([1.0, 2.0], dtype=torch.float32)
trail_value = 5.0
gamma = 0.9

gt = compute_returns(rewards, gamma, trail_value)
print(gt)
