# 练习21：对已结束的奖励序列反向递推，计算每个时间步的折扣回报。
# 使用同形状张量保存结果，末步之后的回报取零，并按原时间顺序输出。
import torch


rewards = torch.tensor(
    [1.0, 0.0, 2.0], dtype=torch.float32
)  # shape: (3,)，按时间顺序记录三步奖励

gamma = 0.9  # 折扣因子，每往后一步，奖励的权重再乘一次 gamma
g_t = torch.zeros_like(rewards)  # shape: (3,)，存储每一步的折扣奖励

for i in range(rewards.numel() - 1, -1, -1):

    if i + 1 < rewards.numel():
        g_t[i] = rewards[i] + gamma * g_t[i + 1]  # 贝尔曼方程  反向递推
    else:
        g_t[i] = rewards[i]
    
print(g_t)
