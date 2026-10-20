import torch


rewards = torch.tensor(
    [1.0, 0.0, 2.0], dtype=torch.float32
)  # shape: (3,)，按时间顺序记录三步奖励

gamma = 0.9  # 折扣因子，每往后一步，奖励的权重再乘一次 gamma

all_reward = 0
for i in range(rewards.numel()):
    all_reward += rewards[i].item() * gamma ** (i)
    
print(all_reward)