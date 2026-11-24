import torch
import torch.nn as nn


class Actor_net(nn.Module):
    def __init__(self, state_space, action_space):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(state_space, 256),
            nn.Tanh(),
        )
        self.mean_head = nn.Linear(256, action_space)
        self.log_std_head = nn.Linear(256, action_space)

    def forward(self, state):
        feature = self.net(state)
        mean = self.mean_head(feature)
        log_std = self.log_std_head(feature)
        return mean, log_std

    def sample(self, state):
        mean, log_std = self(state)
        log_std = torch.clamp(log_std, min=-20, max=2)
        std = torch.exp(log_std)
        # 1. 构建高斯分布并重参数化采样 (rsample = mean + std * N(0, 1))
        distributions = torch.distributions.Normal(mean, std)
        u = distributions.rsample()

        # 2. 用 tanh 将无界动作压缩到 [-1, 1]（若环境动作范围是 [-2, 2]，再乘以 action_scale）
        action = torch.tanh(u)

        # 3. 计算带雅可比行列式修正的 log_prob（用于更新策略网络和温度系数 alpha）
        log_prob = distributions.log_prob(u) - torch.log(1 - action ** 2 + 1e-6)
        log_prob = log_prob.sum(dim=-1, keepdim=True)
        return action, log_prob
