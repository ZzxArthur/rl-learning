# 离散PPO网络：独立Actor/Critic，支持采样动作或重算指定动作的log-prob。
# 同时计算状态价值和策略熵，供后续rollout采集与PPO更新使用。
import torch


class Ppo_net(torch.nn.Module):
    def __init__(self, state_dimension=4, action_dimension=2, hidden_dimension=[64, 64]):
        super().__init__()
        self.actor_net = torch.nn.Sequential(
            torch.nn.Linear(state_dimension, hidden_dimension[0]),
            torch.nn.Tanh(),
            torch.nn.Linear(hidden_dimension[0], hidden_dimension[1]),
            torch.nn.Tanh(),
            torch.nn.Linear(hidden_dimension[1], action_dimension)
        )
        self.critic_net = torch.nn.Sequential(
            torch.nn.Linear(state_dimension, hidden_dimension[0]),
            torch.nn.Tanh(),
            torch.nn.Linear(hidden_dimension[0], hidden_dimension[1]),
            torch.nn.Tanh(),
            torch.nn.Linear(hidden_dimension[1], 1)
        )

    def forward(self, state, action=None):

        logits = self.actor_net(state)
        action_distribution = torch.distributions.Categorical(logits=logits)
        entropy = action_distribution.entropy()  # 策略熵衡量的是：在当前状态下，动作概率有多分散。
        if action is None:
            action = action_distribution.sample()
        action_log_prob = action_distribution.log_prob(action)

        values = self.critic_net(state).squeeze(-1)  # [batch_size, 1] -> [batch_size]
        return action, action_log_prob, values, entropy


