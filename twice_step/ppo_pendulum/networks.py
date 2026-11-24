# 连续PPO网络：Actor预测高斯均值并学习标准差，Critic预测状态价值。
# 保留原始连续动作，支持采样或重算指定动作的log-prob与熵。
import torch


class Ppo_net(torch.nn.Module):
    def __init__(self, state_dimension, action_dimension, hidden_dimension, activation="tanh"):
        super().__init__()

        activation_cls = {"tanh": torch.nn.Tanh, "elu": torch.nn.ELU}[activation]
        widths = [state_dimension, *hidden_dimension]
        for name, output_dimension in (("actor_net", action_dimension), ("critic_net", 1)):
            layers = []
            for input_width, output_width in zip(widths, widths[1:]):
                layers.extend((torch.nn.Linear(input_width, output_width), activation_cls()))
            layers.append(torch.nn.Linear(widths[-1], output_dimension))
            setattr(self, name, torch.nn.Sequential(*layers))

        self.log_std = torch.nn.Parameter(torch.zeros(action_dimension))

    def forward(self, state, action=None):
        action_mean = self.actor_net(state)
        action_std = self.log_std.exp()
        distributions = torch.distributions.Normal(loc=action_mean, scale=action_std)
        entropy = distributions.entropy().sum(-1)  # 策略熵衡量的是：在当前状态下，动作概率有多分散。
        if action is None:
            action = distributions.sample()
        action_log_probs = distributions.log_prob(action).sum(-1)
        values = self.critic_net(state).squeeze(-1)  # [batch_size, 1] -> [batch_size]
        return action, action_log_probs, values, entropy


if __name__ == "__main__":
    state_dimension = 3
    action_dimension = 1
    hidden_dimension = [64, 64]

    test_states = torch.tensor([
        [1.0,  0.0,  0.0],
        [0.0,  1.0,  0.0],
        [-1.0, 0.0,  0.5],
        [0.0, -1.0, -0.5],
    ], dtype=torch.float32)

    torch.manual_seed(42)

    distributions = torch.distributions.Normal(torch.tensor(0.0), torch.tensor(1.0))
    log_std = torch.nn.Parameter(torch.zeros(1))


