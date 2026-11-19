# SAC 网络：Actor 生成有界动作，Critic 为状态动作对估值。
import torch
import torch.nn as nn


def _hidden_layers(input_dimension, hidden_dimensions, activation):
    activations = {"relu": nn.ReLU, "tanh": nn.Tanh}
    if activation not in activations:
        raise ValueError("activation 必须为 relu 或 tanh")
    layers = []
    for width in hidden_dimensions:
        layers.extend((nn.Linear(input_dimension, width), activations[activation]()))
        input_dimension = width
    return layers, input_dimension


class Actornet(nn.Module):
    def __init__(self, state_dimension=3, action_dimension=1, hidden_dimensions=(128,),
                 activation="relu", log_std_min=-5.0, log_std_max=2.0,
                 log_prob_epsilon=1e-6, action_low=-2.0, action_high=2.0):
        super().__init__()
        if log_std_min > log_std_max or log_prob_epsilon <= 0:
            raise ValueError("log_std 范围必须有序，log_prob_epsilon 必须大于 0")
        layers, width = _hidden_layers(state_dimension, hidden_dimensions, activation)
        self.net = nn.Sequential(*layers)
        self.mean_head = nn.Linear(width, action_dimension)
        self.log_std_head = nn.Linear(width, action_dimension)
        self.log_std_min = log_std_min
        self.log_std_max = log_std_max
        self.log_prob_epsilon = log_prob_epsilon

        low = torch.as_tensor(action_low, dtype=torch.get_default_dtype()).detach()
        high = torch.as_tensor(action_high, dtype=torch.get_default_dtype()).detach()
        if low.ndim == 0:
            low = low.expand(action_dimension)
        if high.ndim == 0:
            high = high.expand(action_dimension)
        if low.shape != (action_dimension,) or high.shape != (action_dimension,):
            raise ValueError("动作上下界需为标量或与动作维数相同的一维序列")
        if not (torch.isfinite(low).all() and torch.isfinite(high).all() and (high > low).all()):
            raise ValueError("动作上下界必须有限，且每维上界都大于下界")
        # 动作范围来自环境配置；非持久 buffer 随模型移动并兼容旧权重。
        self.register_buffer("action_scale", (high - low) / 2, persistent=False)
        self.register_buffer("action_bias", (high + low) / 2, persistent=False)

    def forward(self, state):
        features = self.net(state)
        mean = self.mean_head(features)
        log_std = self.log_std_head(features)
        return mean, log_std

    def sample_action(self, state):
        mean, log_std = self(state)
        std = log_std.clamp(self.log_std_min, self.log_std_max).exp()
        distribution = torch.distributions.Normal(mean, std)
        raw_action = distribution.rsample()
        u = torch.tanh(raw_action)
        action = u * self.action_scale + self.action_bias
        correction = torch.log(self.action_scale * (1.0 - u.square()) + self.log_prob_epsilon)
        log_prob = (distribution.log_prob(raw_action) - correction).sum(dim=-1)
        return action, log_prob

    def deterministic_action(self, state):
        mean, _ = self(state)
        return torch.tanh(mean) * self.action_scale + self.action_bias


class Criticnet(nn.Module):
    def __init__(self, state_dimension=3, action_dimension=1, hidden_dimensions=(128,), activation="relu"):
        super().__init__()
        layers, width = _hidden_layers(state_dimension + action_dimension, hidden_dimensions, activation)
        self.net = nn.Sequential(*layers, nn.Linear(width, 1))

    def forward(self, states, actions):
        q_input = torch.cat((states, actions), dim=-1)
        values = self.net(q_input)
        return values.squeeze(-1)


# 采样与估值逻辑沿用用户的 Actor/Critic 实现。
# 本文件主要代码由 AI 助手完成。
