import math

import torch
from torch import nn
from torch.nn import functional as F


def elu_layers(input_dim, hidden_dims):
    widths = [input_dim, *hidden_dims]
    layers = []
    for input_width, output_width in zip(widths, widths[1:]):
        layers.extend((nn.Linear(input_width, output_width), nn.ELU()))
    return layers


class Actor(nn.Module):
    def __init__(self, state_dim, action_dim, hidden_dims):
        super().__init__()
        self.backbone = nn.Sequential(*elu_layers(state_dim, hidden_dims))
        self.mean = nn.Linear(hidden_dims[-1], action_dim)
        self.log_std = nn.Linear(hidden_dims[-1], action_dim)
        nn.init.zeros_(self.log_std.weight)
        nn.init.zeros_(self.log_std.bias)

    def forward(self, state):
        return self.mean(self.backbone(state)).tanh()

    def sample(self, state):
        features = self.backbone(state)
        mean = self.mean(features)
        log_std = self.log_std(features).clamp(-5, 2)
        distribution = torch.distributions.Normal(mean, log_std.exp())
        raw_action = distribution.rsample()
        log_jacobian = 2 * (math.log(2) - raw_action - F.softplus(-2 * raw_action))
        log_prob = (distribution.log_prob(raw_action) - log_jacobian).sum(dim=-1)
        return raw_action.tanh(), log_prob


class Critics(nn.Module):
    def __init__(self, state_dim, action_dim, hidden_dims):
        super().__init__()
        input_dim = state_dim + action_dim
        self.q1 = nn.Sequential(*elu_layers(input_dim, hidden_dims), nn.Linear(hidden_dims[-1], 1))
        self.q2 = nn.Sequential(*elu_layers(input_dim, hidden_dims), nn.Linear(hidden_dims[-1], 1))

    def forward(self, state, action):
        state_action = torch.cat((state, action), dim=-1)
        return self.q1(state_action).squeeze(-1), self.q2(state_action).squeeze(-1)


# 本文件主要代码由 AI 助手完成。
