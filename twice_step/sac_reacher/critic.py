import torch
import torch.nn as nn


class Critic_net(nn.Module):
    def __init__(self, state_space, action_space):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(state_space + action_space, 256),
            nn.Tanh(),

            nn.Linear(256, 1)
        )

    def forward(self, state, action):
        input = torch.cat((state, action), dim=1)
        values = self.net(input)
        return values
