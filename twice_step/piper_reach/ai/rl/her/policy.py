"""Piper's goal-conditioned SAC actor and shared observation contract."""

import math
from pathlib import Path

import numpy as np
import torch
from torch import nn
from torch.nn import functional as F


def features(state, goal):
    return torch.cat((state[..., :6] / math.pi, state[..., 6:12] / 3.0,
                      state[..., 12:15] * 2.0, state[..., 15:18] / 2.0,
                      (goal - state[..., 12:15]) * 2.0), dim=-1)


class Actor(nn.Module):
    def __init__(self, width=256):
        super().__init__()
        self.trunk = nn.Sequential(nn.Linear(21, width), nn.ReLU(), nn.Linear(width, width),
                                   nn.ReLU(), nn.Linear(width, width), nn.ReLU())
        self.head = nn.Linear(width, 12)
        nn.init.uniform_(self.head.weight, -0.003, 0.003)
        nn.init.zeros_(self.head.bias)

    def forward(self, observation):
        return self.head(self.trunk(observation))[..., :6].tanh()

    def sample(self, observation):
        mean, log_std = self.head(self.trunk(observation)).chunk(2, dim=-1)
        log_std = log_std.clamp(-5, 1)
        noise = torch.randn_like(mean)
        raw = mean + log_std.exp() * noise
        log_prob = (-0.5 * noise.square() - log_std - 0.5 * math.log(2 * math.pi)
                    - 2 * (math.log(2) - raw - F.softplus(-2 * raw))).sum(dim=-1)
        return raw.tanh(), log_prob


class LearnedPolicy:
    def __init__(self, weights):
        checkpoint = torch.load(Path(weights), map_location="cpu", weights_only=True)
        self.actor = Actor(checkpoint["width"])
        self.actor.load_state_dict(checkpoint["actor"])
        self.actor.eval()
        self.joint_step = checkpoint["joint_step"]
        self.training_steps = checkpoint["steps"]

    @torch.inference_mode()
    def action(self, env):
        linear_velocity, _ = env._reach_velocity()
        measured = np.concatenate((env.data.qpos[:6], env.data.qvel[:6],
                                   env.data.site("reach_point").xpos, linear_velocity))
        state = torch.as_tensor(measured, dtype=torch.float32)
        goal = torch.as_tensor(env.target_position, dtype=torch.float32)
        return self.actor(features(state, goal)).numpy()


# 本文件主要代码由 AI 助手完成。
