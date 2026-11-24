"""Goal-conditioned actor and learned action-outcome model."""

from dataclasses import dataclass

import numpy as np
import torch
from torch import nn


def mlp(in_features, out_features, width=256):
    return nn.Sequential(nn.Linear(in_features, width), nn.SiLU(), nn.Linear(width, width), nn.SiLU(),
                         nn.Linear(width, width), nn.SiLU(), nn.Linear(width, out_features))


class GoalActor(nn.Module):
    count = 6

    def __init__(self):
        super().__init__()
        self.net = mlp(5, self.count * 6, 384)

    def forward(self, goal):
        radius = goal[..., :2].norm(dim=-1, keepdim=True)
        azimuth = torch.atan2(goal[..., 1:2], goal[..., :1]) / torch.pi
        return self.net(torch.cat((2 * goal, 2 * radius, azimuth), dim=-1)).reshape(*goal.shape[:-1], self.count, 6).tanh()


class Outcome(nn.Module):
    def __init__(self):
        super().__init__()
        self.position = mlp(6, 3, 192)
        self.contact = mlp(6, 1, 160)

    def positions(self, q):
        return self.position(q) / 2

    def risk(self, q):
        return self.contact(q).squeeze(-1).sigmoid()


class Dynamics(nn.Module):
    def __init__(self):
        super().__init__()
        self.net = mlp(18, 12, 192)

    def forward(self, state, action):
        position, velocity = state.chunk(2, dim=-1)
        change, next_v = self.net(torch.cat((position, velocity, action - position), dim=-1)).chunk(2, dim=-1)
        return torch.cat((position + 0.05 * change, next_v), dim=-1)


class FeedbackActor(nn.Module):
    def __init__(self):
        super().__init__()
        self.net = mlp(21, 6, 192)

    def forward(self, state, error, previous):
        inp = torch.cat((state, 15 * error, 5 * (previous - state[..., :6])), dim=-1)
        return (previous + 0.035 * self.net(inp).tanh()).clamp(-1.0, 1.0)


def scores(model, commands, goal, home):
    fraction = torch.linspace(0.2, 1.0, 5, device=commands.device)
    points = home + fraction[:, None] * (commands.unsqueeze(-2) - home)
    risk = model.risk(points).amax(dim=-1)
    distance = (model.positions(commands) - goal.unsqueeze(-2)).norm(dim=-1)
    travel = (commands - home).abs().amax(dim=-1)
    return distance + 0.25 * risk + 0.006 * travel


@dataclass
class Policy:
    actor: GoalActor
    outcome: Outcome
    feedback: FeedbackActor | None
    home: torch.Tensor

    @torch.inference_mode()
    def __call__(self, state, target, previous_action=None):
        goal = torch.as_tensor(target, dtype=torch.float32)
        if previous_action is None:
            candidates = self.actor(goal)
            best = scores(self.outcome, candidates, goal, self.home).argmin()
            return candidates[best].numpy().copy()
        current = torch.as_tensor(state, dtype=torch.float32)
        previous = torch.as_tensor(previous_action, dtype=torch.float32)
        error = goal - current[12:]
        if self.feedback is None or error.norm() > 0.09 or (previous - current[:6]).abs().max() > 0.09:
            return np.asarray(previous_action, dtype=np.float32)
        return self.feedback(current[:12], error, previous).numpy().copy()


def load_policy(path):
    torch.set_num_threads(1)
    weights = torch.load(path, map_location="cpu", weights_only=True)
    actor, outcome = GoalActor(), Outcome()
    actor.load_state_dict(weights["actor"])
    outcome.load_state_dict(weights["outcome"])
    feedback = None
    if weights.get("feedback") is not None:
        feedback = FeedbackActor()
        feedback.load_state_dict(weights["feedback"])
        feedback.eval()
    return Policy(actor.eval(), outcome.eval(), feedback, weights["home"])


# 本文件主要代码由 AI 助手完成。
