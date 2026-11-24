import torch
import torch.nn as nn


class Actornet(nn.Module):
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(3, 128),
            nn.ReLU()
        )
        self.mean_head = nn.Linear(128, 1)
        self.log_std_head = nn.Linear(128, 1)

    def forward(self, state):
        features = self.net(state)
        mean = self.mean_head(features)
        log_std = self.log_std_head(features)
        return mean, log_std

    def sample_action(self, state):
        mean, log_std = self(state)
        std = log_std.clamp(-5, 2).exp()
        distribution = torch.distributions.Normal(mean, std)
        raw_action = distribution.rsample()
        u = torch.tanh(raw_action) # 双曲正切函数, 把-无穷, +无穷限制范围到-1, 1
        action = u * 2  # 满足pendulum的需求,电机只能在-2, 2之间
        correction = torch.log(2.0 * (1.0 - u.square()) + 1e-6) # u压缩到-1,1之后 概率密度函数也需要改变
        log_prob = (distribution.log_prob(raw_action) - correction).sum(dim=-1)

        return action, log_prob


if __name__ == "__main__":
    actor = Actornet()
    with torch.no_grad():
        for p in actor.parameters():
            p.zero_()

    torch.manual_seed(42)
    state = torch.tensor([[1.0, 0.0, 0.0]])
    action, log_prob = actor.sample_action(state)

    print(action.shape, action.item())      # torch.Size([1, 1])，约  0.64904
    print(log_prob.shape, log_prob.item())  # torch.Size([1])，约 -1.55749
