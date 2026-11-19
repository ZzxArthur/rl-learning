# SAC Critic 教学示例，网络实现见 networks.py。
import torch

from networks import Criticnet  # Criticnet(**network_config) -> Q-network


if __name__ == "__main__":
    states = torch.tensor([[1., 0., 0.], [0., 1., 0.5]])
    actions = torch.tensor([[0.4], [-0.8]])
    q1 = Criticnet()
    q2 = Criticnet()
    print(q1(states, actions), q2(states, actions))
