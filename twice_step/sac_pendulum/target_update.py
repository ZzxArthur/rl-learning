# SAC 目标 Critic：用 Polyak 平均让两套目标参数分别跟随在线网络。
import torch

from updates import soft_update  # soft_update(online_q1, online_q2, target_q1, target_q2, tau) -> None


if __name__ == "__main__":
    from critic import Criticnet  # Criticnet() -> Q-network

    online_q1 = Criticnet()
    online_q2 = Criticnet()
    target_q1 = Criticnet()
    target_q2 = Criticnet()
    target_q1.load_state_dict(online_q1.state_dict())
    target_q2.load_state_dict(online_q2.state_dict())

    with torch.no_grad():
        online_q1.net[0].weight[0, 0] = 10.0
        target_q1.net[0].weight[0, 0] = 2.0

    soft_update(online_q1, online_q2, target_q1, target_q2, tau=0.25)
    print("online_q1:", online_q1.net[0].weight[0, 0].item())
    print("target_q1:", target_q1.net[0].weight[0, 0].item())
