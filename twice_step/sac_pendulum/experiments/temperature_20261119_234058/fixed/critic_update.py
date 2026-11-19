# SAC Critic 更新：两个在线 Q 网络分别拟合同一个 TD 目标。
import torch

from updates import update_critics  # update_critics(q1, q2, optimizer, states, actions, td_target) -> critic_loss


if __name__ == "__main__":
    from critic import Criticnet  # Criticnet() -> Q-network

    q1, q2 = Criticnet(), Criticnet()
    with torch.no_grad():
        for q, value in ((q1, -3.0), (q2, -11.0)):
            for p in q.parameters():
                p.zero_()
            q.net[-1].bias.fill_(value)

    states = torch.tensor([[1., 0., 0.]])
    actions = torch.tensor([[0.4]])
    td_target = torch.tensor([-10.0])
    optimizer = torch.optim.SGD(
        list(q1.parameters()) + list(q2.parameters()), lr=0.1
    )
    print(update_critics(q1, q2, optimizer, states, actions, td_target))
