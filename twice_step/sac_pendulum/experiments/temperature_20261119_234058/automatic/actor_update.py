# SAC Actor 更新：用在线 Critic 的评分与熵项优化动作分布。
import torch

from updates import update_actor  # update_actor(actor, q1, q2, optimizer, states, alpha) -> actor_loss


if __name__ == "__main__":
    from actor import Actornet  # Actornet() -> actor
    from critic import Criticnet  # Criticnet() -> Q-network

    actor, q1, q2 = Actornet(), Criticnet(), Criticnet()
    with torch.no_grad():
        for net in (actor, q1, q2):
            for p in net.parameters():
                p.zero_()
        for q in (q1, q2):
            q.net[0].weight[0, 3] = 1.0  # 读取拼接输入的第 4 项，也就是动作
            q.net[-1].weight[0, 0] = 1.0
        q2.net[-1].bias.fill_(1.0)

    states = torch.tensor([[1., 0., 0.]])
    alpha = 0.2
    optimizer = torch.optim.SGD(actor.parameters(), lr=0.1)
    torch.manual_seed(42)
    print(update_actor(actor, q1, q2, optimizer, states, alpha))
