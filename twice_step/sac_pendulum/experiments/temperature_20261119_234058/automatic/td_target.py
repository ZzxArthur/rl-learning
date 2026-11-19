# SAC TD 目标：用两套目标 Critic 的较小估计与下一动作的熵项做自举。
import torch

from updates import computer_td_target  # computer_td_target(actor, target_q1, target_q2, next_states, rewards, terminated, gamma, alpha) -> y


if __name__ == "__main__":
    from actor import Actornet  # Actornet() -> actor
    from critic import Criticnet  # Criticnet() -> Q-network

    actor = Actornet()
    target_q1, target_q2 = Criticnet(), Criticnet()

    with torch.no_grad():
        for model in (actor, target_q1, target_q2):
            for p in model.parameters():
                p.zero_()
        target_q1.net[-1].bias.fill_(-3.0)
        target_q2.net[-1].bias.fill_(-4.0)

    next_states = torch.tensor([[1., 0., 0.], [0., 1., 0.5]])
    rewards = torch.tensor([-1., -2.])
    terminated = torch.tensor([0., 1.])
    gamma = 0.9
    alpha = 0.2

    torch.manual_seed(42)
    td_target = computer_td_target(actor, target_q1, target_q2, next_states, rewards, terminated, gamma, alpha)
    print(td_target)
