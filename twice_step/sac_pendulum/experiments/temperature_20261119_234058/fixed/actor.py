# SAC Actor 教学示例，网络实现见 networks.py。
import torch

from networks import Actornet  # Actornet(**network_config) -> actor


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
