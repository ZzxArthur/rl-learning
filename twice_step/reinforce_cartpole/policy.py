# CartPole 随机策略的网络：输入状态，输出两个动作的 logits。
import torch


class PolicyNetwork(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.layer = torch.nn.Linear(4, 2)

    def forward(self, states):
        logits = self.layer(states)

        return logits


if __name__ == "__main__":
    torch.manual_seed(42)
    states = torch.tensor(
        [
            [0.1, 0.0, 0.2, 0.0],
            [-0.1, 0.1, -0.2, 0.0],
            [0.0, 0.0, 0.0, 0.0],
        ],
        dtype=torch.float32,
    )
    model = PolicyNetwork()
    logits = model(states)
    print("logits:", logits)
    print("logits shape:", logits.shape)
    for name, parameter in model.named_parameters():
        print(name, parameter.shape)
