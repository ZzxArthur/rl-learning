# CartPole 随机策略的网络：输入状态，输出两个动作的 logits。
import torch


class PolicyNetwork(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.fc1 = torch.nn.Linear(4, 64)
        self.activation = torch.nn.Tanh()
        self.fc2 = torch.nn.Linear(64, 2)

    def forward(self, states):
        f1 = self.fc1(states)
        a1 = self.activation(f1)
        f2 = self.fc2(a1)
        return f2


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
