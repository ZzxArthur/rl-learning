import torch


class QNetwork(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.fc1 = torch.nn.Linear(4, 64)
        self.relu = torch.nn.ReLU()
        self.fc2 = torch.nn.Linear(64, 2)

    def forward(self, states):
        f1 = self.fc1(states)
        r1 = self.relu(f1)
        f2 = self.fc2(r1)
        return f2
