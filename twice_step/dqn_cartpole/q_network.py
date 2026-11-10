import torch
import random


class QNetwork(torch.nn.Module):
    def __init__(self, states_num, actions_num, hidden_width):
        super().__init__()
        self.states_num = states_num
        self.actions_num = actions_num
        self.hidden_width = hidden_width
        self.fc1 = torch.nn.Linear(states_num, hidden_width)
        self.relu = torch.nn.ReLU()
        self.fc2 = torch.nn.Linear(hidden_width, actions_num)

    def forward(self, state):
        f1 = self.fc1(state)
        r1 = self.relu(f1)
        f2 = self.fc2(r1)
        return f2

    def select_action(self, state, epsilon):
        with torch.no_grad():
            q_values = self(state)  # 调用网络,相当于self.forward(state)
        if random.random() < epsilon:
            action = int(random.randrange(self.actions_num))
        else:
            action = int(torch.argmax(q_values))
        return action

