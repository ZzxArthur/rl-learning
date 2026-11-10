import torch
import random


class Q_network(torch.nn.Module):
    def __init__(self, states_num=8, actions_num=4, hidden_width=[128, 256]):
        super().__init__()  # super().__init__()而不是super.__init__()
        self.states_num = states_num
        self.actions_num = actions_num
        self.hidden_width = hidden_width
        self.network()  # 记得初始化网络层

    def network(self):
        self.fc1 = torch.nn.Linear(self.states_num, self.hidden_width[0])
        self.relu1 = torch.nn.ReLU()
        self.fc2 = torch.nn.Linear(self.hidden_width[0], self.hidden_width[1])
        self.relu2 = torch.nn.ReLU()
        self.fc3 = torch.nn.Linear(self.hidden_width[1], self.actions_num)

    def forward(self, state):
        f1 = self.fc1(state)
        r1 = self.relu1(f1)
        f2 = self.fc2(r1)
        r2 = self.relu2(f2)
        f3 = self.fc3(r2)
        return f3

    def select_action(self, state, epsilion):
        q_value = self(state)
        if random.random() >= epsilion:
            action = int(q_value.argmax().item())
        else:
            action = int(random.randrange(self.actions_num))
        return action

