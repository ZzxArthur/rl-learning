import torch
from lesson_02_q_network import QNetwork  # QNetwork() 创建一个 Q 网络


train_net = QNetwork()  # 训练网络
target_net = QNetwork()  # 目标网络

train_parameters = train_net.state_dict()  # 取出训练网络的参数,将其赋值给目标网络
target_net.load_state_dict(train_parameters)
optimizer = torch.optim.SGD(train_net.parameters(), lr=0.1)

state = torch.tensor([
    [0, 1, 2, 3],
    [-1, 2, 3, 1]
], dtype=torch.float32)

actions = torch.tensor([1, 0])

next_states = torch.tensor([
    [0.1, 1.1, 1.9, 2.9],
    [-0.9, 1.9, 2.9, 1.1],
], dtype=torch.float32)

rewards = torch.tensor([1.0, 1.0])
terminated = torch.tensor([False, True])
gamma = 0.9

with torch.no_grad():
    next_states_q = target_net(next_states)       # [2, 2]
    max_next_q = next_states_q.max(dim=1).values  # [2]
    td_target = rewards + gamma * max_next_q * (~terminated)  # [2]
print(td_target)

q = train_net(state)
selected_q = q.gather(dim=1, index=actions.unsqueeze(1)).squeeze(1)
print(selected_q)
mse_errors = torch.nn.functional.mse_loss(selected_q, td_target)
mse_errors.backward()
with torch.no_grad():
    optimizer.step()
    optimizer.zero_grad()
same = all(
    torch.equal(train_param, target_param)
    for train_param, target_param in zip(
        train_net.parameters(), target_net.parameters()
    )
)
print("同步前：", same)
target_net.load_state_dict(train_net.state_dict())
same = all(
    torch.equal(train_param, target_param)
    for train_param, target_param in zip(
        train_net.parameters(), target_net.parameters()
    )
)
print("同步前：", same)
