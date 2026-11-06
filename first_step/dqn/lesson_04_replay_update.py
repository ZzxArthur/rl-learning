import torch
from lesson_01_replay_buffer import ReplayBuffer  # ReplayBuffer(capacity) 创建回放池
from lesson_02_q_network import QNetwork  # QNetwork() 创建 Q 网络


transitions = [
    ([0.0, 0.1, 0.2, 0.0], 0, 1.0, [0.1, 0.0, 0.2, 0.1], False),
    ([0.1, 0.0, 0.2, 0.1], 1, 1.0, [0.2, 0.1, 0.1, 0.0], False),
    ([0.2, 0.1, 0.1, 0.0], 0, 1.0, [0.3, 0.0, 0.0, 0.1], True),
    ([0.3, 0.0, 0.0, 0.1], 1, 1.0, [0.2, 0.0, 0.1, 0.2], False),
]

gamma = 0.9
capacity = 4
batch_size = 2
replaybuffer = ReplayBuffer(capacity)
for transition in transitions:
    state, action, reward, next_state, terminated = transition
    replaybuffer.push(state, action, reward, next_state, terminated)

sampled = replaybuffer.sample(batch_size)
states_list, actions_list, rewards_list, next_states_list, terminated_list = zip(*sampled)
states = torch.tensor(states_list, dtype=torch.float32)
actions = torch.tensor(actions_list, dtype=torch.long)
rewards = torch.tensor(rewards_list, dtype=torch.float32)
next_states = torch.tensor(next_states_list, dtype=torch.float32)
terminated = torch.tensor(terminated_list, dtype=torch.bool)

train_net = QNetwork()
target_net = QNetwork()
train_parameters = train_net.state_dict()
target_net.load_state_dict(train_parameters)
optimizer = torch.optim.SGD(train_net.parameters(), lr=0.1)


with torch.no_grad():
    next_states_q = target_net(next_states)       # [2, 2]
    max_next_q = next_states_q.max(dim=1).values  # [2]
    td_target = rewards + gamma * max_next_q * (~terminated)  # [2]
print(td_target)

q = train_net(states)
selected_q = q.gather(dim=1, index=actions.unsqueeze(1)).squeeze(1)
print(selected_q)
mse_errors = torch.nn.functional.mse_loss(selected_q, td_target)
mse_errors.backward()
with torch.no_grad():
    optimizer.step()
    optimizer.zero_grad()
