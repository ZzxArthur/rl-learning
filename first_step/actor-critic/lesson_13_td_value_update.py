# 练习13：构造批次 TD(0) 目标，并更新一次神经网络状态价值。
import torch


def compute_td_data(value_model, value_optimizer, states, next_states, rewards, terminated, gamma):
    targets_value = []
    with torch.no_grad():
        next_states_value = value_model(next_states)
        print(next_states_value)
    for i in range(len(terminated)):
        if terminated[i]:
            target_value = rewards[i]
        else:
            target_value = rewards[i] + gamma * next_states_value[i].squeeze(-1)
        targets_value.append(target_value)
    targets_value = torch.stack(targets_value)
    states_value = value_model(states).squeeze(-1)
    value_loss = torch.mean((targets_value - states_value) ** 2)
    print('TD 目标：', targets_value)
    print('TD 误差：', (targets_value - states_value).detach())
    print('更新前 MSE：', value_loss.item())
    value_optimizer.zero_grad()
    value_loss.backward()
    with torch.no_grad():
        value_optimizer.step()
        value_optimizer.zero_grad()
    return states_value


if __name__ == "__main__":
    states = torch.tensor([
        [1.0, 0.0],
        [0.0, 1.0],
        [1.0, 0.0],
        [0.0, 1.0],
    ], dtype=torch.float32)

    next_states = torch.tensor([
        [0.0, 1.0],
        [1.0, 0.0],
        [0.0, 1.0],
        [1.0, 0.0],
    ], dtype=torch.float32)

    rewards = torch.tensor([2.0, 4.0, 1.0, 3.0], dtype=torch.float32)
    terminated = torch.tensor([False, True, False, True], dtype=torch.bool)
    gamma = 0.9

    value_model = torch.nn.Linear(2, 1)
    with torch.no_grad():
        value_model.weight.copy_(torch.tensor([[4.5, 3.5]]))
        value_model.bias.zero_()

    value_optimizer = torch.optim.SGD(value_model.parameters(), lr=0.05)
    compute_td_data(value_model, value_optimizer, states, next_states, rewards, terminated, gamma)
