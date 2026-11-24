# 练习17：为完整轨迹计算 n 步 TD 目标，终止后不再 bootstrap。
import torch


def computer_n_steps_targets(value_model, n, rewards, next_states, terminated, gamma):
    targets_reward_list = []
    for i in range(len(terminated)):
        if terminated[i]:
            target = rewards[i]
        else:
            target = 0
            for j in range(n):
                step = i + j
                target = target + (gamma ** j) * rewards[step]
                if terminated[step]:
                    # 当前奖励已经加过；终止后不再加价值预测
                    break
                if j == n - 1:
                    # 已经走满 n 步，且当前步没有终止
                    with torch.no_grad():
                        next_value = value_model(next_states[step]).squeeze()
                    target = target + (gamma ** n) * next_value
        # 内层结束后，当前起点只有一个目标
        targets_reward_list.append(target)
    targets_reward = torch.stack(targets_reward_list, dim=0)
    return targets_reward


if __name__ == "__main__":
    torch.manual_seed(42)
    # s0 -> s1 -> s2 -> terminal
    next_states = torch.tensor(
        [
            [0.0, 1.0, 0.0, 0.0],  # s1
            [0.0, 0.0, 1.0, 0.0],  # s2
            [0.0, 0.0, 0.0, 1.0],  # terminal
        ],
        dtype=torch.float32,
    )

    rewards = torch.tensor([1.0, 2.0, 3.0], dtype=torch.float32)
    terminated = torch.tensor([False, False, True], dtype=torch.bool)

    gamma = 0.9
    for n in range(1, 4):
        # 对四个状态分别预测 6、5、4、0
        value_model = torch.nn.Linear(4, 1, bias=False)
        with torch.no_grad():
            value_model.weight.copy_(torch.tensor([[6.0, 5.0, 4.0, 0.0]]))
        print(n, computer_n_steps_targets(value_model, n, rewards, next_states, terminated, gamma))



