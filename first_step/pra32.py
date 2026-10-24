# 练习32：用表格 TD(0) 更新状态价值，真正终止时后续价值取 0。
import torch


def td_update(values, transitions, gamma, alpha):
    for i in transitions:
        state, reward, next_state, terminated = i
        if terminated:
            target = reward
        else:
            target = reward + gamma * values[next_state]
        values[state] += alpha * (target - values[state])
    return values


values = {"A": 0.0, "B": 0.0}

transitions = [
    ("A", 1.0, "B", False),
    ("B", 2.0, "END", True),
]

gamma = 0.9
alpha = 0.5
values1 = td_update(values, transitions, gamma, alpha)

print(values1)


