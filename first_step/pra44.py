# 练习44：沿固定策略采样网格转移，用 TD(0) 估计访问过的状态价值。
from pra41 import step
import numpy as np


def td_update(values, transitions, gamma, alpha):  # transitions = [(state, reward, next_state, terminated)]
    for i in transitions:
        state, reward, next_state, terminated = i
        if terminated:
            target = reward
        else:
            target = reward + gamma * values[next_state]
        values[state] += alpha * (target - values[state])
    return values


def learn_values(policy, values, start_state, gamma, alpha, num_episodes, max_steps):
    for _ in range(num_episodes):
        state = start_state
        for _ in range(max_steps):
            action_tag = policy[state]
            if action_tag == -1:
                
                break
            reward, next_state, terminated = step(np.array(state), action_tag)
            row, col = next_state
            next_state = (row, col)

            transitions = [(state, reward, next_state, terminated)]
            values = td_update(values, transitions, gamma, alpha)
            state = next_state
            if terminated:
                break
    return values


rows, cols = 3, 3
start_state = (0, 0)
goal_state = (2, 2)
walls = {(1, 1)}

# 0 上、1 右、2 下、3 左
action_deltas = {
    0: (-1, 0),
    1: (0, 1),
    2: (1, 0),
    3: (0, -1),
}

policy = np.array([
    [1,  1,  2],
    [2, -1,  2],
    [1,  1, -1],
], dtype=np.int64)

# 字典的键是坐标元组，例如 (0, 0)，值是浮点状态价值
values = {(row, col): 0.0 for row in range(rows) for col in range(cols)}
print(values[start_state])
gamma = 0.9
alpha = 0.5
num_episodes = 1000
max_steps = 50

print(learn_values(policy, values, start_state, gamma, alpha, num_episodes, max_steps))
