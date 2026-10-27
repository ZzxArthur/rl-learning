# 练习39：利用已知网格转移规则进行价值迭代，按价值变化阈值或迭代上限停止。
# 每轮统一读取旧表并写入新表，返回价值表、实际轮数和最后的最大变化。
import numpy as np


def step(state, action):
    action = action_deltas[action]  # action_tag = 0, 1, 2, 3 action = (-1, 0)...代表坐标
    if np.array_equal(state, np.array([2, 2])):
        next_state = state
        reward = 0
        terminated = True
    else:
        next_state = state + action

        if not np.array_equal(next_state, np.array([2, 2])):
            reward = -1
            terminated = False
        else:
            reward = 10
            terminated = True
        limits = np.array([rows, cols])
        out_of_bounds = (
            (next_state < 0).any()
            or (next_state >= limits).any()
        )
        if out_of_bounds or np.array_equal(next_state, np.array([1, 1])):

            next_state = state
    return reward, next_state, terminated   # 返回 reward, next_state, terminated


def value_iteration(values, gamma, tol, max_iterations):
    for iteration in range(max_iterations):
        new_values = np.zeros_like(values)
        for state in np.ndindex(values.shape):
            if state in walls or state == goal_state:
                continue
            reward_list = []

            for action_tag in range(len(action_deltas)):
                reward, next_state, terminated = step(np.array(state), action_tag)
                next_row, next_col = next_state

                if terminated:
                    reward_list.append(reward)
                else:
                    reward_list.append(reward + gamma * values[next_row, next_col])

            new_values[state] = max(reward_list)
        delta = np.max(np.abs(new_values - values))
        values = new_values
        if delta < tol:
            break
    return values, iteration + 1, delta













rows = 3
cols = 3
start_state = (0, 0)
goal_state = (2, 2)
walls = {(1, 1)}

action_deltas = {
    0: (-1, 0),
    1: (0, 1),
    2: (1, 0),
    3: (0, -1),
}

values = np.zeros((rows, cols), dtype=np.float64)
gamma = 0.9
tol = 1e-6
max_iterations = 100
values, iterations, delta = value_iteration(values, gamma, tol, max_iterations)
print(values)
print("实际迭代轮数：", iterations)
print("最后的最大变化：", delta)
