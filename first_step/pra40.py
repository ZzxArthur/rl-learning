# 练习40：从网格状态价值表提取贪心动作策略，并评估一局的奖励、步数和路径。
import numpy as np


def step(state, action_tag):
    action = action_deltas[action_tag]

    if np.array_equal(state, np.array([2, 2])):
        next_state = state
        reward = 0
        terminated = True
    else:
        next_state = state + action

        if np.array_equal(next_state, (2, 2)):
            reward = 10
            terminated = True

        else:
            reward = -1
            terminated = False
        limits = np.array([rows, cols])
        out_of_bounds = ((next_state < 0).any()
                         or (next_state >= limits).any())

        if out_of_bounds or np.array_equal(next_state, (1, 1)):
            next_state = state
    return reward, next_state, terminated   # 返回 reward, next_state, terminated


def extract_policy(values, gamma):
    action_table = np.full(values.shape, -1, dtype=int)
    for state in np.ndindex(values.shape):

        action_max_tag = -1
        action__max_reward = float("-inf")  # 负无穷
        if state in walls or state == goal_state:
            continue
        for action_tag in action_deltas.keys():
            reward, next_state, terminated = step(np.array(state), action_tag)
            if terminated:
                action_reward = reward
            else:
                action_reward = reward + gamma * values[tuple(next_state)]

            if action_reward > action__max_reward:
                action__max_reward = action_reward
                action_max_tag = action_tag
        action_table[state] = action_max_tag
    return action_table


def evaluate_policy(policy, start_state, max_steps):  # policy = action_table
    start_state = np.array(start_state)

    trajectory = [tuple(start_state)]
    total_reward = 0
    steps = 0
    terminated = False
    for _ in range(max_steps):
        action_tag = policy[tuple(start_state)]

        action = action_deltas[action_tag]
        reward, next_state, terminated = step(start_state, action_tag)
        trajectory.append(tuple(next_state))
        total_reward += reward
        start_state = next_state
        steps += 1
        if terminated:
            break
    return total_reward, steps, terminated, trajectory


rows = 3
cols = 3
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

values = np.array([
    [4.58, 6.2,  8.0],
    [6.2,  0.0, 10.0],
    [8.0, 10.0,  0.0],
], dtype=np.float64)

gamma = 0.9
max_steps = 50

policy = extract_policy(values, gamma)
print(policy)
print(evaluate_policy(policy, start_state, max_steps))
