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


def evaluate_policy(policy, values, gamma, tol, max_iterations):
    while max_iterations > 0:
        new_values = np.zeros_like(values)
        
        for state in np.ndindex(values.shape):
            if state in walls or state == goal_state:
                continue
            action = policy[state]
            reward, next_state, terminated = step(np.array(state), action)
            next_row, next_col = next_state
            if terminated:
                target_reward = reward
            else:
                target_reward = reward + gamma * values[next_row, next_col]
            new_values[state] = target_reward
        delta = np.max(np.abs(new_values - values))
        values = new_values
        print(values)
        max_iterations -= 1
        if  delta < tol:
            break

    return values

rows, cols = 3, 3
start_state = (0, 0)
goal_state = (2, 2)
walls = {(1, 1)}

# 0上、1右、2下、3左
action_deltas = {
    0: (-1, 0),
    1: (0, 1),
    2: (1, 0),
    3: (0, -1),
}

policy = np.array([
    [0,  1,  2],
    [2, -1,  2],
    [1,  1, -1],
], dtype=np.int64)

values = np.zeros((3, 3), dtype=np.float64)
gamma = 0.9
tol = 1e-6
max_iterations = 200

evaluate_policy(policy, values, gamma, tol, max_iterations)