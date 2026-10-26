# 练习38：用 NumPy 坐标和整数动作编号实现网格环境的一步转移，函数内查询动作位移。
# 越界或撞墙时原地停留；到达目标终止，终止后不重复发放奖励。
# 用 NumPy Q 表训练多局 Q-learning，并贪心评估总奖励、步数、是否到达目标及路径。
import numpy as np


def choose_action(q_values, state, epsilon, rng):
    row, col = state
    if rng.random() < epsilon:
        action_space = len(q_values[row][col])
        action = int(rng.integers(action_space))
    else:
        action = np.argmax(q_values[row][col])
    return action


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


def q_learning_update(q_values, state, action, reward,
                      next_state, terminated, gamma, alpha):
    row, col = state
    next_row, next_col = next_state
    if terminated:
        target_reward = reward
    else:
        target_reward = reward + gamma * np.max(q_values[next_row][next_col])
    q_values[row, col, action] += alpha * (target_reward - q_values[row, col, action])
    return q_values


def evaluate(q_values, start_state, max_steps, eval_rng):
    eval_reward = 0
    eval_steps = 0
    terminated = False
    eval_trajectory = [tuple(start_state)]
    state = np.array(start_state)
    epsilon = 0
    for _ in range(max_steps):
        action = choose_action(q_values, state, epsilon, eval_rng)
        reward, next_state, terminated = step(state, action)
        eval_reward += reward
        eval_steps += 1
        eval_trajectory.append(tuple(next_state))
        if terminated:
            break
        state = next_state
    return eval_reward, eval_steps, terminated, eval_trajectory


rows = 3
cols = 3

start_state = (0, 0)
goal_state = (2, 2)
walls = {(1, 1)}

# 动作：0上、1右、2下、3左
action_deltas = {
    0: (-1, 0),
    1: (0, 1),
    2: (1, 0),
    3: (0, -1),
}

q_values = np.zeros((rows, cols, 4), dtype=np.float64)

gamma = 0.9
alpha = 0.5
epsilon = 0.2
num_episodes = 1000
max_steps = 50
rng = np.random.default_rng(42)

training_rewards = []


for _ in range(num_episodes):
    state = np.array(start_state)

    total_reward = 0
    for _ in range(max_steps):
        action = choose_action(q_values, state,
                               epsilon, rng)
        reward, next_state, terminated = step(state, action)
        q_values = q_learning_update(q_values, state, action, reward,
                                     next_state, terminated, gamma, alpha)
        state = next_state
        total_reward += reward

        if terminated:
            break
    training_rewards.append(total_reward)

print(q_values[0][0])
print(sum(training_rewards[-10:]))

eval_rng = np.random.default_rng(123)
eval_reward, eval_steps, success, eval_trajectory = evaluate(q_values, start_state, max_steps, eval_rng)
print(eval_reward, eval_steps, success, eval_trajectory)
