# 练习37：在字典环境中训练多局 SARSA，下一动作既用于更新，也用于下一步执行。
# 每局重置状态并保留 Q 表，真正终止时只用即时奖励更新。
# 提供按指定 epsilon 评估平均原始奖励的函数，当前主程序打印训练后的 Q 表。
import numpy as np


def choose_action(q_values, state, epsilon, rng):
    if rng.random() < epsilon:
        action_space = len(q_values[state])
        action = int(rng.integers(action_space))
    else:
        action = int(np.argmax(q_values[state]))
    return action


def sarsa_update(q_values, state, action, reward, next_state,
                 next_action, terminated, gamma, alpha):
    target_reward = 0
    if terminated:
        target_reward = reward
    else:
        target_reward = reward + gamma * q_values[next_state][next_action]
    q_values[state][action] += alpha * (target_reward - q_values[state][action])
    return q_values


def evaluate(q_values, env_table, epsilon, num_episodes, rng):

    sum_reward = 0
    for _ in range(num_episodes):
        now_state = 'A'
        while True:
            action = choose_action(q_values, now_state, epsilon, rng)
            reward, next_state, terminated = env_table[now_state, action]
            sum_reward += reward
            if terminated:
                break
            now_state = next_state

    return sum_reward / num_episodes


env_table = {
    ("A", 0): (1.0, "END", True),
    ("A", 1): (0.0, "B", False),
    ("B", 0): (2.0, "END", True),
    ("B", 1): (-1.0, "END", True),
}

q_values = {
    "A": [0.0, 0.0],
    "B": [0.0, 0.0],
}
gamma = 0.9
alpha = 0.5
epsilon = 0.5
num_episodes = 1000
rng = np.random.default_rng(44)
for i in range(num_episodes):
    state = 'A'
    action = choose_action(q_values, state, epsilon, rng)
    while True:
        reward, next_state, terminated = env_table[state, action]

        if terminated:
            next_action = None
            q_values = sarsa_update(q_values, state, action, reward,
                                    next_state, next_action, terminated, gamma, alpha)

            break
        else:
            next_action = choose_action(q_values, next_state, epsilon, rng)
            q_values = sarsa_update(q_values, state, action, reward,
                                    next_state, next_action, terminated, gamma, alpha)

            state = next_state
            action = next_action
print(q_values)
# print(q_values)
# epsilon = 1
# print(evaluate(q_values, env_table, epsilon, num_episodes, rng))
# epsilon = 0
# print(evaluate(q_values, env_table, epsilon, num_episodes, rng))

