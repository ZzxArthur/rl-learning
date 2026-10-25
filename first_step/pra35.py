# 练习35：在字典环境中用 epsilon-greedy 选择动作，每步只更新当前转移的 Q 值。
# 连续训练 1000 局，每局从 A 出发、到达 END 后停止，跨局保留 Q 表。
# 训练后分别评估贪心与完全随机策略的平均原始奖励，评估不更新 Q 表。
import numpy as np


def choose_action(q_values, states, epsilon, rng):
    if rng.random() < epsilon:
        
        action_space = len(q_values[states])
        action_arg = int(rng.integers(action_space))
        return action_arg
    else:

        action_arg = int(np.argmax(q_values[states]))
        return action_arg


def q_learning(q_values, transitions, gamma, alpha):
    for k, v in transitions.items():
        state, action, reward, next_state, terminated = k[0], k[1], v[0], v[1], v[2]

        if terminated:
            target_reward = reward
        else:

            target_reward = reward + gamma * max(q_values[next_state])
        q_values[state][action] += alpha * (target_reward - q_values[state][action])
    return q_values


def evaluate(q_values, env_table, epsilon, episodes, rng):

    total_reward = 0
    for i in range(episodes):
        initial_state = 'A'
        if epsilon == 0:
            while True:
                action_arg = np.argmax(q_values[initial_state])
                reward, next_state, terminated = env_table[(initial_state, action_arg)]
                total_reward += reward
                initial_state = next_state
                if next_state == 'END' or terminated:
                    break
        else:
            while True:
                action_space = len(q_values[initial_state])
                action_arg = int(rng.integers(action_space))
                reward, next_state, terminated = env_table[(initial_state, action_arg)]
                total_reward += reward
                initial_state = next_state
                if next_state == 'END' or terminated:
                    break
    return total_reward / episodes
# 键：(当前状态, 动作)
# 值：(奖励, 下一状态, 是否真正终止)


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
rng = np.random.default_rng(22)
transitions = dict()
for i in range(1000):
    initial_state = 'A'
    while True:
        action = choose_action(q_values, initial_state, epsilon, rng)
        transitions = {(initial_state, action): env_table[(initial_state, action)]}
        q_values = q_learning(q_values, transitions, gamma, alpha)
        next_state = transitions[(initial_state, action)][1]
        terminated = transitions[(initial_state, action)][2]
        initial_state = next_state
        if terminated or next_state == 'END':
            break


episodes = 1000
epsilon = 0
mean_reward = evaluate(q_values, env_table, epsilon, episodes, rng)
print(mean_reward)
epsilon = 1
mean_reward = evaluate(q_values, env_table, epsilon, episodes, rng)
print(mean_reward)

# 俩层判断,外层负责循环,内层负责 取样动作 + 得到这一步发生的转移transitions + 更新q_values, 得到下一状态和terminated 更新状态 判断是否结束
