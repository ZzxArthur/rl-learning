# 练习33：用表格 Q-learning 更新状态—动作价值，未终止时取下一状态的最大 Q 值。
# 反复遍历给定转移，观察奖励经折扣向前传播及 Q 值逐渐稳定。
def q_learning_update(q_values, transitions, gamma, alpha):
    for obs in transitions:
        state, action, reward, next_state, terminated = obs
        if terminated:
            target_reward = reward
        else:
            target_reward = reward + gamma * max(q_values[next_state])
        q_values[state][action] += alpha * (target_reward - q_values[state][action])
    return q_values


q_values = {
    "A": [0.0, 0.0],
    "B": [0.0, 0.0],
}

transitions = [
    ("A", 0,  1.0, "END", True),
    ("A", 1,  0.0, "B",   False),
    ("B", 0,  2.0, "END", True),
    ("B", 1, -1.0, "END", True),
]

gamma = 0.9
alpha = 0.5
a = 0
for i in range(1000):
    q_values1 = q_learning_update(q_values, transitions, gamma, alpha)
    q_values = q_values1

    print(q_values1)
