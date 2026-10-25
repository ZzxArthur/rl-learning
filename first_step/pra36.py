# 练习36：对单条转移执行 SARSA 更新，未终止时使用指定下一动作的 Q 值。
# 真正终止时目标仅为即时奖励；对比不同下一动作如何影响当前 Q 值。
def sarsa_update(q_values, state, action, reward,
                 next_state, next_action, terminated, gamma, alpha):
    target_reward = 0
    if terminated:
        target_reward = reward
    else:
        target_reward = reward + gamma * q_values[next_state][next_action]
    q_values[state][action] += alpha * (target_reward - q_values[state][action])
    return q_values


q_values = {
    "A": [0.0, 0.0],
    "B": [2.0, -1.0],
}
state = "A"
action = 1
reward = 0.0
next_state = "B"
next_action = 1
terminated = False
gamma = 0.9
alpha = 0.5

print(sarsa_update(q_values, state, action, reward, next_state, next_action, terminated, gamma, alpha))
