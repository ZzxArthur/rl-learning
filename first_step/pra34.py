# 练习34：用 epsilon-greedy 在随机探索与贪心动作选择之间切换。
# 重复采样，观察动作频率与理论概率的对应关系。
import numpy as np

def choose_action(q_values, state, epsilon, rng):
    # 一种简单方法叫 epsilon-greedy（ε 贪心）：
    if rng.random() < epsilon:
        action_space = len(q_values[state])

        return int(rng.integers(action_space))  # rng.integers(n)：从 0 到 n-1 均匀随机选一个整数。
    else:
        return int(np.argmax(q_values[state]))  # np.argmax(values)：返回最大值的索引，并列时返回第一个。


q_values = {
    "A": [1.0, 1.8],
    "B": [2.0, -1.0],
    "C": [0.0, 0.0],
}
states = ["A", "B", "C"]
epsilon = 0.2
rng = np.random.default_rng(42)
a = []
for i in range(10000):
    a.append(choose_action(q_values, states[0], epsilon, rng))

print(a.count(1)/len(a))
