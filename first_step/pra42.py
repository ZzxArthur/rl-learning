# 练习42：评估固定策略，按价值表改进一次策略，再评估改进后的策略。
from pra41 import step, evaluate_policy
from pra40 import extract_policy
import numpy as np



rows, cols = 3, 3
goal_state = (2, 2)
walls = {(1, 1)}
action_deltas = {
    0: (-1, 0),  # 上
    1: (0, 1),   # 右
    2: (1, 0),   # 下
    3: (0, -1),  # 左
}

old_policy = np.array([
    [0,  1,  2],
    [2, -1,  2],
    [1,  1, -1],
], dtype=np.int64)

gamma = 0.9
tol = 1e-6
max_iterations = 200


initial_values = np.zeros_like(old_policy, dtype=np.float64)


old_values = evaluate_policy(old_policy, initial_values, gamma, tol, max_iterations)
print("旧策略价值表：\n", old_values)
new_policy = extract_policy(old_values, gamma)
print("改进后的策略：\n", new_policy)

new_values = evaluate_policy(new_policy, np.zeros_like(initial_values), gamma, tol, max_iterations)
print("新策略价值表：\n", new_values)
