# 练习43：交替评估和改进网格策略，直到策略稳定或达到迭代上限。
import numpy as np
from pra41 import evaluate_policy
from pra40 import extract_policy


def policy_iteration(initial_policy, gamma, tol,
                     max_eval_iterations, max_policy_iterations):

    rounds = 0
    zeros_values = np.zeros_like(initial_policy, dtype=np.float64)
    while True:
        new_values = evaluate_policy(initial_policy, zeros_values, gamma, tol, max_eval_iterations)
        new_policy = extract_policy(new_values, gamma)
        stable = np.array_equal(new_policy, initial_policy)
        rounds += 1
        if stable:
            return initial_policy, new_values, rounds, True
        elif rounds >= max_policy_iterations:
            return initial_policy, new_values, rounds, False
        initial_policy = new_policy




rows, cols = 3, 3
goal_state = (2, 2)
walls = {(1, 1)}

# 0 上、1 右、2 下、3 左
action_deltas = {
    0: (-1, 0),
    1: (0, 1),
    2: (1, 0),
    3: (0, -1),
}

initial_policy = np.array([
    [0,  1,  2],
    [2, -1,  2],
    [1,  1, -1],
], dtype=np.int64)

gamma = 0.9
tol = 1e-6
max_eval_iterations = 200    # 每次价值评估的轮数上限
max_policy_iterations = 20  # 策略评估与改进的轮数上限


print(policy_iteration(initial_policy, gamma, tol, max_eval_iterations, max_policy_iterations))
