import numpy as np


def shape_reward(reward, next_state, velocity_weight):
    w = np.sum(np.array(next_state[6:8]) ** 2)
    reward = reward - velocity_weight * w
    return reward


reward = -0.2
next_state = np.array(
    [0, 0, 0, 0, 0, 0, 2.0, -1.0, 0, 0],
    dtype=np.float32,
)
velocity_weight = 0.01
shape_reward(reward, next_state, velocity_weight)
