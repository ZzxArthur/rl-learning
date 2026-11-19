# SAC 经验回放：保存 Pendulum 转移，抽样时返回 Critic 和 TD 目标所需的张量。
from collections import deque
import random
import torch
import numpy as np


class ReplayBuffer():
    def __init__(self, capacity):
        self.buffer = deque(maxlen=capacity)

    def push(self, state, action, reward, next_state, terminated):
        self.buffer.append((
            np.asarray(state, dtype=np.float32),
            np.asarray(action, dtype=np.float32).reshape(1),
            float(reward),
            np.asarray(next_state, dtype=np.float32),
            float(terminated),
        ))

    def sample(self, batch_size):
        if not 1 <= batch_size <= len(self.buffer):
            raise ValueError("抽样数量必须在 1 和经验池当前条数之间")
        indices = random.sample(range(len(self.buffer)), batch_size)
        transitions = [self.buffer[i] for i in indices]
        states, actions, rewards, next_states, terminated = zip(*transitions)
        return (
            torch.as_tensor(np.stack(states)),
            torch.as_tensor(np.stack(actions)),
            torch.as_tensor(rewards, dtype=torch.float32),
            torch.as_tensor(np.stack(next_states)),
            torch.as_tensor(terminated, dtype=torch.float32),
        )
