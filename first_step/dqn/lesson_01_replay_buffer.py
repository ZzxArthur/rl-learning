from collections import deque
import random


class ReplayBuffer():
    def __init__(self, capacity):
        self.buffer = deque(maxlen=capacity)

    def push(self, state, action, reward, next_state, terminated):
        tranisition = (state, action, reward, next_state, terminated)
        self.buffer.append(tranisition)

    def sample(self, batch_size):
        if batch_size > len(self.buffer):
            raise ValueError("经验池中的数据不足")
        else:
            transitions = random.sample(list(self.buffer), batch_size)
        return transitions