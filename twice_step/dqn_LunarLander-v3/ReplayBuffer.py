from collections import deque
import random


class ReplayBuffer():
    def __init__(self, capacity):
        self.buffer = deque(maxlen=capacity)

    def push(self, state, action, reward, next_state, terminated):
        tranisition = (state, action, reward, next_state, terminated)
        self.buffer.append(tranisition)  # append只能接受一个对象, 所以要把5个组成一条元组再存
        # TD 目标只用 terminated 屏蔽后续价值；时间限制截断时仍保留 bootstrap。

    def sample(self, batch_size):
        if len(self.buffer) >= batch_size:
            transitions = random.sample(list(self.buffer), batch_size)
            return transitions
        else:
            raise ValueError("经验池中容量不够")
