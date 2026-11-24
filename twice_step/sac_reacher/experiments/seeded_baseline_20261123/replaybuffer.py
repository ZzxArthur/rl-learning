from collections import deque
import numpy as np
import torch


class ReplayBuffer():
    def __init__(self, capacity):
        self.buffer = deque(maxlen=capacity)

    def __len__(self):
        return len(self.buffer)

    def push(self, state, action, reward, next_state, terminated):
        self.buffer.append((
            np.asarray(state, dtype=np.float32),
            np.asarray(action, dtype=np.float32),
            float(reward),
            np.asarray(next_state, dtype=np.float32),
            float(terminated)
        ))

    def get_transitions(self, batch_size):
        if batch_size > len(self.buffer):
            raise ValueError("经验池数量不够")

        indices = np.random.choice(len(self.buffer), size=batch_size, replace=False)
        transitions = [self.buffer[i] for i in indices]
        states, actions, rewards, next_states, terminated = zip(*transitions)
        return (
            torch.as_tensor(np.array(states), dtype=torch.float32),
            torch.as_tensor(np.array(actions), dtype=torch.float32),
            torch.as_tensor(rewards, dtype=torch.float32).unsqueeze(-1),
            torch.as_tensor(np.array(next_states), dtype=torch.float32),
            torch.as_tensor(terminated, dtype=torch.float32).unsqueeze(-1),
        )

