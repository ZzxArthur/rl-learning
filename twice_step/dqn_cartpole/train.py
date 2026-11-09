import random
import gymnasium as gym
import torch

from pathlib import Path
from agent import DQNAgent  # DQNAgent(gamma, lr, batch_size, target_update_interval) -> agent
from replay_buffer import ReplayBuffer  # ReplayBuffer(capacity) -> replay buffer

seed = 42
env_id = "CartPole-v1"
num_episodes = 500
capacity = 2000
gamma = 0.9
lr = 0.02
target_update_interval = 100
initial_epsilon = 0.5
lowest_epsilon = 0.05
epsilon_decay_steps = 20_000
batch_size = 64

torch.manual_seed(seed)
random.seed(seed)
env = gym.make(env_id)
env_step = 0
agent = DQNAgent(gamma, lr, batch_size, target_update_interval)
replay_buffer = ReplayBuffer(capacity)
low_step = (initial_epsilon - lowest_epsilon) / epsilon_decay_steps
for step in range(1, 1+num_episodes):
    state, _ = env.reset(seed=seed if step == 1 else None)
    episode_reward = 0.0
    while True:
        epsilon = max(lowest_epsilon, initial_epsilon - low_step * env_step)
        state = torch.tensor(state, dtype=torch.float32)
        action = agent.select_action(state, epsilon)
        next_state, reward, terminated, truncated, _ = env.step(action)
        env_step += 1
        episode_reward += reward
        replay_buffer.push(state, action, reward, next_state, terminated)
        agent.update(replay_buffer)
        if terminated or truncated:
            break
        state = next_state
    print(f"局数{step}, 奖励{episode_reward}, epsilon:{epsilon}, 累计环境步数{env_step}")
env.close()
model_path = Path(__file__).resolve().parent / "dqn_weights.pt"
agent.save(model_path)
print(f"模型参数已保存: {model_path}")
