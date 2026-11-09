import gymnasium as gym
import numpy as np
import torch
from pathlib import Path
import random

from lesson_01_replay_buffer import ReplayBuffer  # ReplayBuffer(capacity) 创建回放池
from lesson_02_q_network import QNetwork  # QNetwork() 创建 Q 网络


seed = 42
num_episodes = 500
capacity = 20000
lr = 0.02
initial_epsilon = 0.5
lowest_epsilon = 0.05
batch_size = 256

target_update_interval = 100
gamma = 0.9
random.seed(seed)
torch.manual_seed(seed)
env = gym.make('CartPole-v1')
train_net = QNetwork()
target_net = QNetwork()
train_net_parameters = train_net.state_dict()
target_net.load_state_dict(train_net_parameters)
replaybuffer = ReplayBuffer(capacity)
train_optimizer = torch.optim.SGD(train_net.parameters(), lr=lr)
step = 0
env_steps = 0
low_step = (initial_epsilon - lowest_epsilon) / 20000
for episode in range(1, num_episodes + 1):
    state, info = env.reset(seed=seed if episode == 1 else None)
    episode_reward = 0.0
    while True:
        epsilon = max(lowest_epsilon, initial_epsilon - low_step * env_steps)
        state = torch.tensor(state, dtype=torch.float32)
        action = train_net.select_action(state, epsilon)
        next_state, reward, terminated, truncated, info = env.step(action)
        env_steps += 1
        episode_reward += reward
        replaybuffer.push(state, action, reward, next_state, terminated)

        if len(replaybuffer.buffer) >= batch_size:
            tranisitions = replaybuffer.sample(batch_size)
            states_list, actions_list, rewards_list, next_states_list, terminated_list = zip(*tranisitions)

            states = torch.stack(states_list, dim=0)
            actions = torch.tensor(actions_list, dtype=torch.long)
            rewards = torch.tensor(rewards_list, dtype=torch.float32)
            next_states = torch.as_tensor(np.stack(next_states_list), dtype=torch.float32)
            terminated_batch = torch.tensor(terminated_list, dtype=torch.bool)

            with torch.no_grad():
                next_states_q = target_net(next_states)       # [2, 2]
                max_next_q = next_states_q.max(dim=1).values  # [2]
                td_target = rewards + gamma * max_next_q * (~terminated_batch)  # [2]
            q = train_net(states)
            selected_q = q.gather(dim=1, index=actions.unsqueeze(1)).squeeze(1)

            mse_errors = torch.nn.functional.mse_loss(selected_q, td_target)
            mse_errors.backward()
            with torch.no_grad():
                train_optimizer.step()
                train_optimizer.zero_grad()
            step += 1
            if step % target_update_interval == 0:
                train_net_parameters = train_net.state_dict()
                target_net.load_state_dict(train_net_parameters)
        if terminated or truncated:
            break
        state = next_state
    print(f"局数: {episode}, 随机探索程度:{epsilon:.4f}, 本局总奖励: {episode_reward:.1f}, 累计环境步数: {env_steps}, 累计更新次数: {step}")

env.close()
model_path = Path(__file__).resolve().parent / 'dqn_weights.pt'
torch.save(train_net.state_dict(), model_path)
print(f"模型参数已保存: {model_path}")
