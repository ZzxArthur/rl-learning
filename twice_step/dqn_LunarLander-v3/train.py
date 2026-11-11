import torch
import random
import gymnasium as gym
from pathlib import Path
from Agent import DqnAgent
from ReplayBuffer import ReplayBuffer


seed = 42
update_interval = 50
capacity = 20000
num_episode = 500
batch_size = 128
epsilon = 0.8
lowest_epsilon = 0.01
epsilon_step = 60000
gamma = 0.99
lr = 0.001

env_id = "LunarLander-v3"
if __name__ == "__main__":
    random.seed(seed)
    torch.manual_seed(seed)

    env = gym.make(env_id)
    agent = DqnAgent(update_interval, batch_size, gamma, lr)
    replaybuffer = ReplayBuffer(capacity)

    step = 0
    critical_value_episilon = (epsilon - lowest_epsilon) / epsilon_step
    for episode in range(1, num_episode+1):
        state, _ = env.reset(seed=seed if episode == 1 else None)
        num_rewards = 0
        episode_step = 0
        while True:
            select_epsilon = max(lowest_epsilon, epsilon - critical_value_episilon * step)
            with torch.no_grad():
                action = agent.select_action(torch.as_tensor(state, dtype=torch.float32), select_epsilon)
            next_state, reward, terminated, truncated, _ = env.step(action)
            num_rewards += reward
            step += 1
            episode_step += 1
            replaybuffer.push(state, action, reward, next_state, terminated)
            agent.update(replaybuffer)
            if terminated or truncated:
                break
            state = next_state
        print(f"局数{episode}, 总奖励{num_rewards:.2f}, epsilon:{select_epsilon:.4f}, 累计环境步数{step}, 一局累计步数{episode_step}")

    agent.save(str(Path(__file__).with_name("dqn_checkpoint.pt")))
    env.close()



