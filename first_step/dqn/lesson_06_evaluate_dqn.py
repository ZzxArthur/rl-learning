from pathlib import Path

import gymnasium as gym
import torch

from lesson_02_q_network import QNetwork  # QNetwork() 创建一个 Q 网络


def main():
    model_path = Path(__file__).resolve().parent / 'dqn_weights.pt'
    if not model_path.exists():
        raise FileNotFoundError(f'找不到模型参数 {model_path}，请先运行 lesson_05_cartpole_dqn.py 完成训练。')

    model = QNetwork()
    model.load_state_dict(torch.load(model_path, map_location='cpu', weights_only=True))
    model.eval()
    eval_seeds = [100, 101, 102, 103, 104]
    rewards = []
    env = gym.make('CartPole-v1')
    try:
        for seed in eval_seeds:
            state, _ = env.reset(seed=seed)
            total_reward = 0.0
            while True:
                state_tensor = torch.as_tensor(state, dtype=torch.float32).unsqueeze(0)
                action = model.select_action(state_tensor, epsilon=0)
                state, reward, terminated, truncated, _ = env.step(action)
                total_reward += reward
                if terminated or truncated:
                    break
            rewards.append(total_reward)
            print(f'评估种子: {seed}, 总奖励: {total_reward:.1f}')
    finally:
        env.close()
    print(f'{len(rewards)} 局平均奖励: {sum(rewards) / len(rewards):.1f}')


if __name__ == '__main__':
    main()
