# 分别用多个训练种子训练模型，再用同一组评估种子评估每个模型并记录结果。参数由你设计。
import torch
import gymnasium as gym
from pathlib import Path
from policy import PolicyNetwork
from config import TrainConfig
from train import train_policy
from evaluate import evalue_policy
from plot_rewards import plot_rewards


def run_experiments(train_seeds, evaluate_seeds, num_train, gamma, lr, env_id):
    rewards_by_seed = {}
    for train_seed in train_seeds:

        print(f"训练种子: {train_seed}")
        torch.manual_seed(train_seed)
        env = gym.make(env_id)
        policy_model = PolicyNetwork()
        policy_optimizer = torch.optim.SGD(policy_model.parameters(), lr=lr)

        rewards_by_seed[train_seed] = train_policy(
            env, policy_model, policy_optimizer, num_train, gamma, train_seed
        )
        env.close()
        env = gym.make(env_id)
        num_rewards = evalue_policy(env, policy_model, evaluate_seeds)
        print(num_rewards)
        env.close()
    return rewards_by_seed


if __name__ == "__main__":
    train_seeds = [42, 43, 44]
    eval_seeds = [100, 101, 102, 103, 104]

    config = TrainConfig()
    rewards_by_seed = run_experiments(
        train_seeds, eval_seeds, config.num_train, config.gamma, config.lr, config.env_id
    )
    plot_rewards(rewards_by_seed, Path(__file__).resolve().parent / 'experiment_rewards.png')
    
