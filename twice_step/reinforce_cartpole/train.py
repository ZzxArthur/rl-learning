# 将一局轨迹转换为策略梯度损失，并更新策略模型。
import torch
import gymnasium as gym

from pathlib import Path
from policy import PolicyNetwork
from returns import compute_returns  # compute_returns(trajectory, gamma) -> returns
from rollout import collect_trajectory  # collect_trajectory(env, policy_model, seed) -> trajectory
from checkpoint import save_model_weights
from config import TrainConfig
from plot_rewards import plot_rewards


def compute_loss(actions_log_probs, returns):
    loss = - torch.mean(actions_log_probs * returns)
    return loss


def train_policy(env, policy_model, policy_optimizer, num_train, gamma, seed):
    episode_rewards = []
    for step in range(num_train):
        episode_seed = seed if step == 0 else None
        trajectory = collect_trajectory(env, policy_model, episode_seed)
        episode_rewards.append(sum(transition['reward'] for transition in trajectory))
        states_list = [trajectory[step]['state'] for step in range(len(trajectory))]
        actions_list = [trajectory[step]['action'] for step in range(len(trajectory))]

        states = torch.cat(states_list, dim=0)
        actions = torch.tensor(actions_list, dtype=torch.long)
        returns = compute_returns(trajectory, gamma)

        policy_optimizer.zero_grad()
        logits = policy_model(states)
        actions_probs = torch.distributions.Categorical(logits=logits)
        actions_log_probs = actions_probs.log_prob(actions)
        loss = compute_loss(actions_log_probs, returns)
        loss.backward()
        with torch.no_grad():
            policy_optimizer.step()
            policy_optimizer.zero_grad()
    return episode_rewards


if __name__ == "__main__":
    trainconfig = TrainConfig()
    seed, env_id, gamma, sgd_lr, num_train = (
        trainconfig.train_seed,
        trainconfig.env_id,
        trainconfig.gamma,
        trainconfig.lr,
        trainconfig.num_train,
    )
    torch.manual_seed(seed)

    env = gym.make(env_id)
    policy_model = PolicyNetwork()

    policy_optimizer = torch.optim.SGD(policy_model.parameters(), lr=sgd_lr)
    trajectory = collect_trajectory(env, policy_model, seed)
    before_train_rewards = sum([step['reward'] for step in trajectory])
    print(f"训练前总回报{before_train_rewards}")
    episode_rewards = train_policy(env, policy_model, policy_optimizer, num_train, gamma, seed)
    plot_rewards({seed: episode_rewards}, Path(__file__).resolve().parent / 'train_rewards.png')
    trajectory = collect_trajectory(env, policy_model, seed)
    after_train_rewards = sum([step['reward'] for step in trajectory])
    print(f"训练后总回报{after_train_rewards}")
    model_path = Path(__file__).resolve().parent / "policy_weights.pt"
    save_model_weights(policy_model, model_path)
    env.close()



























