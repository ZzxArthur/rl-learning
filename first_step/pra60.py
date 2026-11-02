# 练习60：对照批量 TD 和逐步 TD 的策略表现与更新次数。
import torch
from pra46 import prepare_td_batch  # prepare_td_batch(episodes) -> states, next_states, actions, rewards, terminated
from pra47 import TwoStepEnv, collect_episodes  # collect_episodes(env, model, num_episodes) -> episodes
from pra50 import expected_total_reward  # expected_total_reward(model) -> expected_total_reward
from pra58 import compute_td_data  # compute_td_data(value_model, states, next_states, rewards, terminated, gamma) -> td_targets, td_errors
from pra58 import update_policy_td  # update_policy_td(policy_model, policy_optimizer, states, actions, td_errors) -> policy_loss
from pra58 import update_value_td  # update_value_td(value_model, value_optimizer, states, td_targets) -> value_loss
from pra59 import train_online_actor_critic  # train_online_actor_critic(env, policy_model, value_model, policy_optimizer, value_optimizer, num_episodes, gamma) -> None


def compare_training_progress(method, seed=42, gamma=0.9):
    torch.manual_seed(seed)
    env = TwoStepEnv()
    policy_model = torch.nn.Linear(2, 2)
    value_model = torch.nn.Linear(2, 1)
    with torch.no_grad():
        value_model.weight.copy_(torch.tensor([[4.5, 3.5]]))
        value_model.bias.zero_()
    policy_optimizer = torch.optim.SGD(policy_model.parameters(), lr=0.01)
    value_optimizer = torch.optim.SGD(value_model.parameters(), lr=0.05)

    results = [(0, 0, 0, expected_total_reward(policy_model))]
    completed = 0
    updates = 0
    for checkpoint in (50, 100, 250, 500):
        num_episodes = checkpoint - completed
        if method == 'batch_td':
            for _ in range(num_episodes // 10):
                episodes = collect_episodes(env, policy_model, 10)
                states, next_states, actions, rewards, terminated = prepare_td_batch(episodes)
                targets, errors = compute_td_data(value_model, states, next_states, rewards, terminated, gamma)
                update_policy_td(policy_model, policy_optimizer, states, actions, errors)
                update_value_td(value_model, value_optimizer, states, targets)
                updates += 1
        elif method == 'online_td':
            train_online_actor_critic(env, policy_model, value_model, policy_optimizer,
                                      value_optimizer, num_episodes, gamma)
            # TwoStepEnv always takes exactly two steps per episode.
            updates += num_episodes * 2
        else:
            raise ValueError(f'Unknown method: {method}')
        completed = checkpoint
        results.append((completed, completed * 2, updates, expected_total_reward(policy_model)))
    return results


if __name__ == '__main__':
    print(f"{'method':<12} {'episodes':>8} {'steps':>8} {'updates':>8} {'reward':>10}")
    for method in ('batch_td', 'online_td'):
        for episodes, steps, updates, reward in compare_training_progress(method):
            print(f'{method:<12} {episodes:>8} {steps:>8} {updates:>8} {reward:>10.4f}')
