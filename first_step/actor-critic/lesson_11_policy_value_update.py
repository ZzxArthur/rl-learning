# 练习11：用采样回报先更新策略，再单独更新状态价值模型。
import torch

from lesson_03_collect_episodes import TwoStepEnv
from lesson_03_collect_episodes import collect_episodes  # collect_episodes(env, model, num_episodes) 返回 episodes
from lesson_02_prepare_batches import prepare_batch  # prepare_batch(episodes, gamma) 返回 states, next_states, actions, returns, terminated
from lesson_10_state_value_baseline import update_policy  # update_policy(policy_model, value_model, optimizer, states, actions, returns) 返回更新后的策略损失


def update_value(value_model, value_optimizer, states, returns):
    # 用这批采样回报更新一次价值模型
    # 返回更新前的 MSE，Python float
    value_optimizer.zero_grad()
    baseline_values = value_model(states).squeeze(-1)
    value_loss = torch.mean((returns - baseline_values) ** 2)
    value_loss.backward()
    with torch.no_grad():
        value_optimizer.step()
        value_optimizer.zero_grad()

    return value_loss.item()


if __name__ == "__main__":
    # 初始价值基线，与 lesson_10_state_value_baseline 相同
    value_model = torch.nn.Linear(2, 1)
    with torch.no_grad():
        value_model.weight.copy_(torch.tensor([[4.5, 3.5]]))
        value_model.bias.zero_()

    torch.manual_seed(42)
    env = TwoStepEnv()
    policy_model = torch.nn.Linear(2, 2)

    policy_optimizer = torch.optim.SGD(policy_model.parameters(), lr=0.01)
    value_optimizer = torch.optim.SGD(value_model.parameters(), lr=0.05)

    gamma = 0.9
    episodes = collect_episodes(env, policy_model, num_episodes=3)
    states, next_states, actions, returns, terminated = prepare_batch(episodes, gamma)
    policy_loss = update_policy(policy_model, value_model, policy_optimizer, states, actions, returns)
    value_loss = update_value(value_model, value_optimizer, states, returns)
    print(policy_loss, value_loss)
