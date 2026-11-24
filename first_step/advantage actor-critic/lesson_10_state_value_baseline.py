# 练习10：以固定状态价值预测作为基线，更新一次策略模型。
import torch

from lesson_03_collect_episodes import TwoStepEnv
from lesson_03_collect_episodes import collect_episodes  # collect_episodes(env, model, num_episodes) 返回 episodes
from lesson_02_prepare_batches import prepare_batch  # prepare_batch(episodes, gamma) 返回 states, next_states, actions, returns, terminated


def update_policy(policy_model, value_model, optimizer, states, actions, returns):
    # 用状态价值作为基线，更新策略模型一次
    # 在更新前清理策略优化器的旧梯度。
    # 用 torch.no_grad() 计算基线，切断策略损失到价值模型的梯度路径。
    optimizer.zero_grad()
    with torch.no_grad():
        baseline_values = value_model(states).squeeze(-1)
    logits = policy_model(states)
    actions_prob = torch.distributions.Categorical(logits=logits)
    actions_log_prob = actions_prob.log_prob(actions)

    loss = - torch.mean(actions_log_prob * (returns - baseline_values))
    loss.backward()
    with torch.no_grad():
        optimizer.step()
        optimizer.zero_grad()
        baseline_values = value_model(states).squeeze(-1)
    logits = policy_model(states)
    actions_prob = torch.distributions.Categorical(logits=logits)
    actions_log_prob = actions_prob.log_prob(actions)

    loss = - torch.mean(actions_log_prob * (returns - baseline_values))
    return loss.item()


if __name__ == "__main__":
    # 固定价值模型：两个状态分别预测 4.5 和 3.5
    value_model = torch.nn.Linear(2, 1)
    with torch.no_grad():
        value_model.weight.copy_(torch.tensor([[4.5, 3.5]]))
        value_model.bias.zero_()
    print(value_model.weight)
    # 在创建策略模型前设置种子，方便复现采样
    torch.manual_seed(42)

    env = TwoStepEnv()
    policy_model = torch.nn.Linear(2, 2)
    policy_optimizer = torch.optim.SGD(policy_model.parameters(), lr=0.01)

    gamma = 0.9
    episodes = collect_episodes(env, policy_model, num_episodes=3)
    states, next_states, actions, returns, terminated = prepare_batch(episodes, gamma)
    print(states)
    loss = update_policy(policy_model, value_model, policy_optimizer, states, actions, returns)
    print(loss)
