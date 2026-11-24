# 练习14：多轮采集轨迹，使用一步 TD 更新策略和状态价值模型。
import torch
from lesson_03_collect_episodes import TwoStepEnv
from lesson_03_collect_episodes import collect_episodes  # collect_episodes(env, model, num_episodes) 返回 episodes
from lesson_02_prepare_batches import prepare_td_batch  # prepare_td_batch(episodes) 返回 states, next_states, actions, rewards, terminated
from lesson_06_exact_expected_reward import expected_total_reward  # expected_total_reward(model) 返回策略的期望原始总奖励


def compute_td_data(value_model, states, next_states, rewards, terminated, gamma):

    target_value_list = []
    with torch.no_grad():
        next_states_value = value_model(next_states).squeeze(-1)

    for step in range(len(terminated)):
        if terminated[step]:
            target_value = rewards[step]
        else:
            target_value = rewards[step] + gamma * next_states_value[step]
        target_value_list.append(target_value)
    td_targets = torch.stack(target_value_list, dim=0)
    with torch.no_grad():
        states_value = value_model(states).squeeze(-1)
        td_errors = td_targets - states_value

    return td_targets, td_errors


def update_policy_td(policy_model, policy_optimizer, states, actions, td_errors):
    # 更新策略模型一次，返回更新前策略损失，Python float
    policy_optimizer.zero_grad()
    logits = policy_model(states)
    actions_probs = torch.distributions.Categorical(logits=logits)
    actions_log_probs = actions_probs.log_prob(actions)
    policy_loss = - torch.mean(actions_log_probs * td_errors)
    policy_loss.backward()
    with torch.no_grad():
        policy_optimizer.step()
        policy_optimizer.zero_grad()
    return policy_loss.item()


def update_value_td(value_model, value_optimizer, states, td_targets):
    # 更新价值模型一次，返回更新前价值 MSE，Python float
    value_optimizer.zero_grad()
    states_value = value_model(states).squeeze(-1)
    value_loss = torch.nn.functional.mse_loss(td_targets, states_value)
    value_loss.backward()
    with torch.no_grad():
        value_optimizer.step()
        value_optimizer.zero_grad()

    return value_loss.item()


if __name__ == "__main__":
    env = TwoStepEnv()

    gamma = 0.9
    num_episodes = 10
    num_batches = 50
    value_model = torch.nn.Linear(2, 1)
    with torch.no_grad():
        value_model.weight.copy_(torch.tensor([[4.5, 3.5]]))
        value_model.bias.zero_()

    torch.manual_seed(42)
    policy_model = torch.nn.Linear(2, 2)

    policy_optimizer = torch.optim.SGD(policy_model.parameters(), lr=0.01)
    value_optimizer = torch.optim.SGD(value_model.parameters(), lr=0.05)

    check_states = torch.tensor(
        [[1.0, 0.0], [0.0, 1.0]],
        dtype=torch.float32,)
    print(f"训练前策略期望奖励:{expected_total_reward(policy_model)}")
    print(f"训练前价值预测:{value_model(check_states).detach()}")
    for step in range(num_batches):

        episodes = collect_episodes(env, policy_model, num_episodes)
        states, next_states, actions, rewards, terminated = prepare_td_batch(episodes)
        td_targets, td_errors = compute_td_data(value_model, states, next_states, rewards, terminated, gamma)
        policy_loss = update_policy_td(policy_model, policy_optimizer, states, actions, td_errors)
        value_loss = update_value_td(value_model, value_optimizer, states, td_targets)

    print(f"训练后策略期望奖励:{expected_total_reward(policy_model)}")
    print(f"训练后价值预测:{value_model(check_states).detach()}")

    # print('前四条 TD 目标：', td_targets[:4])
    # print('前四条 TD 误差：', td_errors[:4])
    # print('更新前策略损失：', policy_loss)
    # print('更新前价值 MSE：', value_loss)
