# 练习56：多轮采集轨迹，分别更新策略模型和状态价值模型。
import torch

from pra47 import TwoStepEnv
from pra47 import collect_episodes  # collect_episodes(env, model, num_episodes) 返回 episodes
from pra46 import prepare_batch  # prepare_batch(episodes, gamma) 返回 states, next_states, actions, returns, terminated
from pra54 import update_policy  # update_policy(policy_model, value_model, optimizer, states, actions, returns) 返回更新后的策略损失
from pra55 import update_value  # update_value(value_model, value_optimizer, states, returns) 返回更新前的价值 MSE
from pra50 import expected_total_reward  # expected_total_reward(model) 返回策略的期望原始总奖励

if __name__ == "__main__":
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
    num_episodes = 10
    num_batches = 50

    # 用于查看两个状态的价值预测
    check_states = torch.tensor(
        [[1.0, 0.0], [0.0, 1.0]],
        dtype=torch.float32,
    )

    print('训练前策略期望奖励：', expected_total_reward(policy_model))
    for step in range(num_batches):
        episodes = collect_episodes(env, policy_model, num_episodes)
        states, next_states, actions, returns, terminated = prepare_batch(episodes, gamma)
        policy_loss = update_policy(policy_model, value_model, policy_optimizer, states, actions, returns)
        value_loss = update_value(value_model, value_optimizer, states, returns)
        if (step + 1) % 10 == 0:
            raw_gt = expected_total_reward(policy_model)
            with torch.no_grad():
                predicted_values = value_model(check_states).squeeze(-1)  # 用model就会有梯度
            print(f'第 {step + 1} 轮：策略期望奖励 {raw_gt:.4f}，价值预测 {predicted_values.tolist()}，价值更新前 MSE {value_loss:.4f}')
