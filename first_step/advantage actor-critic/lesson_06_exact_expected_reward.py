# 练习6：计算两步环境中随机策略的期望原始总奖励。
import torch


def expected_total_reward(model):
    # 返回当前策略一局的期望原始总奖励，Python float
    states = torch.tensor([[1.0, 0.0], [0.0, 1.0]], dtype=torch.float32)
    rewards = torch.tensor([[1.0, 2.0], [3.0, 4.0]], dtype=torch.float32)
    with torch.no_grad():
        action_prob = model(states)
        action_prob = torch.softmax(action_prob, dim=1)
        action_reward = torch.sum(action_prob * rewards)
    return action_reward.item()


if __name__ == "__main__":
    model = torch.nn.Linear(in_features=2, out_features=2)
    with torch.no_grad():
        model.weight.zero_()
        model.bias.zero_()
    print('等概率策略：', expected_total_reward(model))

    with torch.no_grad():
        model.bias.copy_(torch.tensor([-10.0, 10.0]))
    print('接近总是选择动作1：', expected_total_reward(model))
