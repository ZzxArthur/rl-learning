# 练习51：用固定基线调整回报，并完成一次策略梯度批次更新。
import torch

from pra47 import TwoStepEnv
from pra47 import collect_episodes  # collect_episodes(env, model, num_episodes) 返回 episodes
from pra46 import prepare_batch  # prepare_batch(episodes, gamma) 返回 states, actions, returns


def update_with_baseline(model, optimizer, states, actions, gt, baseline):
    # 更新模型一次，返回同一批数据更新后的损失，类型为 Python float
    optimizer.zero_grad()

    logits = model(states)
    action_probs = torch.distributions.Categorical(logits=logits)
    action_log_probs = action_probs.log_prob(actions)
    loss = -torch.mean(action_log_probs * (gt - torch.tensor(baseline)))
    loss.backward()
    with torch.no_grad():
        optimizer.step()
        optimizer.zero_grad()
    logits = model(states)
    action_probs = torch.distributions.Categorical(logits=logits)
    action_log_probs = action_probs.log_prob(actions)
    loss = -torch.mean(action_log_probs * (gt - torch.tensor(baseline)))
    return loss.item()


if __name__ == "__main__":
    torch.manual_seed(42)

    env = TwoStepEnv()
    model = torch.nn.Linear(2, 2)
    optimizer = torch.optim.SGD(model.parameters(), lr=0.01)

    gamma = 0.9
    baseline = 4.0
    num_episodes = 3

    episodes = collect_episodes(env, model, num_episodes)

    states, actions, gt = prepare_batch(episodes, gamma)
    before = [p.detach().clone() for p in model.parameters()]

    loss = update_with_baseline(model, optimizer, states, actions, gt, baseline)
    changed = [not torch.equal(old, new) for old, new in zip(before, model.parameters())]
    print('回报：', gt)
    print('基线调整后的回报：', gt - baseline)
    print('更新后损失：', loss)

    print('参数是否变化：', changed)
