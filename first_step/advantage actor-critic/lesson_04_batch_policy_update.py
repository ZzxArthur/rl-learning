# 练习4：使用同一策略采集的多局轨迹，计算批次损失并更新一次模型。
import torch
from lesson_03_collect_episodes import TwoStepEnv, collect_episodes
from lesson_02_prepare_batches import prepare_batch  # prepare_batch(episodes, gamma) 返回 states, next_states, actions, returns, terminated


def update_batch(model, optimizer, states, actions, gt):
    # 用这个批次更新一次模型
    # 返回更新前计算出的标量损失

    optimizer.zero_grad()
    logits = model(states)
    actions_probs = torch.distributions.Categorical(logits=logits)
    actions_logs_probs = actions_probs.log_prob(actions)  # 动作对数概率
    loss = -torch.mean(actions_logs_probs * gt)
    loss.backward()
    optimizer.step()
    return loss.item()


if __name__ == "__main__":
    torch.manual_seed(42)

    env = TwoStepEnv()
    model = torch.nn.Linear(2, 2)
    optimizer = torch.optim.SGD(model.parameters(), lr=0.01)

    gamma = 0.9
    num_episodes = 3

    # 先采集全部局，再更新模型
    episodes = collect_episodes(env, model, num_episodes)
    states, next_states, actions, returns, terminated = prepare_batch(episodes, gamma)

    before = [p.detach().clone() for p in model.parameters()]
    print('更新前损失：', update_batch(model, optimizer, states, actions, returns))
    changed = [
        not torch.equal(old, new)
        for old, new in zip(before, model.parameters())
    ]
    print("参数是否变化：", changed)
