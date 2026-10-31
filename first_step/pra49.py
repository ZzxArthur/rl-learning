# 练习49：每轮采集新轨迹并更新一次策略，记录平均奖励和损失。
import torch

from pra47 import TwoStepEnv, collect_episodes  # collect_episodes(env, model, num_episodes)
from pra46 import prepare_batch  # prepare_batch(episodes, gamma) 返回 states, actions, returns
from pra48 import update_batch  # update_batch(model, optimizer, states, actions, gt)
from pra50 import expected_total_reward

if __name__ == "__main__":
    torch.manual_seed(42)

    env = TwoStepEnv()
    model = torch.nn.Linear(2, 2)
    optimizer = torch.optim.SGD(model.parameters(), lr=0.01)

    gamma = 0.9
    num_episodes = 10   # 每轮采集 10 局
    num_batches = 50    # 共更新 50 轮
    print(expected_total_reward(model))
    for batch in range(num_batches):
        episodes = collect_episodes(env, model, num_episodes)
        mean_reward = sum(episode['rewards'].sum().item() for episode in episodes) / len(episodes)
        states, actions, gt = prepare_batch(episodes, gamma)
        loss = update_batch(model, optimizer, states, actions, gt)
        print(f'第 {batch + 1} 轮：平均奖励 {mean_reward:.2f}，更新前损失 {loss:.4f}')
        if (batch + 1) % 10 == 0:
            print(expected_total_reward(model))




















