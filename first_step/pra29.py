# 练习29：复用采样模型，将整局折扣回报与已执行动作的对数概率组合为标量损失。
# 保留损失的梯度连接，仅处理真正终止的轨迹；尚未估计截断后的价值，不执行参数更新。
import torch
import gymnasium as gym
import numpy as np


def compute_returns(rewards, gamma, trajectory, trail_value=0.0):
    if rewards.ndim != 1:
        raise ValueError("奖励函数维度不匹配")
    g_t = torch.zeros_like(rewards)
    terminated = trajectory[-1][4]
    truncated = trajectory[-1][5]

    for t in range(g_t.numel()-1, -1, -1):

        if t + 1 < g_t.numel():
            g_t[t] = rewards[t] + gamma * g_t[t+1] 
        elif terminated:
            g_t[t] = rewards[t]
        elif truncated:
            raise ValueError("没有后续价值估计")
    return g_t


def collect_episode(env, seed, model_sample):
    obs, info = env.reset(seed=seed)
    trajectory = []
    while True:
        x = torch.tensor(obs, dtype=torch.float32).unsqueeze(0)  # 维度转换,unsqueeze(0) 在最前面增加一个维度，把 (4,) 变为 (1,4)
        logits = model_sample(x)
        action_probs = torch.nn.functional.softmax(logits, dim=1)
        action_distribution_probs = torch.distributions.Categorical(probs=action_probs)
        action = action_distribution_probs.sample()
        next_obs, reward, terminated, truncated, info = env.step(action.item())
        trajectory.append([obs, action.item(), reward, next_obs, terminated, truncated])
        obs = next_obs
        if terminated or truncated:
            break
    return trajectory


def loss(actions_log_prob, gt):
    L = torch.tensor(0.0)
    T = actions_log_prob.numel()
    for i in range(T):
        L += - actions_log_prob[i] * gt[i] / T
    return L


if __name__ == "__main__":
    env = gym.make("CartPole-v1")   # osb[0]:小车水平位置 osb[1]:小车水平位置 osb[2]:小车水平位置 osb[3]:小车水平位置
    seed = 42
    gamma = 0.99
    lr = 0.01
    torch.manual_seed(seed)
    model = torch.nn.Linear(in_features=4, out_features=2)
    trajectory = collect_episode(env, seed, model)

    states = torch.tensor(
        np.stack([step[0] for step in trajectory]),
        dtype=torch.float32,
    )

    actions = torch.tensor(
        [step[1] for step in trajectory],
        dtype=torch.long,
    )
    rewards = torch.tensor(
        [step[2] for step in trajectory],
        dtype=torch.float32,
    )
    gt = compute_returns(rewards, gamma, trajectory)

    logits = model(states)
    logits_distribution = torch.distributions.Categorical(logits=logits)
    actions_log_prob = logits_distribution.log_prob(actions)
    print(actions_log_prob)
    L = loss(actions_log_prob, gt)
    print
    optimizer = torch.optim.SGD(model.parameters(), lr=lr)
    L.backward()
    with torch.no_grad():
        optimizer.step()
        optimizer.zero_grad()
    logits = model(states)
    logits_distribution = torch.distributions.Categorical(logits=logits)
    actions_log_prob = logits_distribution.log_prob(actions)
    L = loss(actions_log_prob, gt)
    print(L)

    env.close()
