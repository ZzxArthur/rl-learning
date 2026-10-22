# 练习30：逐局采集 CartPole 轨迹、计算回报并更新模型，训练后用窗口演示模型表现。
# 训练仅处理真正终止的轨迹；可视化使用最大概率动作，关闭窗口可提前退出。
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
            g_t[t] = rewards[t] + gamma * trail_value

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


def update(model, optimizer, states, actions, returns):
    logits = model(states)
    logits_distribution = torch.distributions.Categorical(logits=logits)
    actions_log_prob = logits_distribution.log_prob(actions)
    L = loss(actions_log_prob, returns)
    L.backward()
    with torch.no_grad():
        optimizer.step()
        optimizer.zero_grad()

    return L


if __name__ == "__main__":
    env = gym.make("CartPole-v1")   # osb[0]:小车水平位置 osb[1]:小车水平速度 osb[2]:杆相对竖直方向的倾斜角度 osb[3]:杆的角速度
    seed = int(np.random.randint(0, 100000))
    gamma = 0.999
    lr = 0.06

    torch.manual_seed(seed)

    model = torch.nn.Linear(in_features=4, out_features=2)
    trajectory = collect_episode(env, seed, model)

    states = torch.tensor(np.stack([step[0] for step in trajectory]),dtype=torch.float32,)
    actions = torch.tensor([step[1] for step in trajectory],dtype=torch.long)
    rewards = torch.tensor([step[2] for step in trajectory],dtype=torch.float32)

    gt = compute_returns(rewards, gamma, trajectory)
    optimizer = torch.optim.SGD(model.parameters(), lr=lr)
    L = update(model, optimizer, states, actions, gt)
    step = 0
    while step < 2000:

        trajectory = collect_episode(env, None, model)
        states = torch.tensor(np.stack([step[0] for step in trajectory]),dtype=torch.float32)
        actions = torch.tensor([step[1] for step in trajectory],dtype=torch.long)
        rewards = torch.tensor([step[2] for step in trajectory],dtype=torch.float32)
        gt = compute_returns(rewards, gamma, trajectory)
        L = update(model, optimizer, states, actions, gt)
        step += 1
    print(actions)
    print(L)
    env.close()

    # 训练结束后，用独立环境显示当前模型控制小车的效果。
    import pygame

    render_env = gym.make("CartPole-v1", render_mode="human")
    render_running = True
    try:
        with torch.no_grad():
            for render_episode in range(3):
                obs, info = render_env.reset(seed=1000 + render_episode)
                total_reward = 0.0

                while render_running:
                    for event in pygame.event.get():
                        if event.type == pygame.QUIT:
                            render_running = False
                    if not render_running:
                        break

                    x = torch.tensor(obs, dtype=torch.float32).unsqueeze(0)
                    action = model(x).argmax(dim=1).item()
                    obs, reward, terminated, truncated, info = render_env.step(action)
                    total_reward += reward
                    if terminated or truncated:
                        break

                if not render_running:
                    break
                print(f"可视化第 {render_episode + 1} 局，总奖励：{total_reward:.0f}")
    finally:
        render_env.close()
