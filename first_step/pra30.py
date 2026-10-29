# 练习30：逐局采集 CartPole 轨迹、更新模型并记录训练奖励，训练前后用独立环境评估成绩。
# 采集按终止或截断结束，截断尾值当前默认取 0；评估使用最大概率动作，窗口演示代码目前注释保留。
# 绘制逐局原始总奖励和最近 100 局平均奖励，并保存模型参数与本次实验数据。
# REINFORCE
import torch
import gymnasium as gym
import numpy as np
import time
import matplotlib.pyplot as plt


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
        with torch.no_grad():
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
    # L = torch.tensor(0.0)
    # T = actions_log_prob.numel()
    # for i in range(T):
    #     L += -actions_log_prob[i] * gt[i] / T
    L = torch.tensor(0.0)
    L += - torch.mean(actions_log_prob * gt)

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


def evaluate(model, eval_env, seeds):
    total_reward = 0
    enal_obs, info = eval_env.reset(seed=seeds)
    while True:
        with torch.no_grad():
            states = torch.tensor(enal_obs, dtype=torch.float32).unsqueeze(0)
            action = torch.argmax(model(states))
            next_obs, reward, terminated, truncated, info = eval_env.step(action.item())
            enal_obs = next_obs
            total_reward += reward
        if terminated or truncated:
            break
    return total_reward


if __name__ == "__main__":
    env = gym.make("CartPole-v1")   # osb[0]:小车水平位置 osb[1]:小车水平速度 osb[2]:杆相对竖直方向的倾斜角度 osb[3]:杆的角速度
    eval_env = gym.make("CartPole-v1")
    seed = int(np.random.randint(0, 100000))
    gamma = 0.999
    lr = 0.06
    torch.manual_seed(seed)
    model = torch.nn.Linear(in_features=4, out_features=2)
    trajectory = collect_episode(env, seed, model)
    training_rewards = []
    states = torch.tensor(np.stack([step[0] for step in trajectory]),dtype=torch.float32,)
    actions = torch.tensor([step[1] for step in trajectory],dtype=torch.long)
    rewards = torch.tensor([step[2] for step in trajectory],dtype=torch.float32)
    training_rewards.append(rewards.sum().item())
    seeds = list(range(1000, 1010))  # 1000～1009，共10个
    total_reward = 0

    for i in seeds:
        reward = evaluate(model, eval_env, i)
        print(f"训练前seed{i}, reward{reward}")
        total_reward += reward

    mean_reward = total_reward / len(seeds)
    print(f"训练前mean_reward{mean_reward}")

    gt = compute_returns(rewards, gamma, trajectory)
    optimizer = torch.optim.SGD(model.parameters(), lr=lr)
    L = update(model, optimizer, states, actions, gt)
    train_start = time.perf_counter()
    step = 1

    while step < 2000:

        trajectory = collect_episode(env, None, model)
        states = torch.tensor(np.stack([step[0] for step in trajectory]),dtype=torch.float32)
        actions = torch.tensor([step[1] for step in trajectory],dtype=torch.long)
        rewards = torch.tensor([step[2] for step in trajectory],dtype=torch.float32)
        training_rewards.append(rewards.sum().item())
        gt = compute_returns(rewards, gamma, trajectory)
        L = update(model, optimizer, states, actions, gt)
        step += 1
        if len(training_rewards) % 100 == 0:
            print(f"已经完成{len(training_rewards)}局")
            print(f"最近一百局的平均奖励{sum(training_rewards[-100:]) / 100.0}")

    seeds = list(range(1000, 1010))  # 1000～1009，共10个
    total_reward = 0
    train_start = time.perf_counter()
    for i in seeds:
        reward = evaluate(model, eval_env, i)
        print(f"训练后seed{i}, reward{reward}")
        total_reward += reward

    mean_reward = total_reward / len(seeds)
    print(f"训练后mean_reward{mean_reward}")

    model_parameters = {
        "model_state_dict": model.state_dict(),
        "seed": seed,
        "gamma": gamma,
        "lr": lr,
        "training_rewards": training_rewards
    }

    torch.save(model_parameters, f"/home/hatepp/piper/RL/first_step_model/pra30_seed{seed}_lr{lr}.pt")
    env.close()
    eval_env.close()

    # 每个均值点放在对应窗口的最后一局：首点是第 1—100 局的平均值。
    window = 100
    mean_rewards = []
    for end in range(window, len(training_rewards) + 1):
        mean_rewards.append(sum(training_rewards[end - window:end]) / window)

    fig, ax = plt.subplots(figsize=(10, 5), layout="constrained")
    ax.plot(
        range(1, len(training_rewards) + 1), training_rewards,
        color="#888888", linewidth=0.7, label="Episode reward (sampled actions)",
    )
    ax.plot(
        range(window, len(training_rewards) + 1), mean_rewards,
        color="#005A9C", linewidth=2, linestyle="--", label="Trailing mean (100 episodes)",
    )
    ax.set(
        xlabel="Training episode", ylabel="Undiscounted episode reward",
        title=f"CartPole-v1 training | seed={seed}, lr={lr}", ylim=(0, 510),
    )
    ax.grid(axis="y", alpha=0.2)
    ax.legend(loc="upper left")
    fig.savefig(f"/home/hatepp/piper/RL/first_step_model/pra30_seed{seed}_lr{lr}.png", dpi=180)
    plt.show()

    # 训练结束后，用独立环境显示当前模型控制小车的效果。
    # import pygame

    # render_env = gym.make("CartPole-v1", render_mode="human")
    # render_running = True
    # try:
    #     with torch.no_grad():
    #         for render_episode in range(3):
    #             obs, info = render_env.reset(seed=1000 + render_episode)
    #             total_reward = 0.0

    #             while render_running:
    #                 for event in pygame.event.get():
    #                     if event.type == pygame.QUIT:
    #                         render_running = False
    #                 if not render_running:
    #                     break

    #                 x = torch.tensor(obs, dtype=torch.float32).unsqueeze(0)
    #                 action = model(x).argmax(dim=1).item()
    #                 obs, reward, terminated, truncated, info = render_env.step(action)
    #                 total_reward += reward
    #                 if terminated or truncated:
    #                     break

    #             if not render_running:
    #                 break
    #             print(f"可视化第 {render_episode + 1} 局，总奖励：{total_reward:.0f}")
    # finally:
    #     render_env.close()
