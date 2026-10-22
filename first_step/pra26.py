# 练习26：复用外部线性模型，按动作概率采样并采集 CartPole 整局六字段轨迹。
# 固定每局采样种子，记录整数动作，在任务终止或时间截断时停止并关闭环境。
import gymnasium as gym
import torch


def collect_episode(env, seed, model):
    obs, info = env.reset(seed=seed)
    torch.manual_seed(seed)
    trajectory = []
    while True:
        x = torch.tensor(obs, dtype=torch.float32).unsqueeze(0)  # 维度转换,unsqueeze(0) 在最前面增加一个维度，把 (4,) 变为 (1,4)
        logits = model(x)
        action_probs = torch.nn.functional.softmax(logits, dim=1)
        action_distribution_probs = torch.distributions.Categorical(probs=action_probs)
        action = action_distribution_probs.sample()
        next_obs, reward, terminated, truncated, info = env.step(action.item())
        trajectory.append([obs, action.item(), reward, next_obs, terminated, truncated])
        obs = next_obs
        if terminated or truncated:
            break
    return trajectory


env = gym.make("CartPole-v1")   # osb[0]:小车水平位置 osb[1]:小车水平位置 osb[2]:小车水平位置 osb[3]:小车水平位置
seed = 42
model = torch.nn.Linear(in_features=4, out_features=2)
trajectory = collect_episode(env, seed, model)
print(trajectory)
print(len(trajectory))
print(trajectory[-1][4])
print(trajectory[-1][5])


env.close()
