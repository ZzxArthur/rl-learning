# 练习28：将模型采集的轨迹组成批次，复用采样模型查询已执行动作的对数概率。
# 比较同一条记录的单独与批量查询结果，并保留对模型参数的梯度连接。
import numpy as np
import torch
import gymnasium as gym


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


env = gym.make("CartPole-v1")   # osb[0]:小车水平位置 osb[1]:小车水平位置 osb[2]:小车水平位置 osb[3]:小车水平位置
seed = 42
torch.manual_seed(seed)
model_sample = torch.nn.Linear(in_features=4, out_features=2)
trajectory = collect_episode(env, seed, model_sample)

obs_batch = torch.tensor(
    np.stack([step[0] for step in trajectory]),
    dtype=torch.float32,
)
print(obs_batch)
actions = torch.tensor(
    [step[1] for step in trajectory],
    dtype=torch.long,
)
print(actions)

logits = model_sample(obs_batch)
distribution_logits = torch.distributions.Categorical(logits=logits)
print(distribution_logits)
action_log_prob = distribution_logits.log_prob(actions)
print(action_log_prob)
env.close()

# 单独计算第 0 条记录；[:1] 保留单样本批维，不重新采样动作。
single_obs = obs_batch[:1]  # (1, 4)
single_action = actions[:1]  # (1,)
single_logits = model_sample(single_obs)  # (1, 2)
single_distribution = torch.distributions.Categorical(logits=single_logits)
single_log_prob = single_distribution.log_prob(single_action)  # (1,)
batch_first_log_prob = action_log_prob[:1]  # (1,)

print("单条 log_prob:", single_log_prob)
print("批量第 0 项 log_prob:", batch_first_log_prob)
print("单条与批量结果是否接近:", torch.allclose(single_log_prob, batch_first_log_prob))
