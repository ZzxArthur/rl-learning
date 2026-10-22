# 练习25：将CartPole观测整理为(1, 4)输入，用Linear(4, 2)生成logits。
# 经softmax按概率采样动作、查询对应log_prob，再将动作转为整数执行一步。

import gymnasium as gym
import torch

# 方法/属性	中文含义	简化理解
# step(action)	执行动作，返回下一状态、奖励、结束标志和环境信息	执行一步动作
# reset()	将环境重置到初始状态，返回初始观测和相关信息	开始新一轮训练
# render()	渲染环境，便于观察环境状态	显示环境画面
# close()	关闭环境及其使用的外部资源	释放资源
# action_space	动作空间，规定智能体允许执行的动作	能做什么
# observation_space	观测空间，规定智能体能够接收的观测格式	能看到什么
# spec	环境的配置信息，用于创建环境	环境配置
# metadata	环境的元数据，例如支持的渲染模式、帧率等	环境说明信息
# np_random	环境使用的 NumPy 随机数生成器	控制环境随机性


env = gym.make("CartPole-v1")   # osb[0]:小车水平位置 osb[1]:小车水平位置 osb[2]:小车水平位置 osb[3]:小车水平位置
obs, info = env.reset(seed=42)
torch.manual_seed(42)
x = torch.tensor(obs, dtype=torch.float32).unsqueeze(0)  #.unsqueeze(0) 在最前面增加一个维度，把 (4,) 变为 (1,4)，四个特征值保持原样。
model = torch.nn.Linear(in_features=4, out_features=2)  # 四个观测输入对应obs 2个动作输出对应0,1
logits = model(x)
print(logits)
action_probs = torch.nn.functional.softmax(logits, dim=1)
print(action_probs)
action_probs = torch.distributions.Categorical(probs=action_probs)
action_sample = action_probs.sample()
print(action_sample)
print(action_sample.shape)
action_log_probs = action_probs.log_prob(action_sample)  # 查询这个动作的对数概率 用来优化的
print(action_log_probs)
print(action_log_probs.shape)
next_obs, reward, terminated, truncated, info = env.step(action_sample.item())
print(next_obs.shape, next_obs.dtype, reward, terminated, truncated)
env.close()

