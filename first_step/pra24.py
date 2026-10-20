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
env = gym.make("CartPole-v1")
seed = 42
obvertion = env.reset(seed=seed) # 获得初始观测
print(obvertion)


# env.action_space.seed(seed) # 固定随机动作种子
# action = env.action_space.sample()
# env.step(action)






























