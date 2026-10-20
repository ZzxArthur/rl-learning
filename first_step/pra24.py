# 练习24：使用Gymnasium随机动作采集一局CartPole，固定环境和动作采样种子。
# 按时间保存六项转移记录，遇到terminated或truncated时保留最后一步并停止。

import gymnasium as gym


def collect_episode(env, seed):
    obs, info = env.reset(seed=seed) 
    env.action_space.seed(seed)
    action_space = env.action_space
    transition = [] 
    step = 0
    while True:
        action = action_space.sample()
        next_obs, reward, terminated, truncated, info = env.step(action)
        transition.append([obs, action, reward, next_obs, terminated, truncated])
        obs = next_obs
        step += 1
        if terminated or truncated:
            break
    return transition

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
env = gym.make("CartPole-v1")  # osb[0]:小车水平位置 osb[1]:小车水平位置 osb[2]:小车水平位置 osb[3]:小车水平位置
seed = 42
trajectory = collect_episode(env, seed)
print(trajectory)
print(sum(trajectory[i][2] for i in range(len(trajectory))))

env.close()
# env.action_space.seed(seed) # 固定随机动作种子
# action = env.action_space.sample()
# env.step(action)






























