# 本文件主要代码由 AI 助手完成。
import time

from env import PiperReachEnv


env = PiperReachEnv(render_mode="human")
observation, info = env.reset()
print(f"初始距离: {info['distance']:.4f} 米", flush=True)

try:
    while env.viewer.is_running():
        start = time.monotonic()
        # 随机动作仅用于观察环境，不是训练或到达目标的策略。
        action = env.action_space.sample()
        observation, reward, terminated, truncated, info = env.step(action)
        print(f"步骤 {env.steps:3d} | 距离 {info['distance']:.4f} 米 | 奖励 {reward:.4f}", flush=True)
        if terminated or truncated:
            print("已到达目标，重新开始" if terminated else "达到步数上限，重新开始", flush=True)
            time.sleep(1)
            observation, info = env.reset()
        elapsed = time.monotonic() - start
        time.sleep(max(0, env.frame_skip * env.model.opt.timestep - elapsed))
finally:
    env.close()
