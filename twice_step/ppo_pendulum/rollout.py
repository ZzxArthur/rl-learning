# 连续PPO固定步采样：执行裁剪动作，保存原始动作、旧log-prob和价值。
# 回合结束后重置并继续，返回当前状态以衔接下一批采样。
from pathlib import Path
import sys

import torch
import numpy as np
import gymnasium as gym
# 直接运行文件时补入仓库根目录，同时支持包导入。
if not __package__:
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from twice_step.ppo_pendulum.networks import Ppo_net  # Ppo_net(state_dimension, action_dimension, hidden_dimension) -> model


def collect_rollout(env, model, state, num_steps):
    transitions = []

    for _ in range(num_steps):
        with torch.no_grad():
            state_tensor = torch.as_tensor(state, dtype=torch.float32)
            raw_action, old_log_prob, value, _ = model(state_tensor)

        action_array = raw_action.numpy()  # Tensor → NumPy数组，形状仍是[1]
        env_action = np.clip(action_array, env.action_space.low, env.action_space.high)
        next_state, reward, terminated, truncated, _ = env.step(env_action)
        transition = (state, raw_action, reward, next_state,
                      terminated, truncated, old_log_prob, value)
        transitions.append(transition)
        if terminated or truncated:
            state, _ = env.reset()
        else:
            state = next_state

    return transitions, state


def random_split(states, actions, old_log_probs, advantages,
                 values_target, mb_indices):
    mb_states = states[mb_indices]
    mb_actions = actions[mb_indices]
    mb_old_log_probs = old_log_probs[mb_indices]
    mb_advantages = advantages[mb_indices]
    mb_values_target = values_target[mb_indices]
    return mb_states, mb_actions, mb_old_log_probs, mb_advantages, mb_values_target


if __name__ == "__main__":
    seed = 42
    num_steps = 256
    torch.manual_seed(seed)
    env = gym.make("Pendulum-v1")
    model = Ppo_net(3, 1, [64, 64])
    state, _ = env.reset(seed=seed)
    transitions, state = collect_rollout(env, model, state, num_steps)
    print(f"采集转移数:{len(transitions)}, 截断次数:{sum(t[5] for t in transitions)}")
    env.close()
