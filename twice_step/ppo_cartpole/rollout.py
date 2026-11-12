# PPO固定步数采样：保存转移、旧log-prob和价值，回合结束后重置并继续采集。
# 采样期间关闭梯度且不更新网络，末条next_state保留环境实际返回的状态。
import torch
import gymnasium as gym
from networks import Ppo_net  # Ppo_net(state_dimension=4, action_dimension=2, hidden_dimension=[64, 64]) -> model


def collect_rollout(env, model, num_step, seed):
    transitions = []
    state, _ = env.reset(seed=seed)
    for step in range(num_step):
        with torch.no_grad():
            state_tensor = torch.as_tensor(state, dtype=torch.float32)
            action, old_action_log_prob, values, _ = model(state_tensor)
        action = int(action.item())
        next_state, reward, terminated, truncated, _ = env.step(action)
        transition = (state, action, reward, next_state,
                      terminated, truncated, old_action_log_prob, values)
        transitions.append(transition)
        if terminated or truncated:
            state, _ = env.reset()
        else:
            state = next_state
    return transitions


if __name__ == "__main__":
    seed = 42
    num_steps = 128
    torch.manual_seed(seed)

    env = gym.make("CartPole-v1")
    model = Ppo_net()
    transitions = collect_rollout(env, model, num_steps, seed)
    print(f"采集转移数: {len(transitions)}")
    env.close()
