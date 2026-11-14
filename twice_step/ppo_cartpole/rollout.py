# PPO固定步数采样：保存转移、旧log-prob和价值，回合结束后重置并继续采集。
# 采样期间关闭梯度且不更新网络，末条next_state保留环境实际返回的状态。
import torch
import numpy as np
import gymnasium as gym
from networks import Ppo_net  # Ppo_net(state_dimension=4, action_dimension=2, hidden_dimension=[64, 64]) -> model
from advantages import compute_gae  # compute_gae(rewards, values, next_values, terminated, truncated, gamma, gae_lambda) -> td_error, advantage, values_target
from losses import computer_ppo_loss


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
    num_steps = 128
    gamma = 0.9
    gae_lambda = 0.7
    lr = 0.01
    clip_epsilon = 0.2
    value_coef = 0.5
    entropy_coef = 0.01
    update_epochs = 4
    minibatch_size = 32
    torch.manual_seed(seed)

    env = gym.make("CartPole-v1")
    model = Ppo_net()
    optimizer = torch.optim.SGD(model.parameters(), lr=lr)
    transitions = collect_rollout(env, model, num_steps, seed)
    (
        states, actions, rewards, next_states,
        terminated, truncated, old_log_probs, values
    ) = zip(*transitions)

    states = torch.as_tensor(np.stack(states), dtype=torch.float32)
    actions = torch.tensor(actions, dtype=torch.long)
    rewards = torch.tensor(rewards, dtype=torch.float32)
    next_states = torch.as_tensor(np.stack(next_states), dtype=torch.float32)
    terminated = torch.tensor(terminated, dtype=torch.bool)
    truncated = torch.tensor(truncated, dtype=torch.bool)
    old_log_probs = torch.stack(old_log_probs)
    values = torch.stack(values)
    # 以上批次格式转换代码主要由 AI 助手完成。

    with torch.no_grad():
        next_values = model.critic_net(next_states).squeeze(-1)
    td_error, advantages, values_target = compute_gae(
        rewards, values, next_values, terminated, truncated, gamma, gae_lambda)
    print(f"TD误差形状: {td_error.shape}, GAE形状: {advantages.shape}, 价值目标形状: {values_target.shape}")
    print(f"采集转移数: {len(transitions)}")

    for epoch in range(update_epochs):
        indices = torch.randperm(num_steps, device=states.device)
        for start in range(0, num_steps, minibatch_size):
            mb_indices = indices[start: start + minibatch_size]
            mb_states, mb_actions, mb_old_log_probs, \
                mb_advantages, mb_values_target = random_split
            (states, actions, old_log_probs, advantages, values_target, mb_indices)

            _, mb_new_log_probs, mb_new_values, mb_entropy = model(mb_states, mb_actions)
            ppo_loss = computer_ppo_loss(mb_new_log_probs, mb_old_log_probs, mb_advantages,
                                         mb_new_values, mb_values_target, mb_entropy,
                                         clip_epsilon, value_coef, entropy_coef)
            ppo_loss.backward()
            with torch.no_grad():
                ratio = torch.exp(mb_new_log_probs - mb_old_log_probs)
                print(f"更新轮次{epoch + 1}, 小批{start // minibatch_size + 1}, 更新前loss:{ppo_loss.item():.6f}, "
                      f"ratio范围:[{ratio.min().item():.4f}, {ratio.max().item():.4f}]")
                optimizer.step()
                optimizer.zero_grad()

    env.close()
