# PPO总损失：组合裁剪策略损失、价值回归损失和熵奖励。
import torch


def computer_ppo_loss(new_log_probs, old_log_probs, advantages, new_values, values_target,
                      entropy, clip_epsilon, value_coef, entropy_coef):
    rho = torch.exp(new_log_probs - old_log_probs)
    actor_loss = -torch.mean(torch.minimum(rho * advantages,
                                 torch.clip(rho, 1 - clip_epsilon, 1 + clip_epsilon) * advantages))
    critic_loss = torch.nn.functional.mse_loss(new_values, values_target)
    total_loss = actor_loss + value_coef * critic_loss - entropy_coef * torch.mean(entropy)
    return total_loss


if __name__ == "__main__":
    old_log_probs = torch.log(torch.tensor([0.2] * 6))
    new_log_probs = torch.log(torch.tensor([0.3, 0.1, 0.1, 0.3, 0.2, 0.2]))
    advantages = torch.tensor([2.0, -2.0, 2.0, -2.0, 1.0, -1.0])

    new_values = torch.ones(6)
    values_target = torch.zeros(6)
    entropy = torch.full((6,), 0.5)

    clip_epsilon = 0.2
    value_coef = 0.5
    entropy_coef = 0.01
    total_loss = computer_ppo_loss(
        new_log_probs, old_log_probs, advantages, new_values, values_target,
        entropy, clip_epsilon, value_coef, entropy_coef)
    print("总损失:", total_loss)
