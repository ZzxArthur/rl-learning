# PPO基础：比较新旧动作概率，计算正负优势下的裁剪代理损失。
# 六个固定样本的预期loss为0.2；仅验证损失公式，尚未接入策略网络训练。
import torch


def computer_ppo_loss(old_probs, new_probs, advantages, clip_epsilon):
    rho = torch.exp(torch.log(new_probs) - torch.log(old_probs))

    ppo_loss = -torch.mean(torch.min(rho * advantages,
                    torch.clip(rho, 1 - clip_epsilon, 1 + clip_epsilon) * advantages))
    return ppo_loss


# 六个不同样本中，已选动作的新旧概率；不是一个六动作分布
old_probs = torch.tensor([0.2, 0.2, 0.2, 0.2, 0.2, 0.2])
new_probs = torch.tensor([0.3, 0.1, 0.1, 0.3, 0.2, 0.2])
advantages = torch.tensor([2.0, -2.0, 2.0, -2.0, 1.0, -1.0])
clip_epsilon = 0.2
ppo_loss = computer_ppo_loss(old_probs, new_probs, advantages, clip_epsilon)
print(ppo_loss)
