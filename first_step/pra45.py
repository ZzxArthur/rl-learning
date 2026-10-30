# 练习45：逐局计算折扣回报，拼接不同长度的结果并保留局间边界。
import torch


def computer_once_rewards(rewards_by_once, gamma):
    gt = torch.zeros_like(rewards_by_once, dtype=torch.float32)
    step = len(gt) - 1
    running_return = 0
    for reward in reversed(rewards_by_once):
        running_return = reward + running_return * gamma
        gt[step] = running_return
        step -= 1
    return gt


def computer_batch_rewards(rewards_by_episode, gamma):
    gt_by_episode = []
    for rewards_by_once in rewards_by_episode:
        gt = computer_once_rewards(rewards_by_once, gamma)
        gt_by_episode.append(gt)
    flat_returns = torch.cat(gt_by_episode, dim=0)
    return gt_by_episode, flat_returns


if __name__ == "__main__":
    rewards_by_episode = [
        torch.tensor([1.0, 1.0], dtype=torch.float32),  # 第 1 局，2 步
        torch.tensor([2.0], dtype=torch.float32),        # 第 2 局，1 步
        torch.tensor([2.0, 2.0, 2.0], dtype=torch.float32), # 第 3 局, 3步
    ]
    gamma = 0.9

    returns_by_episode, flat_returns = computer_batch_rewards(rewards_by_episode, gamma)
    print("各局回报：", returns_by_episode)
    print("拼接回报：", flat_returns)
    
