# 练习18：对完整轨迹从后向前计算广义优势估计（GAE）。
import torch


def compute_gae(td_errors, terminated, gamma, gae_lambda):
    #  对于一条正常终止的完整轨迹，GAE 可以从后向前计算
    episode_gae = torch.zeros_like(td_errors)

    gae = 0
    length = td_errors.shape[0] + 1
    for i in range(-1, -length, -1):
        if terminated[i]:
            gae = td_errors[i]
        else:
            gae = td_errors[i] + gamma * gae_lambda * episode_gae[i + 1]
        episode_gae[i] = gae
    return episode_gae


if __name__ == "__main__":
    td_errors = torch.tensor([-0.5, 0.6, -1.0], dtype=torch.float32)
    terminated = torch.tensor([False, False, True], dtype=torch.bool)
    gamma = 0.9
    for gae_lambda in (0.0, 0.5, 1.0):
        print(gae_lambda, compute_gae(td_errors, terminated, gamma, gae_lambda))
