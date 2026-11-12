import torch


def compute_gae(rewards, values, next_values, terminated, truncated, gamma, gae_lambda):
    lenth = len(rewards)
    gae_list = torch.zeros_like(rewards)
    td_target_list = []

    for i in range(-1, -lenth-1, -1):
        




rewards = torch.tensor([1.0, 2.0, 3.0])
values = torch.tensor([0.5, 0.6, 0.7])
next_values = torch.tensor([0.6, 4.0, 0.0])
terminated = torch.tensor([False, False, True])
truncated = torch.tensor([False, True, False])
gamma = 0.9
gae_lambda = 0.8
compute_gae(rewards, values, next_values, terminated, truncated, gamma, gae_lambda)