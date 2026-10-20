import torch

batch_probs = torch.tensor(
    [[0.1, 0.2, 0.7],
     [0.6, 0.3, 0.1]],
    dtype=torch.float32,
)  # shape: (2, 3)  每个动作的概率

batch_actions = torch.tensor(
    [2, 0], dtype=torch.long,
)  # shape: (2,)  选择哪个动作

expected_log_probs = torch.tensor(
    [-0.35667494, -0.51082562],
    dtype=torch.float32,
)  # shape: (2,)

batch_probs = torch.distributions.Categorical(probs=batch_probs)
batch_log_probs = batch_probs.log_prob(batch_actions)

print(batch_log_probs)
print(batch_log_probs.shape)
print(torch.allclose(batch_log_probs, expected_log_probs))