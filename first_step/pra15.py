import torch


probs = torch.tensor([0.1, 0.2, 0.7], dtype=torch.float32)
actions = torch.tensor([0, 1, 2], dtype=torch.long)

expected_log_probs = torch.tensor(
    [-2.30258509, -1.60943791, -0.35667494],
    dtype=torch.float32,
)
torch.manual_seed(42)
distribution = torch.distributions.Categorical(probs=probs)
log_probe 




