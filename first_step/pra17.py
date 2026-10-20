import torch

logits = torch.tensor(
    [0.0, 0.0, 0.0],
    dtype=torch.float32,
    requires_grad=True,
)  # shape: (3,) 模型对不同类别的相对评分

action = torch.tensor(2, dtype=torch.long)  # shape: ()
reward = torch.tensor(-1.0, dtype=torch.float32)  # shape: ()

logits_probs = torch.softmax(logits, dim=0, dtype=torch.float32)
logits_probs_distribution = torch.distributions.Categorical(probs=logits_probs)
action_log_probs = logits_probs_distribution.log_prob(action)
optimizer = torch.optim.SGD([logits], lr=0.1)
loss = -action_log_probs * reward
loss.backward()
with torch.no_grad():
    optimizer.step()
    optimizer.zero_grad()

logits_probs = torch.softmax(logits, dim=0, dtype=torch.float32)
logits_probs_distribution = torch.distributions.Categorical(probs=logits_probs)
action_log_probs = logits_probs_distribution.log_prob(action)
loss = -action_log_probs * reward
print(logits_probs)


# reward = 1 tensor([0.3220, 0.3220, 0.3559], grad_fn=<SoftmaxBackward0>)
# reward = 0 tensor([0.3333, 0.3333, 0.3333], grad_fn=<SoftmaxBackward0>)
# reward = -1 tensor([0.3443, 0.3443, 0.3115], grad_fn=<SoftmaxBackward0>)