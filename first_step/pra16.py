# 练习16：直接优化三个logits，观察log_prob损失的符号如何影响动作概率。
# 当前loss=log_prob(action)，一次SGD更新会降低动作2的概率。

import torch

logits = torch.tensor(
    [0.0, 0.0, 0.0],
    dtype=torch.float32,
    requires_grad=True,
)  # shape: (3,)，三个可学习的动作分数

target_action = torch.tensor(
    2, dtype=torch.long,
)  # shape: ()，指定希望提高概率的动作

expected_probs = torch.softmax(logits, dim=0, dtype=torch.float32)
target_action_distribution = torch.distributions.Categorical(probs=expected_probs)
target_action_probs = target_action_distribution.log_prob(target_action)
print(target_action_probs)
optimizer = torch.optim.SGD([logits], lr=0.1)
# [L=lnp(动作2)]  减小动作概率
loss = target_action_probs  #sgd永远都在最小化loss，loss=target_action_probs， loss小 target_action_probs小 loss=-target_action_probs  loss小 target_action_probs大
print(loss)
loss.backward()
print(logits.grad)
with torch.no_grad():
    print(logits.grad)
    optimizer.step()
    optimizer.zero_grad()

expected_probs = torch.softmax(logits, dim=0, dtype=torch.float32)
target_action_distribution = torch.distributions.Categorical(probs=expected_probs)
target_action_probs = target_action_distribution.log_prob(target_action)
print(target_action_probs)
loss = target_action_probs
print(loss)