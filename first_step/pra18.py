# 练习18：将两条样本的log_prob按奖励加权并取均值，直接优化两行独立logits。
# 执行一次SGD更新，再重新计算动作概率和批量平均损失。

import torch

batch_logits = torch.tensor(
    [[0.0, 0.0, 0.0],
     [0.0, 0.0, 0.0]],
    dtype=torch.float32,
    requires_grad=True,
)  # shape: (2, 3)

batch_actions = torch.tensor(
    [2, 0], dtype=torch.long,
)  # shape: (2,)

batch_rewards = torch.tensor(
    [1.0, 2.0], dtype=torch.float32,
)  # shape: (2,)

batch_logits_probs = torch.softmax(batch_logits, dim=1, dtype=torch.float32)
#dim=0：沿着行的方向进行计算，也就是每一列分别计算。
#dim=1：沿着列的方向进行计算，也就是每一行分别计算。
print(batch_logits_probs)
batch_logits_probs_distribution = torch.distributions.Categorical(probs=batch_logits_probs)
batch_action_log_probs = batch_logits_probs_distribution.log_prob(batch_actions)
optimizer = torch.optim.SGD([batch_logits], lr=0.1)
per_sample_loss = -batch_action_log_probs * batch_rewards
loss = per_sample_loss.mean()
loss.backward()
print(batch_logits.grad)
with torch.no_grad():
    optimizer.step()
    print(batch_logits.grad)
    optimizer.zero_grad()

batch_logits_probs = torch.softmax(batch_logits, dim=1, dtype=torch.float32)
batch_logits_probs_distribution = torch.distributions.Categorical(probs=batch_logits_probs)
batch_action_log_probs = batch_logits_probs_distribution.log_prob(batch_actions)
per_sample_loss = -batch_action_log_probs * batch_rewards
loss = per_sample_loss.mean()
print(batch_logits_probs)
print(loss)












