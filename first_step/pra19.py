# 练习19：用共享Linear(2, 3)生成logits，计算奖励加权的批量平均损失。
# 一次SGD更新模型权重与偏置，查看梯度，并重新计算更新后的损失。

import torch

torch.manual_seed(42)

x = torch.tensor(
    [[1.0, 0.0], [0.0, 1.0]],
    dtype=torch.float32,
    requires_grad=True
)  # 两个给定状态，每个有两个特征，shape: (2, 2)


actions = torch.tensor([2, 0], dtype=torch.long)  # shape: (2,)动作
rewards = torch.tensor([1.0, 2.0], dtype=torch.float32)  # shape: (2,)奖励

model = torch.nn.Linear(in_features=2, out_features=3)

logits = model(x)

logits_probs = torch.nn.functional.softmax(logits, dim=1)
distribution_logits_probs = torch.distributions.Categorical(probs=logits_probs)
distribution_logits_log_probs = distribution_logits_probs.log_prob(actions)  # log_prob(value) 里面存放的是待优化的索引值

optimizer = torch.optim.SGD(model.parameters(), lr=0.01)
every_loss = -distribution_logits_log_probs * rewards
mean_loss = torch.mean(every_loss)
print(mean_loss)

mean_loss.backward()
print(model.weight.grad)
print(model.bias.grad)
with torch.no_grad():
    optimizer.step()
    print(model.weight.grad)
    print(model.bias.grad)
    optimizer.zero_grad()

logits = model(x)
logits_probs = torch.nn.functional.softmax(logits, dim=1)
distribution_logits_probs = torch.distributions.Categorical(probs=logits_probs)
distribution_logits_log_probs = distribution_logits_probs.log_prob(actions)  # log_prob(value) 里面存放的是待优化的索引值
every_loss = -distribution_logits_log_probs * rewards
mean_loss = torch.mean(every_loss)

print(mean_loss)




      