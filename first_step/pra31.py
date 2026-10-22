# 练习31：对照两动作问题中精确期望奖励与蒙特卡洛采样损失的参数梯度。
# 两条路线分别反向传播；当前采样 1000 次，损失数值不同但梯度接近。
import torch

torch.manual_seed(42)
sample_count = 10000

logits_exact = torch.tensor(
    [0.0, 0.0], dtype=torch.float32, requires_grad=True
)
logits_mc = torch.tensor(
    [0.0, 0.0], dtype=torch.float32, requires_grad=True
)
reward_table = torch.tensor([1.0, 3.0], dtype=torch.float32)

softmax_logits_exact = torch.nn.functional.softmax(logits_exact)
l_exact = -sum(softmax_logits_exact * reward_table)

l_exact.backward()
print(logits_exact.grad)
distribution_logits_ms = torch.distributions.Categorical(logits=logits_mc)
action_sample = distribution_logits_ms.sample((1000,))

sample_rewards = reward_table[action_sample]
action_probs = distribution_logits_ms.log_prob(action_sample)
sample_rewards = reward_table[action_sample]  #把 action_sample 中的每个数字当作索引，从 reward_table 中取出对应位置的值。


l_mc = -(action_probs * sample_rewards).mean()
l_mc.backward()
print(logits_mc.grad)













