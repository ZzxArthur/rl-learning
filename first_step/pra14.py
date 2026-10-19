# 练习14：用softmax转换动作分数，并使用Categorical分布采样。
# 固定随机种子，抽取10000个动作并统计各动作的出现频率。
import torch

logits = torch.tensor(
    [[1, 2, 3], [3, 1, 0]],
    dtype=torch.float32,
)
actions_space = torch.tensor([0, 1, 2], dtype=torch.long)
expected_probs = torch.tensor(
    [[0.09003057, 0.24472848, 0.66524094],
     [0.84379473, 0.11419520, 0.04201007]],
    dtype=torch.float32,
)
expected_log_probs = torch.tensor(
    [-2.30258509, -1.60943791, -0.35667494],
    dtype=torch.float32,
)

y = torch.softmax(logits, dim=1)
# 用 torch.distributions.Categorical(probs=probs) 创建分布。
# 调用分布的 .sample()，查看一次动作编号。
# 调用 .sample((10000,))，抽取一万个动作。
# 用 torch.bincount(actions, minlength=3) 统计三个编号的次数，再除以总次数得到频率。
torch.manual_seed(42)
num = 100000
probs = torch.tensor([0.1, 0.2, 0.7], dtype=torch.float32)
distribution = torch.distributions.Categorical(probs=probs)
action_distribution = distribution.sample((num, ))
log_action_probs = distribution.log_prob(actions_space)

print(action_distribution)
print(torch.bincount(action_distribution, minlength=3)/num)
print(log_action_probs)
print(torch.allclose(log_action_probs, expected_log_probs))




