# 练习20：计算固定动作分布的期望奖励，与10000次采样的平均奖励对照。
# 通过动作出现次数加权奖励，练习概率加权平均和样本平均的关系。

import torch

probs = torch.tensor(
    [0.2, 0.3, 0.5], dtype=torch.float32
)  # shape: (3,)，动作 0、1、2 的概率

reward_table = torch.tensor(
    [-1.0, 0.0, 2.0], dtype=torch.float32
)  # shape: (3,)，动作 0、1、2 分别对应的奖励


expect_r = (probs * reward_table).sum() 
# norm 是求范数,也就是平方和开根号
print(expect_r)
torch.manual_seed(42)
num = 10000
distribution_probs = torch.distributions.Categorical(probs=probs)
action_space = distribution_probs.sample((num, ))
sum_0 = (action_space == 0).sum()
sum_1 = (action_space == 1).sum()
sum_2 = (action_space == 2).sum()
action_sum = torch.tensor([sum_0, sum_1, sum_2], dtype=torch.float32)
expect_action_sum_r = (action_sum * reward_table).sum() / num
print(expect_action_sum_r)














