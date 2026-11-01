# 练习52：固定策略采样，比较有无基线时梯度的波动。
import torch

from pra47 import TwoStepEnv
from pra47 import collect_episodes  # collect_episodes(env, model, num_episodes) 返回 episodes
from pra46 import prepare_batch  # prepare_batch(episodes, gamma) 返回 states, next_states, actions, returns, terminated


def compute_gradient(model, states, actions, returns, baseline):
    # 返回一维梯度张量，不更新模型
    logits = model(states)
    action_probs = torch.distributions.Categorical(logits=logits)
    action_log_probs = action_probs.log_prob(actions)
    loss = - torch.mean(action_log_probs * (returns - torch.tensor(baseline)))
    gradients = torch.autograd.grad(loss, tuple(model.parameters()))
    weight_grad = gradients[0].reshape(-1)
    bias_grad = gradients[1].reshape(-1)
    gradients = torch.cat([weight_grad, bias_grad], dim=0)
    return gradients


torch.manual_seed(42)

env = TwoStepEnv()
model = torch.nn.Linear(2, 2)

gamma = 0.9
num_episodes = 10  # 每批 10 局
num_batches = 200  # 采集 200 批
nobaseline_gradients_list = []
baseline_gradients_list = []
for _ in range(num_batches):
    episodes = collect_episodes(env, model, num_episodes)
    states, next_states, actions, returns, terminated = prepare_batch(episodes, gamma)
    nobaseline_gradients_list.append(compute_gradient(model, states, actions, returns, baseline=0))
    baseline_gradients_list.append(compute_gradient(model, states, actions, returns, baseline=4))
nobaseline_gradients = torch.stack(nobaseline_gradients_list, dim=0)
baseline_gradients = torch.stack(baseline_gradients_list, dim=0)
nobaseline_gradient_variance = nobaseline_gradients.var(dim=0, unbiased=False).mean()
baseline_gradient_variance = baseline_gradients.var(dim=0, unbiased=False).mean()
print('梯度均值（无基线）：', nobaseline_gradients.mean(dim=0))
print('梯度均值（基线4）：', baseline_gradients.mean(dim=0))
print('平均方差（无基线、基线4）：', nobaseline_gradient_variance, baseline_gradient_variance)
print("该休息了,代码敲多了对眼睛不好")

