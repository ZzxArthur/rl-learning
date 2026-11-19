"""集中放置 SAC 的 TD 目标、Actor/Critic 更新和目标网络软更新。"""

import torch


# SAC TD 目标：用两套目标 Critic 的较小估计与下一动作的熵项做自举。
def computer_td_target(actor, target_q1, target_q2, next_states, rewards, terminated, gamma, alpha):
    with torch.no_grad():
        next_actions, next_log_probs = actor.sample_action(next_states)
        min_next_q = torch.min(target_q1(next_states, next_actions), target_q2(next_states, next_actions))
        # -next_log_probs 是本次抽样对策略熵的估计，取样本期望才是策略熵。
        td_target = rewards + gamma * (1 - terminated) * (min_next_q + alpha * (-next_log_probs))
    return td_target


# SAC Critic 更新：两个在线 Q 网络分别拟合同一个 TD 目标。
def update_critics(q1, q2, optimizer, states, actions, td_target):
    values1, values2 = q1(states, actions), q2(states, actions)
    loss1, loss2 = torch.nn.functional.mse_loss(values1, td_target), torch.nn.functional.mse_loss(values2, td_target)
    loss = loss1 + loss2
    optimizer.zero_grad()
    loss.backward()
    with torch.no_grad():
        optimizer.step()
    return loss.item()


# SAC Actor 更新：用在线 Critic 的评分与熵项优化动作分布。
def update_actor(actor, q1, q2, optimizer, states, alpha):
    actions, log_probs = actor.sample_action(states)
    q1.requires_grad_(False)
    q2.requires_grad_(False)
    values1, values2 = q1(states, actions), q2(states, actions)
    values = torch.min(values1, values2)
    loss = (alpha * log_probs - values).mean()
    optimizer.zero_grad()
    loss.backward()
    with torch.no_grad():
        optimizer.step()
    q1.requires_grad_(True)
    q2.requires_grad_(True)
    return loss.item()


# SAC 目标 Critic：用 Polyak 平均让两套目标参数分别跟随在线网络。
def soft_update(online_q1, online_q2, target_q1, target_q2, tau):
    with torch.no_grad():
        for online_param, target_param in zip(online_q1.parameters(), target_q1.parameters()):
            new_parameters = (1 - tau) * target_param + tau * online_param
            target_param.copy_(new_parameters)

        for online_param, target_param in zip(online_q2.parameters(), target_q2.parameters()):
            new_parameters = (1 - tau) * target_param + tau * online_param
            target_param.copy_(new_parameters)


# 模块整理由 AI 助手完成，四个更新函数沿用原练习实现。

def updata_alpha(alpha_optimizer, log_alpha, log_probs, target_entropy):
    alpha_loss = -(log_alpha * (log_probs.detach() + target_entropy)).mean()
    alpha_optimizer.zero_grad()
    alpha_loss.backward()
    with torch.no_grad():
        alpha_optimizer.step()
    alpha = log_alpha.detach().exp()  # detach() 的作用是从一个张量中取出数值相同、但不再连接原来计算图的张量。
    return alpha