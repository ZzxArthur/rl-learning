import torch
import torch.nn as nn


def actor_update(actor, critic1, critic2, actor_optimizer, states, alpha):
    actions, log_probs = actor.sample(states)
    critic1.requires_grad_(False)
    critic2.requires_grad_(False)
    values1 = critic1(states, actions)
    values2 = critic2(states, actions)
    values = torch.min(values1, values2)
    actor_loss = (alpha * log_probs - values).mean()
    actor_optimizer.zero_grad()
    actor_loss.backward()
    with torch.no_grad():
        actor_optimizer.step()
    critic1.requires_grad_(True)
    critic2.requires_grad_(True)
    return actor_loss.item()


def critic_update(critic1, critic2, critic_optimizer, states, actions, td_target):
    values1 = critic1(states, actions)
    values2 = critic2(states, actions)
    critic1_loss = nn.functional.mse_loss(values1, td_target)
    critic2_loss = nn.functional.mse_loss(values2, td_target)
    critic_loss = critic1_loss + critic2_loss
    critic_optimizer.zero_grad()
    critic_loss.backward()
    with torch.no_grad():
        critic_optimizer.step()
    return critic_loss.item()


def alpha_update(actor, alpha_optimizer, target_entropy, states, log_alpha):
    with torch.no_grad():
        _, log_probs = actor.sample(states)
    alpha_loss = -(log_alpha * (log_probs.detach() + target_entropy)).mean()
    alpha_optimizer.zero_grad()
    alpha_loss.backward()
    with torch.no_grad():
        alpha_optimizer.step()
    alpha = log_alpha.detach().exp()
    return alpha


def soft_update(critic1, critic2, target_critic1, target_critic2, tau):
    with torch.no_grad():
        for online_param, target_param in zip(critic1.parameters(), target_critic1.parameters()):
            new_parameters = (1 - tau) * target_param + tau * online_param
            target_param.copy_(new_parameters)

        for online_param, target_param in zip(critic2.parameters(), target_critic2.parameters()):
            new_parameters = (1 - tau) * target_param + tau * online_param
            target_param.copy_(new_parameters)


def compute_td_target(actor, target_critic1, target_critic2, reward, next_state, terminated, gamma, alpha):
    with torch.no_grad():
        next_action, log_prob = actor.sample(next_state)
        values1, values2 = target_critic1(next_state, next_action), target_critic2(next_state, next_action)
        values = torch.min(values1, values2)
        td_target = reward + gamma * (1 - terminated) * (values - alpha * log_prob)
    return td_target
