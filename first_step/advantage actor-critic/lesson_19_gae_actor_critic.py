import torch
from lesson_03_collect_episodes import TwoStepEnv
from lesson_03_collect_episodes import collect_episodes  # collect_episodes(env, model, num_episodes) -> episodes
from lesson_02_prepare_batches import prepare_td_batch  # prepare_td_batch(episodes) -> states, next_states, actions, rewards, terminated
from lesson_06_exact_expected_reward import expected_total_reward  # expected_total_reward(model) -> expected_total_reward
from lesson_14_batch_td_actor_critic import compute_td_data  # compute_td_data(value_model, states, next_states, rewards, terminated, gamma) -> td_targets, td_errors
from lesson_14_batch_td_actor_critic import update_policy_td  # update_policy_td(policy_model, policy_optimizer, states, actions, td_errors) -> policy_loss
from lesson_14_batch_td_actor_critic import update_value_td  # update_value_td(value_model, value_optimizer, states, td_targets) -> value_loss
from lesson_18_generalized_advantage import compute_gae  # compute_gae(td_errors, terminated, gamma, gae_lambda) -> episode_gae


def train_actor_critic_age(env, policy_model, value_model, policy_optimizer, value_optimizer,
                           num_batches, num_episodes, gamma, gae_lambda):
    # 多轮采集完整轨迹，计算 TD 数据和 GAE，再分别更新 Actor、Critic。
    for step in range(num_batches):
        episodes = collect_episodes(env, policy_model, num_episodes)
        states, next_states, actions, rewards, terminated = prepare_td_batch(episodes)
        td_targets, td_errors = compute_td_data(value_model, states, next_states, rewards, terminated, gamma)
        gae_errors = compute_gae(td_errors, terminated, gamma, gae_lambda)
        update_policy_td(policy_model, policy_optimizer, states, actions, gae_errors)
        update_value_td(value_model, value_optimizer, states, td_targets)
    print(expected_total_reward(policy_model))


env = TwoStepEnv()
gamma = 0.9
gae_lambda = 0
num_episodes = 10
num_batches = 50

value_model = torch.nn.Linear(2, 1)
with torch.no_grad():
    value_model.weight.copy_(torch.tensor([[4.5, 3.5]]))
    value_model.bias.zero_()

torch.manual_seed(42)
policy_model = torch.nn.Linear(2, 2)

policy_optimizer = torch.optim.SGD(policy_model.parameters(), lr=0.01)
value_optimizer = torch.optim.SGD(value_model.parameters(), lr=0.05)

train_actor_critic_age(env, policy_model, value_model, policy_optimizer, value_optimizer,
                        num_batches, num_episodes, gamma, gae_lambda)
