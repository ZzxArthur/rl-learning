# Actor 学的是：这个状态下，动作应该怎样选择？
# Critic 学的是：这个状态对当前策略来说有多好？
import torch
from lesson_03_collect_episodes import TwoStepEnv
from lesson_06_exact_expected_reward import expected_total_reward  # expected_total_reward(model) -> expected_total_reward
from lesson_14_batch_td_actor_critic import compute_td_data  # compute_td_data(value_model, states, next_states, rewards, terminated, gamma) -> td_targets, td_errors
from lesson_14_batch_td_actor_critic import update_policy_td  # update_policy_td(policy_model, policy_optimizer, states, actions, td_errors) -> policy_loss
from lesson_14_batch_td_actor_critic import update_value_td  # update_value_td(value_model, value_optimizer, states, td_targets) -> value_loss


def train_online_actor_critic(env, policy_model, value_model, policy_optimizer, value_optimizer, num_episodes, gamma):
    for _ in range(num_episodes):
        initial_state = env.reset().unsqueeze(0)  # (2,) -> (1, 2)
        while True:
            logits = policy_model(initial_state)
            action_prob = torch.distributions.Categorical(logits=logits)
            action = action_prob.sample().item()
            next_state, reward, terminated = env.step(action)
            next_state = next_state.unsqueeze(0)  # (2,) -> (1, 2)
            action = torch.tensor([action], dtype=torch.long)  # (1,)
            reward = torch.tensor([reward], dtype=torch.float32)  # (1,)
            terminated = torch.tensor([terminated], dtype=torch.bool)  # (1,)
            td_targets, td_errors = compute_td_data(value_model, initial_state,
                                                    next_state, reward, terminated, gamma)
            update_policy_td(policy_model, policy_optimizer, initial_state,
                                                    action, td_errors)
            update_value_td(value_model, value_optimizer, initial_state, td_targets)
            if terminated:
                break
            initial_state = next_state


if __name__ == "__main__":
    torch.manual_seed(42)

    env = TwoStepEnv()
    gamma = 0.9
    num_episodes = 500

    policy_model = torch.nn.Linear(2, 2)
    value_model = torch.nn.Linear(2, 1)

    with torch.no_grad():
        value_model.weight.copy_(torch.tensor([[4.5, 3.5]]))
        value_model.bias.zero_()

    policy_optimizer = torch.optim.SGD(policy_model.parameters(), lr=0.01)
    value_optimizer = torch.optim.SGD(value_model.parameters(), lr=0.05)
    check_states = torch.tensor([[1.0, 0.0], [0.0, 1.0]], dtype=torch.float32)

    print(f"训练前策略期望奖励: {expected_total_reward(policy_model)}")
    print(f"训练前价值预测: {value_model(check_states).detach().flatten()}")
    train_online_actor_critic(env, policy_model, value_model, policy_optimizer, value_optimizer, num_episodes, gamma)
    print(f"训练后策略期望奖励: {expected_total_reward(policy_model)}")
    print(f"训练后价值预测: {value_model(check_states).detach().flatten()}")

