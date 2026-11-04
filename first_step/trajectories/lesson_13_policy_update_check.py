# 观察使用整局轨迹更新一次策略前后的动作概率。
import torch
import gymnasium as gym
from lesson_10_cartpole_reinforce import collect_episode  # collect_episode(env, seed, model_sample) -> trajectory
from lesson_10_cartpole_reinforce import update  # update(model, optimizer, states, actions, returns) -> loss
from lesson_12_trajectory_inspection import inspect_episodes  # inspect_episodes(episodes, gamma) -> states, actions, rewards, returns


def check_policy_update(env, model, optimizer, trajectory, gamma):
    states, actions, rewards, returns = inspect_episodes(trajectory, gamma)
    with torch.no_grad():
        print(f"更新前动作概率: {torch.softmax(model(states[0]), dim=-1)}")
    update(model, optimizer, states, actions, returns)
    with torch.no_grad():
        print(f"更新后动作概率: {torch.softmax(model(states[0]), dim=-1)}")


if __name__ == "__main__":
    torch.manual_seed(42)
    env = gym.make("CartPole-v1")
    model = torch.nn.Linear(4, 2)
    optimizer = torch.optim.SGD(model.parameters(), lr=0.01)
    gamma = 0.99

    trajectory = collect_episode(env, 42, model)
    check_policy_update(env, model, optimizer, trajectory, gamma)
    env.close()

