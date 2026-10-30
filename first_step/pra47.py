# 练习47：用固定策略采集多局轨迹，并整理成对齐的训练批次。
import torch
from pra46 import prepare_batch


def collect_episodes(env, model, num_episodes):
    # 返回 episodes
    episodes = []
    for _ in range(num_episodes):
        initial_state = env.reset()
        states_list = []
        actions_list = []
        rewards_list = []
        info = {}
        while True:
            with torch.no_grad():
                logits = model(initial_state)  # 模型先给每个动作一个分数，
                action_probs = torch.distributions.Categorical(logits=logits)  # 再把分数变成概率,
                action = action_probs.sample().item()  # 最后按概率随机选一个动作。
                next_state, reward, terminated = env.step(action)
                
                states_list.append(initial_state)
                actions_list.append(action)
                rewards_list.append(reward)

                initial_state = next_state
                if terminated:
                    break
        info['states'] = torch.stack(states_list)  # 把列表转换成张量
        info['actions'] = torch.tensor(actions_list, dtype=torch.long)
        info['rewards'] = torch.tensor(rewards_list, dtype=torch.float32)
        episodes.append(info)

    return episodes


class TwoStepEnv:
    def reset(self):
        self.step_count = 0
        return torch.tensor([1.0, 0.0], dtype=torch.float32)

    def step(self, action):
        self.step_count += 1

        if self.step_count == 1:
            next_state = torch.tensor([0.0, 1.0], dtype=torch.float32)
            reward = [1.0, 2.0][action]
            terminated = False
        else:
            next_state = torch.tensor([0.0, 0.0], dtype=torch.float32)
            reward = [3.0, 4.0][action]
            terminated = True

        return next_state, reward, terminated


if __name__ == "__main__":
    torch.manual_seed(42)
    env = TwoStepEnv()
    model = torch.nn.Linear(2, 2)
    gamma = 0.9
    initial_state = env.reset()
    num_episodes = 3
    episodes = collect_episodes(env, model, num_episodes)
    print(episodes)
    states, actions, returns = prepare_batch(episodes, gamma)
    print('拼接形状：', states.shape, actions.shape, returns.shape)
    print('拼接回报：', returns)



