import torch
import numpy as np
from Q_network import Q_network

# agent 根据环境选择动作


class DqnAgent():
    def __init__(self, update_interval, batch_size, gamma, lr,
                 states_num=8, actions_num=4, hidden_width=[128, 256]):
        self.update_interval = update_interval
        self.batch_size = batch_size
        self.gamma = gamma
        self.lr = lr
        self.update_step = 0
        self.train_net = Q_network(states_num, actions_num, hidden_width)
        self.target_net = Q_network(states_num, actions_num, hidden_width)
        self.target_net.load_state_dict(self.train_net.state_dict())
        self.train_optimizer = torch.optim.SGD(self.train_net.parameters(), lr=lr)

    def save(self, model_path):
        checkpoint = {
            "model_state_dict": self.train_net.state_dict(),
            "network_config": {
                "states_num": self.train_net.states_num,
                "actions_num": self.train_net.actions_num,
                "hidden_width": self.train_net.hidden_width,
            },
        }
        torch.save(checkpoint, model_path)

    def select_action(self, state, episilon):
        return self.train_net.select_action(state, episilon)

    def unpacage(self, transitions):
        states_list, actions_list, rewards_list, next_states_list, terminated_list = zip(*transitions)  # *transitions而不是~transitions
        states = torch.as_tensor(np.stack(states_list), dtype=torch.float32)
        actions = torch.tensor(actions_list, dtype=torch.long)
        rewards = torch.as_tensor(np.stack(rewards_list), dtype=torch.float32)  # float64改成32 与网络计算保持一致
        next_states = torch.as_tensor(np.stack(next_states_list), dtype=torch.float32)
        terminated = torch.tensor(terminated_list, dtype=torch.bool)
        return states, actions, rewards, next_states, terminated

    def update(self, replaybuffer):  # 回放经验池应该传输进来, 而不是自己每次重新创建
        if len(replaybuffer.buffer) >= self.batch_size:  # 记得要做判断
            transitions = replaybuffer.sample(self.batch_size)
            states, actions, rewards, next_states, terminated = self.unpacage(transitions)
            with torch.no_grad():
                next_q_values = self.target_net(next_states)  # [batch_size, 4]
                next_max_q_values = next_q_values.max(dim=1).values  # [batch_size]   不会写
                td_target = rewards + self.gamma * (next_max_q_values) * (~terminated)
            q_values = self.train_net(states)  # [batch_size, 4]不能直接使用,还需要从里面取出动作对应的
            select_q = q_values.gather(1, actions.unsqueeze(1)).squeeze(1)
            train_loss = torch.nn.functional.mse_loss(select_q, td_target)
            train_loss.backward()
            with torch.no_grad():
                self.train_optimizer.step()
                self.train_optimizer.zero_grad()
            self.update_step += 1
            if self.update_step % self.update_interval == 0:
                self.target_net.load_state_dict(self.train_net.state_dict())
