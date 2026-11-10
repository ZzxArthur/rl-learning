import numpy as np
import torch
from q_network import QNetwork  # QNetwork() -> Q network


class DQNAgent():
    def __init__(self, gamma, lr, batch_size, target_update_interval,
                 states_num=4, actions_num=2, hidden_width=256):
        self.gamma = gamma
        self.lr = lr
        self.target_update_interval = target_update_interval
        self.batch_size = batch_size
        self.train_net = QNetwork(states_num, actions_num, hidden_width)
        self.target_net = QNetwork(states_num, actions_num, hidden_width)
        self.target_net.load_state_dict(self.train_net.state_dict())
        self.train_optimizer = torch.optim.SGD(self.train_net.parameters(), lr=lr)
        self.step = 0

    def select_action(self, state, epsilon):
        action = self.train_net.select_action(state, epsilon)
        return action

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

    def update(self, replay_buffer):
        if len(replay_buffer.buffer) >= self.batch_size:
            transitions = replay_buffer.sample(self.batch_size)
            states_list, actions_list, rewards_list, next_states_list, terminated_list = zip(*transitions)
            states = torch.stack(states_list)
            actions = torch.tensor(actions_list, dtype=torch.long)
            rewards = torch.tensor(rewards_list, dtype=torch.float32)
            next_states = torch.as_tensor(np.stack(next_states_list), dtype=torch.float32)
            terminated = torch.tensor(terminated_list, dtype=torch.bool)
            with torch.no_grad():
                next_states_q = self.target_net(next_states)  # [batch_size, 2]
                max_next_q = next_states_q.max(dim=1).values  # [batch_size]
                td_target = rewards + self.gamma * max_next_q * (~terminated)

            q = self.train_net(states)  # [batch_size, 2]
            selected_q = q.gather(dim=1, index=actions.unsqueeze(1)).squeeze(1)
            mse_errors = torch.nn.functional.mse_loss(selected_q, td_target)
            mse_errors.backward()
            with torch.no_grad():
                self.train_optimizer.step()
                self.train_optimizer.zero_grad()
            self.step += 1
            if self.step % self.target_update_interval == 0:
                self.target_net.load_state_dict(self.train_net.state_dict())
