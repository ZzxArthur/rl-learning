import torch
import numpy as np
import csv
import math
from pathlib import Path
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
        self.metrics_path = Path(__file__).with_name("training_metrics.csv")

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

    def record_training_metrics(self, train_loss):
        """记录每次更新的 loss 和全网络梯度 L2 范数；异常时在参数更新前停止。"""
        loss_value = train_loss.detach().item()
        # 用 float64 计算范数，避免平方求和本身在 float32 中溢出；不修改梯度。
        grad_norm = math.sqrt(sum(
            parameter.grad.detach().double().square().sum().item()
            for parameter in self.train_net.parameters()
            if parameter.grad is not None
        ))
        update_number = self.update_step + 1
        status = "ok" if math.isfinite(loss_value) and math.isfinite(grad_norm) else "nonfinite"
        mode = "w" if self.update_step == 0 else "a"
        with self.metrics_path.open(mode, newline="", encoding="utf-8") as log_file:
            writer = csv.writer(log_file)
            if log_file.tell() == 0:
                writer.writerow(["update_attempt", "loss", "grad_norm", "status"])
                print(f"训练诊断日志: {self.metrics_path}")
            writer.writerow([update_number, loss_value, grad_norm, status])

        if status == "nonfinite":
            raise FloatingPointError(
                f"第 {update_number} 次更新出现非有限值: loss={loss_value}, "
                f"grad_norm={grad_norm}。已停止，未执行本次 optimizer.step()。"
                f"日志: {self.metrics_path}")
        if update_number == 1 or update_number % 200 == 0:
            print(f"更新{update_number}, loss:{loss_value:.6g}, 梯度范数:{grad_norm:.6g}")
        # 本函数主要代码由 AI 助手完成。

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
            self.record_training_metrics(train_loss)
            with torch.no_grad():
                self.train_optimizer.step()
                self.train_optimizer.zero_grad()
            self.update_step += 1
            if self.update_step % self.update_interval == 0:
                self.target_net.load_state_dict(self.train_net.state_dict())
