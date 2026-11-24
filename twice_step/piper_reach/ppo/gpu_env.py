# 本文件主要代码由 AI 助手完成。
"""Piper Reach 的 MJWarp 批量物理环境。

动作、物理推进、观测与奖励在 CUDA 上计算。每次单局结束后，起点和目标
使用原 CPU MuJoCo 环境新采样并筛除静态穿入；判断完成环境时存在 GPU→CPU 同步。
"""
import numpy as np
import torch
import mujoco_warp as mjw
import warp as wp

from twice_step.piper_reach.ppo.env import PiperReachEnv


class PiperReachGPU:
    def __init__(self, num_envs=1024, device="cuda:0", nconmax=128, njmax=256, fixed_start=False,
                 precision_weight=0.0):
        self.num_envs = int(num_envs)
        self.device = torch.device(device)
        if self.num_envs < 1:
            raise ValueError("num_envs 必须大于 0")
        if self.device.type != "cuda" or not torch.cuda.is_available():
            raise RuntimeError("PiperReachGPU 需要可用的 NVIDIA CUDA 设备")
        if self.device.index is None:
            self.device = torch.device("cuda", torch.cuda.current_device())
        self.nconmax = nconmax
        self.njmax = njmax

        self.cpu_env = PiperReachEnv(render_mode=None, fixed_start=fixed_start, precision_weight=precision_weight)
        self.cpu_model = self.cpu_env.model
        self.frame_skip = self.cpu_env.frame_skip
        self.max_steps = self.cpu_env.max_steps
        self.success_distance = self.cpu_env.success_distance
        self.joint_step = self.cpu_env.joint_step
        self.min_goal_distance = self.cpu_env.min_goal_distance
        self.linear_speed_cost = self.cpu_env.linear_speed_cost
        self.angular_speed_cost = self.cpu_env.angular_speed_cost
        self.precision_weight = self.cpu_env.precision_weight
        self.precision_scale = self.cpu_env.precision_scale
        self.observation_space = self.cpu_env.observation_space
        self.action_space = self.cpu_env.action_space
        self.sim_timestep = self.cpu_env.sim_timestep
        self.target_mocap_id = self.cpu_env.target_mocap_id
        self.reach_site_id = self.cpu_model.site("reach_point").id
        self.reach_body_id = int(self.cpu_model.site_bodyid[self.reach_site_id])
        self.reach_root_id = int(self.cpu_model.body_rootid[self.reach_body_id])

        wp.init()
        self.warp_device = wp.device_from_torch(self.device)
        with wp.ScopedDevice(self.warp_device), wp.ScopedStream(wp.stream_from_torch(torch.cuda.current_stream(self.device))):
            self.model = mjw.put_model(self.cpu_model)
            self.data = mjw.make_data(self.cpu_model, nworld=self.num_envs, nconmax=nconmax, njmax=njmax)
        self.qpos = wp.to_torch(self.data.qpos)
        self.qvel = wp.to_torch(self.data.qvel)
        self.ctrl = wp.to_torch(self.data.ctrl)
        self.site_xpos = wp.to_torch(self.data.site_xpos)
        self.body_velocity = wp.to_torch(self.data.cvel)
        self.subtree_com = wp.to_torch(self.data.subtree_com)
        self.mocap_pos = wp.to_torch(self.data.mocap_pos)
        self.targets = torch.zeros((self.num_envs, 3), dtype=torch.float32, device=self.device)
        self.previous_qpos = torch.empty_like(self.qpos[:, :6])
        self.steps = torch.zeros(self.num_envs, dtype=torch.int64, device=self.device)
        self.limits = torch.as_tensor(
            self.cpu_model.actuator_ctrlrange[:6].copy(), dtype=torch.float32, device=self.device
        )

    def _reach_velocity(self):
        spatial_velocity = self.body_velocity[:, self.reach_body_id]
        angular_velocity = spatial_velocity[:, :3]
        # cvel 的线速度在根子树质心处，平移到末端 site 后才是末端线速度。
        offset = self.site_xpos[:, self.reach_site_id] - self.subtree_com[:, self.reach_root_id]
        linear_velocity = spatial_velocity[:, 3:] + torch.linalg.cross(angular_velocity, offset)
        return linear_velocity, angular_velocity

    def _observation(self, linear_velocity, angular_velocity):
        return torch.cat((self.qpos[:, :6], self.qvel[:, :6],
                          self.site_xpos[:, self.reach_site_id], self.targets,
                          linear_velocity, angular_velocity, self.previous_qpos), dim=-1)

    def reset(self, seed=None):
        # 随机数由 CPU 的 Gymnasium 生成器管理；之后每个世界都独立采样。
        self.cpu_env.reset(seed=seed)
        mask = torch.ones(self.num_envs, dtype=torch.bool, device=self.device)
        return self.reset_done(mask)

    def reset_done(self, done):
        if not isinstance(done, torch.Tensor) or done.device != self.device or done.dtype != torch.bool:
            raise ValueError("done 必须是 GPU 上的布尔张量")
        if done.shape != (self.num_envs,):
            raise ValueError(f"done 应为 ({self.num_envs},)，实际为 {tuple(done.shape)}")
        done = done.contiguous()
        # 每次重置新采样，不复用固定姿态池；此处的索引读取会同步 CUDA。
        indices_cpu = torch.nonzero(done, as_tuple=False).flatten().cpu().numpy()
        if len(indices_cpu) == 0:
            return self._observation(*self._reach_velocity())

        qpos = []
        ctrl = []
        targets = []
        for _ in indices_cpu:
            self.cpu_env.reset()
            qpos.append(self.cpu_env.data.qpos.copy())
            ctrl.append(self.cpu_env.data.ctrl.copy())
            targets.append(self.cpu_env.target_position.copy())

        indices = torch.as_tensor(indices_cpu, dtype=torch.long, device=self.device)
        with wp.ScopedDevice(self.warp_device), wp.ScopedStream(wp.stream_from_torch(torch.cuda.current_stream(self.device))):
            mjw.reset_data(self.model, self.data, reset=wp.from_torch(done.contiguous()))
            self.qpos[indices] = torch.as_tensor(np.asarray(qpos), dtype=torch.float32, device=self.device)
            self.previous_qpos[indices] = self.qpos[indices, :6]
            self.ctrl[indices] = torch.as_tensor(np.asarray(ctrl), dtype=torch.float32, device=self.device)
            self.targets[indices] = torch.as_tensor(np.asarray(targets), dtype=torch.float32, device=self.device)
            self.mocap_pos[indices, self.target_mocap_id] = self.targets[indices]
            self.steps[indices] = 0
            # 重建接触和位姿，避免局部 reset 后读取旧位置或旧接触。
            mjw.forward(self.model, self.data)
        return self._observation(*self._reach_velocity())

    def step(self, actions):
        if not isinstance(actions, torch.Tensor) or actions.device != self.device:
            raise ValueError("actions 必须是 GPU 上的 PyTorch 张量")
        actions = actions.detach().to(dtype=torch.float32)
        if actions.shape != (self.num_envs, 6):
            raise ValueError(f"actions 应为 ({self.num_envs}, 6)，实际为 {tuple(actions.shape)}")
        with wp.ScopedDevice(self.warp_device), wp.ScopedStream(wp.stream_from_torch(torch.cuda.current_stream(self.device))):
            actions = actions.clamp(-1, 1)
            self.previous_qpos.copy_(self.qpos[:, :6])
            self.ctrl[:, :6] = torch.maximum(
                torch.minimum(self.qpos[:, :6] + self.joint_step * actions, self.limits[:, 1]),
                self.limits[:, 0],
            )
            for _ in range(self.frame_skip):
                mjw.step(self.model, self.data)
            mjw.forward(self.model, self.data)
            self.steps += 1
            distance = torch.linalg.vector_norm(
                self.site_xpos[:, self.reach_site_id] - self.targets, dim=-1
            )
            linear_velocity, angular_velocity = self._reach_velocity()
            reward = (-distance
                      - self.linear_speed_cost * torch.linalg.vector_norm(linear_velocity, dim=-1)
                      - self.angular_speed_cost * torch.linalg.vector_norm(angular_velocity, dim=-1))
            if self.precision_weight:
                reward += self.precision_weight * torch.expm1(-distance / self.precision_scale)
            terminated = distance < self.success_distance
            truncated = self.steps >= self.max_steps
            info = {"distance": distance, "is_success": terminated}
            if hasattr(self.data, "overflow"):
                info["overflow"] = wp.to_torch(self.data.overflow).clone()
            # 不自动重置：返回的 next_state 始终是动作后的真实状态，包括终局。
            return self._observation(linear_velocity, angular_velocity), reward, terminated, truncated, info

    def close(self):
        self.cpu_env.close()
