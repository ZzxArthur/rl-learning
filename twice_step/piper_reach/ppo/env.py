# 本文件主要代码由 AI 助手完成。
import math
from pathlib import Path
import time

import gymnasium as gym
import mujoco
import numpy as np
from gymnasium import spaces


class PiperReachEnv(gym.Env):
    metadata = {"render_modes": ["human"], "render_fps": 25}

    def __init__(self, render_mode=None, capture_visual_state=False, legacy_observation=False, fixed_start=False,
                 precision_weight=0.0):
        self.render_mode = render_mode
        self.capture_visual_state = capture_visual_state
        self.legacy_observation = legacy_observation
        self.fixed_start = fixed_start
        self.linear_speed_cost = 0.0 if legacy_observation else 0.02
        self.angular_speed_cost = 0.0 if legacy_observation else 0.005
        self.precision_weight = precision_weight
        self.precision_scale = 0.04
        self.frame_skip = 20
        self.max_steps = 400
        self.success_distance = 0.01
        self.joint_step = 0.05
        self.min_goal_distance = self.success_distance
        self.viewer = None

        scene = Path(__file__).resolve().parents[3] / "assets/agilex_piper/scene.xml"
        spec = mujoco.MjSpec.from_file(str(scene))
        spec.body("link6").add_site(
            name="reach_point", pos=[0, 0, 0.09003],
            size=[0.008, 0, 0], rgba=[1, 0, 0, 1],
        )
        # 创建模型时的占位位置，每次 reset 都会更新目标。
        self.target_position = np.array([0.42252905, 0.18017793, 0.37060046])
        target_body = spec.worldbody.add_body(name="target_marker", mocap=True, pos=self.target_position)
        target_body.add_site(
            name="target", pos=[0, 0, 0],
            size=[0.012, 0, 0], rgba=[0, 1, 0, 1],
        )
        self.model = spec.compile()
        self.sim_timestep = float(self.model.opt.timestep)
        self.data = mujoco.MjData(self.model)
        self.goal_data = mujoco.MjData(self.model)
        self.target_mocap_id = self.model.body_mocapid[self.model.body("target_marker").id]
        self.reach_site_id = self.model.site("reach_point").id
        self.previous_qpos = np.zeros(6)
        self.action_space = spaces.Box(-1.0, 1.0, shape=(6,), dtype=np.float32)
        self.observation_space = spaces.Box(-np.inf, np.inf, shape=(18 if legacy_observation else 30,), dtype=np.float32)
        self.steps = 0

    def _reach_velocity(self):
        velocity = np.empty(6)
        mujoco.mj_objectVelocity(self.model, self.data, mujoco.mjtObj.mjOBJ_SITE,
                                 self.reach_site_id, velocity, 0)
        return velocity[3:], velocity[:3]

    def _observation(self, linear_velocity, angular_velocity):
        parts = (
            self.data.qpos[:6], self.data.qvel[:6],
            self.data.site("reach_point").xpos, self.target_position,
        )
        if not self.legacy_observation:
            parts += (linear_velocity, angular_velocity, self.previous_qpos)
        return np.concatenate(parts).astype(np.float32)

    def _sample_pose(self, data):
        limits = self.model.actuator_ctrlrange[:6]
        for _ in range(1000):
            mujoco.mj_resetDataKeyframe(self.model, data, 0)
            data.qpos[:6] = self.np_random.uniform(limits[:, 0], limits[:, 1])
            data.ctrl[:6] = data.qpos[:6]
            mujoco.mj_forward(self.model, data)
            # 过滤连杆间或连杆与地面的明显穿入；允许微小数值误差。
            if not any(contact.dist < -1e-5 for contact in data.contact):
                return
        raise RuntimeError("无法采样到无明显碰撞的机械臂姿态")

    def reset(self, *, seed=None, options=None):
        super().reset(seed=seed)
        if self.fixed_start:
            mujoco.mj_resetDataKeyframe(self.model, self.data, self.model.key("home").id)
            mujoco.mj_forward(self.model, self.data)
        else:
            self._sample_pose(self.data)
        start_position = self.data.site("reach_point").xpos.copy()
        for _ in range(1000):
            self._sample_pose(self.goal_data)
            self.target_position = self.goal_data.site("reach_point").xpos.copy()
            if np.linalg.norm(start_position - self.target_position) >= self.min_goal_distance:
                break
        else:
            raise RuntimeError("无法采样到与起点有足够距离的目标")
        self.data.mocap_pos[self.target_mocap_id] = self.target_position
        mujoco.mj_forward(self.model, self.data)
        self.previous_qpos[:] = self.data.qpos[:6]
        self.steps = 0
        distance = float(np.linalg.norm(self.data.site("reach_point").xpos - self.target_position))
        if self.render_mode == "human":
            self.render()
        info = {"distance": distance, "is_success": distance < self.success_distance}
        if self.capture_visual_state:
            info["visual_qpos"] = self.data.qpos.copy()
            info["visual_target"] = self.target_position.copy()
        return self._observation(*self._reach_velocity()), info

    def step(self, action):
        action = np.asarray(action, dtype=np.float32)
        action = np.clip(action, self.action_space.low, self.action_space.high)
        limits = self.model.actuator_ctrlrange[:6]
        self.previous_qpos[:] = self.data.qpos[:6]
        self.data.ctrl[:6] = np.clip(
            self.data.qpos[:6] + self.joint_step * action, limits[:, 0], limits[:, 1]
        )
        for _ in range(self.frame_skip):
            mujoco.mj_step(self.model, self.data)
        mujoco.mj_forward(self.model, self.data)
        self.steps += 1

        distance = float(np.linalg.norm(self.data.site("reach_point").xpos - self.target_position))
        linear_velocity, angular_velocity = self._reach_velocity()
        reward = (-distance
                  - self.linear_speed_cost * float(np.linalg.norm(linear_velocity))
                  - self.angular_speed_cost * float(np.linalg.norm(angular_velocity)))
        if self.precision_weight:
            # 让同样的距离改善在接近目标时获得更大的奖励差。
            reward += self.precision_weight * math.expm1(-distance / self.precision_scale)
        terminated = distance < self.success_distance
        truncated = self.steps >= self.max_steps
        if self.render_mode == "human":
            self.render()
        info = {"distance": distance, "is_success": terminated}
        if self.capture_visual_state:
            info["visual_qpos"] = self.data.qpos.copy()
            info["visual_target"] = self.target_position.copy()
        return self._observation(linear_velocity, angular_velocity), reward, terminated, truncated, info

    def render(self):
        if self.render_mode == "human":
            if self.viewer is None:
                import mujoco.viewer
                self.viewer = mujoco.viewer.launch_passive(self.model, self.data)
                self.viewer.cam.lookat[:] = [0.0, 0.0, 0.3]
                self.viewer.cam.distance = 1.5
                self.viewer.cam.azimuth = 135
                self.viewer.cam.elevation = -25
            if self.viewer.is_running():
                self.viewer.sync()

    def close(self):
        if self.viewer is not None:
            self.viewer.close()
            self.viewer = None
            # MuJoCo 的查看器线程异步退出；立即结束进程在本机曾触发崩溃。
            time.sleep(0.3)
