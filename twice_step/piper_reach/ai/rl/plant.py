"""Fixed-home MuJoCo task with absolute normalized joint actions."""

from pathlib import Path
import sys

import mujoco
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[4]))
from twice_step.piper_reach.ppo.env import PiperReachEnv


class Plant:
    def __init__(self, render=False):
        self.env = PiperReachEnv(fixed_start=True)
        self.model = self.env.model
        self.data = self.env.data
        self.site_id = self.env.reach_site_id
        self.home_id = self.model.key("home").id
        self.home = self.model.key("home").qpos.copy()
        lo, hi = self.model.actuator_ctrlrange[:6].T
        self.offset = (hi + lo) / 2
        self.scale = (hi - lo) / 2
        self.dt = self.env.sim_timestep * self.env.frame_skip
        self.max_steps = 400
        self.hold_steps = 10
        self.tolerance = 0.01
        self.render = render

    def reset(self, target):
        self.target = np.asarray(target, dtype=np.float64)
        mujoco.mj_resetDataKeyframe(self.model, self.data, self.home_id)
        self.data.mocap_pos[self.env.target_mocap_id] = self.target
        mujoco.mj_forward(self.model, self.data)
        self.steps = self.consecutive = 0
        self.success = self.collision = False
        self.position = self.data.site_xpos[self.site_id].copy()
        self.distance = float(np.linalg.norm(self.position - self.target))
        if self.render:
            self.env.render_mode = "human"
            self.env.render()

    def state(self):
        q = (self.data.qpos[:6] - self.offset) / self.scale
        return np.concatenate((q, self.data.qvel[:6] / 2, self.data.site_xpos[self.site_id])).astype(np.float32)

    def advance(self, action):
        command = self.offset + self.scale * np.clip(action, -1.0, 1.0)
        present = self.data.qpos[:6]
        increment = command - present
        self.data.ctrl[:6] = present + increment / max(1.0, np.max(np.abs(increment)) / 0.14)
        mujoco.mj_step(self.model, self.data, nstep=self.env.frame_skip)
        mujoco.mj_forward(self.model, self.data)

    def step(self, action):
        if self.render:
            if not self.env.viewer.is_running():
                raise KeyboardInterrupt
        self.advance(action)
        self.steps += 1
        self.position = self.data.site_xpos[self.site_id].copy()
        self.distance = float(np.linalg.norm(self.position - self.target))
        self.collision = any(c.dist < -0.001 for c in self.data.contact)
        self.consecutive = self.consecutive + 1 if self.distance < self.tolerance and not self.collision else 0
        self.success = self.consecutive >= self.hold_steps
        if self.render:
            self.env.render()

    def close(self):
        self.env.close()


def make_goals(plant, count, seed):
    env = plant.env
    env.reset(seed=seed)
    home = env.data.site_xpos[plant.site_id].copy()
    goals = []
    while len(goals) < count:
        env._sample_pose(env.goal_data)
        goal = env.goal_data.site_xpos[plant.site_id].copy()
        if np.linalg.norm(goal - home) >= plant.tolerance:
            goals.append(goal)
    return np.asarray(goals)


# 本文件主要代码由 AI 助手完成。
