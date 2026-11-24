# 本文件主要代码由 AI 助手完成。
"""在一个 MuJoCo 窗口中显示多个独立仿真世界；本文件只负责显示。"""

import math
import time

import mujoco
import numpy as np

from twice_step.piper_reach.ppo.env import PiperReachEnv


class GPUGridViewer:
    """使用同一份 Piper 模型和网格展示多个世界，不推进训练物理。"""

    def __init__(self, num_envs, *, visible=True):
        if num_envs < 1:
            raise ValueError("num_envs 必须大于 0")
        self.num_envs = int(num_envs)
        self.env = PiperReachEnv()
        self.env.reset(seed=0)
        self.model = self.env.model
        self.data = self.env.data
        self.scratch_data = mujoco.MjData(self.model)
        self.target_mocap_id = int(self.env.target_mocap_id)
        self.viewer = None
        self.closed = False

        cols = math.ceil(math.sqrt(self.num_envs))
        rows = math.ceil(self.num_envs / cols)
        spacing = 1.8
        indices = np.arange(self.num_envs)
        self.offsets = np.column_stack((
            indices % cols * spacing,
            indices // cols * spacing,
            np.zeros(self.num_envs),
        ))
        camera_distance = max(2.0, max(cols, rows) * spacing * 1.3)
        # 网格变大时，原单臂场景的远裁剪面会把远处机械臂裁掉。
        self.model.vis.map.zfar = max(
            self.model.vis.map.zfar,
            3.0 * camera_distance / self.model.stat.extent,
        )

        reference = mujoco.MjvScene(self.model, maxgeom=1000)
        mujoco.mjv_updateScene(
            self.model, self.data, mujoco.MjvOption(), mujoco.MjvPerturb(),
            mujoco.MjvCamera(), mujoco.mjtCatBit.mjCAT_ALL, reference,
        )
        floor_id = self.model.geom("floor").id
        self.templates = [
            reference.geoms[i] for i in range(reference.ngeom)
            if not (reference.geoms[i].objtype == mujoco.mjtObj.mjOBJ_GEOM
                    and reference.geoms[i].objid == floor_id)
        ]
        self.reference = reference  # 模板引用的几何体由 reference 持有。
        self.geom_count = len(self.templates)
        self.required_geoms = (self.num_envs - 1) * self.geom_count

        if visible:
            from mujoco import viewer as mj_viewer
            self.viewer = mj_viewer.launch_passive(self.model, self.data)
            self.scene = self.viewer.user_scn
            self.viewer.cam.lookat[:] = [(cols - 1) * spacing / 2, (rows - 1) * spacing / 2, 0.3]
            self.viewer.cam.distance = camera_distance
            self.viewer.cam.azimuth = 135
            self.viewer.cam.elevation = -55
        else:
            self.scene = mujoco.MjvScene(self.model, maxgeom=max(1, self.required_geoms))

        if self.required_geoms + reference.ngeom > self.scene.maxgeom and visible:
            self.close()
            raise ValueError(f"{self.num_envs} 台机械臂超过 MuJoCo 窗口的几何体容量")

        # 复制绘制几何的描述，但 dataid 仍指向同一份模型中的网格。
        attributes = (
            "category", "dataid", "emission", "matid", "modelrbound", "objid",
            "objtype", "reflectance", "segid", "shininess", "specular", "texcoord",
            "texid", "texuniform", "transparent",
        )
        self.geoms = self.scene.geoms
        for i in range(self.num_envs - 1):
            for j, source in enumerate(self.templates):
                target = self.geoms[i * self.geom_count + j]
                mujoco.mjv_initGeom(
                    target, source.type, source.size.astype(np.float64),
                    source.pos.astype(np.float64), source.mat.astype(np.float64).reshape(-1),
                    source.rgba,
                )
                for name in attributes:
                    setattr(target, name, getattr(source, name))
                target.texrepeat[:] = source.texrepeat
        self.scene.ngeom = self.required_geoms

    def sync(self, qpos_numpy, targets_numpy):
        """接收每个世界的完整关节位置和目标世界坐标，更新共享画面。"""
        qpos = np.asarray(qpos_numpy)
        targets = np.asarray(targets_numpy)
        if qpos.shape != (self.num_envs, self.model.nq):
            raise ValueError(f"qpos 形状应为 ({self.num_envs}, {self.model.nq})")
        if targets.shape != (self.num_envs, 3):
            raise ValueError(f"targets 形状应为 ({self.num_envs}, 3)")
        if self.closed or (self.viewer is not None and not self.viewer.is_running()):
            return

        # 第 0 台由原模型绘制；其他台的姿态先在独立 CPU 数据里计算。
        positions = np.empty((self.required_geoms, 3), dtype=np.float32)
        rotations = np.empty((self.required_geoms, 3, 3), dtype=np.float32)
        for i in range(1, self.num_envs):
            self.scratch_data.qpos[:] = qpos[i]
            self.scratch_data.mocap_pos[self.target_mocap_id] = targets[i]
            mujoco.mj_forward(self.model, self.scratch_data)
            for j, source in enumerate(self.templates):
                k = (i - 1) * self.geom_count + j
                if source.objtype == mujoco.mjtObj.mjOBJ_GEOM:
                    positions[k] = self.scratch_data.geom_xpos[source.objid] + self.offsets[i]
                    rotations[k] = self.scratch_data.geom_xmat[source.objid].reshape(3, 3)
                elif source.objtype == mujoco.mjtObj.mjOBJ_SITE:
                    positions[k] = self.scratch_data.site_xpos[source.objid] + self.offsets[i]
                    rotations[k] = self.scratch_data.site_xmat[source.objid].reshape(3, 3)
                else:
                    raise RuntimeError(f"不支持的可视几何类型: {source.objtype}")

        if self.viewer is not None:
            with self.viewer.lock():
                self.data.qpos[:] = qpos[0]
                self.data.mocap_pos[self.target_mocap_id] = targets[0]
                mujoco.mj_forward(self.model, self.data)
                for i in range(self.required_geoms):
                    self.geoms[i].pos[:] = positions[i]
                    self.geoms[i].mat[:] = rotations[i]
            self.viewer.sync()
        else:
            self.data.qpos[:] = qpos[0]
            self.data.mocap_pos[self.target_mocap_id] = targets[0]
            mujoco.mj_forward(self.model, self.data)
            for i in range(self.required_geoms):
                self.geoms[i].pos[:] = positions[i]
                self.geoms[i].mat[:] = rotations[i]

    def is_running(self):
        return not self.closed and (self.viewer is None or self.viewer.is_running())

    def close(self):
        if self.closed:
            return
        self.closed = True
        if self.viewer is not None:
            self.viewer.close()
            self.viewer = None
            time.sleep(0.3)
        self.env.close()
