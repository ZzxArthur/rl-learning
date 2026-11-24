# 本文件主要代码由 AI 助手完成。
# 只显示训练环境的状态；这里的机械臂不会运行物理仿真。
from pathlib import Path
import math
import time
import xml.etree.ElementTree as ET

import mujoco
import numpy as np


class ParallelViewer:
    def __init__(self, num_envs, show_window=True):
        if num_envs < 1:
            raise ValueError("num_envs 必须大于 0")
        self.num_envs = num_envs
        asset_dir = Path(__file__).resolve().parents[3] / "assets/agilex_piper"

        # 保留原场景的天空、地面和光源；每台机械臂单独加入。
        scene_root = ET.parse(asset_dir / "scene.xml").getroot()
        for include in scene_root.findall("include"):
            scene_root.remove(include)
        scene_root.insert(0, ET.Element("option", {
            "integrator": "implicitfast", "cone": "elliptic", "impratio": "10",
        }))
        spec = mujoco.MjSpec.from_string(ET.tostring(scene_root, encoding="unicode"))
        spec.copy_during_attach = True

        robot = mujoco.MjSpec.from_file(str(asset_dir / "piper.xml"))
        robot.body("link6").add_site(
            name="reach_point", pos=[0, 0, 0.09003],
            size=[0.015, 0, 0], rgba=[1, 0, 0, 1],
        )

        cols = math.ceil(math.sqrt(num_envs))
        rows = math.ceil(num_envs / cols)
        spacing = 1.8
        self.offsets = np.array([
            [(i % cols - (cols - 1) / 2) * spacing,
             (i // cols - (rows - 1) / 2) * spacing, 0.0]
            for i in range(num_envs)
        ])
        for i, offset in enumerate(self.offsets):
            frame = spec.worldbody.add_frame(pos=offset)
            spec.attach(robot, prefix=f"p{i}_", frame=frame)
            target = spec.worldbody.add_body(
                name=f"target_marker_{i}", mocap=True, pos=offset + [0, 0, 0.3],
            )
            target.add_site(name=f"target_{i}", pos=[0, 0, 0],
                            size=[0.022, 0, 0], rgba=[0, 1, 0, 1])

        self.model = spec.compile()
        self.data = mujoco.MjData(self.model)
        # 只初始化一次场景灯光等派生状态；后续镜像更新只计算运动学。
        mujoco.mj_forward(self.model, self.data)
        self.qpos_addresses = np.array([
            [int(self.model.joint(f"p{i}_joint{j}").qposadr[0]) for j in range(1, 9)]
            for i in range(num_envs)
        ])
        self.mocap_ids = np.array([
            self.model.body_mocapid[self.model.body(f"target_marker_{i}").id]
            for i in range(num_envs)
        ])

        self.viewer = None
        if show_window:
            from mujoco import viewer as mujoco_viewer
            self.viewer = mujoco_viewer.launch_passive(self.model, self.data)
            self.viewer.cam.lookat[:] = [0, 0, 0.3]
            self.viewer.cam.distance = max(1.5, 1.2 * max(rows, cols) * spacing)
            self.viewer.cam.azimuth = 135
            self.viewer.cam.elevation = -40

    def sync(self, qpos, targets):
        self.data.qpos[self.qpos_addresses] = np.asarray(qpos)
        self.data.mocap_pos[self.mocap_ids] = np.asarray(targets) + self.offsets
        mujoco.mj_kinematics(self.model, self.data)
        if self.viewer is not None and self.viewer.is_running():
            self.viewer.sync()

    def is_running(self):
        return self.viewer is not None and self.viewer.is_running()

    def close(self):
        if self.viewer is not None:
            self.viewer.close()
            self.viewer = None
            # 与单环境窗口一致，给异步退出的查看器线程留出时间。
            time.sleep(0.3)
