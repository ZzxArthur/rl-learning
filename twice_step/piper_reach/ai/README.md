# Piper Reach · SAC+HER

机械臂在 MuJoCo 中从固定的 `home` 起步，到达用户输入的世界坐标 XYZ（米）。Actor 根据关节角、关节速度、末端状态和目标误差，直接输出六轴关节增量动作。连续 10 个控制步保持在 1 cm 内算成功，每局上限 400 步。控制周期为 40 ms。

## 运行

在仓库根目录激活 `piper` 环境，然后传入目标。入口默认加载本项目的 [`actor.pt`](rl/her/actor.pt)。

```bash
python twice_step/piper_reach/ai/reach.py --target 0.35 -0.15 0.30 --render
```

省略 `--render` 可无界面运行。单点结果包括末端坐标、最终误差、最近误差和执行步数。到位时退出码为 0，回合失败时为 1，无效参数为 2。

## 仿真评测

所有结果均以确定性 Actor 在 CPU MuJoCo 中执行。穿入统计取控制步采样，超过 1 mm 时回合提前结束。

| 目标集 | 连续保持成功 | 平均最终误差 | 穿入失败 |
| --- | ---: | ---: | ---: |
| 普通目标 | 194/200（97.0%） | 4.74 mm | 2 |
| 关节边界偏置目标 | 97/101（96.0%） | 5.92 mm | 0 |
| 额外随机目标（种子 43210） | 117/120（97.5%） | 2.81 mm | 1 |

示例目标 `(0.35, -0.15, 0.30)` 在 17 个控制步内达标，包括 10 步保持，最终误差 1.11 mm。逐局数据见 [`普通目标`](rl/her/goals_200.json)、[`边界目标`](rl/her/boundary_101.json) 和 [`额外随机目标`](rl/her/fresh_120.json)。

## 训练与复测

SAC 使用 256 个 MJWarp 仿真环境，完成 1,433,600 个环境步、21,760 次梯度更新。HER 从同一条轨迹选取之后实际达到的位置，作为一部分重放样本的替代目标。近距离目标课程先提供较易的到达经验，再逐渐增加全工作区目标。权重通过单独的 50 个目标选择后，使用上表中的独立目标评测。 [HER 论文](https://arxiv.org/abs/1707.01495)

```bash
twice_step/piper_reach/.venv_mjwarp/bin/python -B \
  twice_step/piper_reach/ai/rl/her/train.py \
  --run-dir twice_step/piper_reach/ai/rl/her/reproduce

python twice_step/piper_reach/ai/rl/her/evaluate.py \
  --goals twice_step/piper_reach/ai/rl/runs/goals_200.json
```

训练参数、日志和分阶段权重位于 [`rl/her/`](rl/her/)。评测覆盖仿真采样到的可达目标和所示指定点。

同时试训了有模型的目标条件 Actor。在同一批普通和边界目标上分别完成 191/200 和 94/101，因此交付入口采用 SAC+HER 权重。 [有模型试验](rl/runs/dynamics_v2/) · [可微模型训练参考](https://arxiv.org/abs/2610.28480)
