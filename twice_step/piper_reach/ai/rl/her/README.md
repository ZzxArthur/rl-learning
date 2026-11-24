# Piper Reach · SAC + HER

固定 `home` 姿态启动，接受任意指定的三维世界坐标（米）。Actor 接收六个关节的位置和速度、末端位置和速度、目标位移，直接输出六个归一化关节增量。每次控制将增量乘以 `0.12 rad`，再按模型关节限幅发送给 MuJoCo 位置伺服。

## 运行

在仓库根目录执行，`--render` 可在 MuJoCo 里观看。

```bash
/home/hatepp/miniconda3/envs/piper/bin/python -B twice_step/piper_reach/ai/rl/her/evaluate.py \
  twice_step/piper_reach/ai/rl/her/actor.pt --goal 0.35 -0.15 0.30 --render
```

`--goals` 接受形如 `[[x,y,z], ...]` 的 JSON 数组。`--report` 可输出逐局记录。去掉 `--render` 可快速复测。

## 仿真评估

每局最多 400 个控制步。连续 10 个控制步保持在目标 1 cm 内判成功。每次控制步结束后发现超过 1 mm 的穿入就停止并计失败。

| 目标 | 成功 | 穿入失败 | 平均最终误差 |
| --- | ---: | ---: | ---: |
| 预留的 200 个普通目标 | 194/200 | 2 | 4.74 mm |
| 预留的 101 个边界目标 | 97/101 | 0 | 5.92 mm |
| 指定坐标 `(0.35, -0.15, 0.30)` | 1/1 | 0 | 1.11 mm |

指定坐标用 17 步完成，包括保持阶段，共 0.68 秒仿真时间。普通目标和边界目标的距离统计包含失败局。评估使用确定性 Actor 输出，最终目标组与权重选择组分开。原始数据见同目录 `goals_200.json`、`boundary_101.json` 和 `specified_goal.json`。结论范围为当前 MuJoCo 模型、固定起点与以上测试目标。

## 复训

在已有 `.venv_mjwarp` 环境运行。轨迹从 `home` 出发，先以近目标占较高比例进行探索，再混合更广的目标范围。训练时从同一轨迹中重采后续达到的位置作为目标，重新计算实际末端位置的奖励。采集动作由随机探索和 SAC Actor 产生，评估由冻结后的 Actor 直接选动作。

```bash
OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 \
  twice_step/piper_reach/.venv_mjwarp/bin/python -B twice_step/piper_reach/ai/rl/her/train.py \
  --run-dir twice_step/piper_reach/ai/rl/her/new_run --seed 44 --worlds 256 \
  --horizon 160 --steps 1400000 --joint-step 0.12 --curriculum-steps 650000
```

训练按完整批次计步，正式权重使用 1,433,600 个 MuJoCo Warp 环境步和 21,760 次 SAC 更新。两个候选点按独立种子 `18000` 的 50 局稳定到达率选择（1,187,840 步为 48/50，最终点为 49/50）。[HER 原论文](https://arxiv.org/abs/1707.01495) · [Stable-Baselines3 的 HER 回放实现](https://github.com/DLR-RM/stable-baselines3/blob/master/stable_baselines3/her/her_replay_buffer.py)
