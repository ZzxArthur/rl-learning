# Piper Reach 的 HER 消融对照

两组从头训练。`no_her` 的 `--her-ratio` 为 `0`，`her_078` 为 `0.78`。两组都使用种子 44、256 个并行环境、每轮每环境 160 步、固定 `home`、同一奖励和近目标课程，最终各采集 1,433,600 个环境步并完成 21,760 次更新。中期检查点为 819,200 步和 12,160 次更新。

评估在 CPU MuJoCo 中使用确定性 Actor，每局最多 400 步，连续 10 步保持在 1 cm 内达标。同一行的两组逐局使用相同的 XYZ。末距统计包含失败局。

| 检查点和评估集 | 无 HER 成功 | 有 HER 成功 | 无 HER 平均末距 | 有 HER 平均末距 | 无 HER 中位末距 | 有 HER 中位末距 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| 819,200 步，普通目标 | 132/200 | 189/200 | 19.14 mm | 13.32 mm | 7.93 mm | 3.43 mm |
| 1,433,600 步，普通目标 | 194/200 | 194/200 | 5.26 mm | 4.78 mm | 3.36 mm | 1.84 mm |
| 1,433,600 步，边界目标 | 86/101 | 92/101 | 9.83 mm | 8.10 mm | 4.80 mm | 2.57 mm |

中期有 HER 多完成 57 个普通目标。最终普通目标成功数持平，有 HER 的末距中位数低 1.52 mm，边界目标多完成 6 个。以上是一个配对训练种子下的结果。有 HER 列使用本次复训的权重，原发布权重来自前一次训练。

- 中期逐局结果分别为 [无 HER](no_her_mid_goals_200.json) 和 [有 HER](her_078_mid_goals_200.json)。
- 最终普通目标的逐局结果分别为 [无 HER](no_her_goals_200.json) 和 [有 HER](her_078_goals_200.json)。
- 最终边界目标的逐局结果分别为 [无 HER](no_her_boundary_101.json) 和 [有 HER](her_078_boundary_101.json)。
- 训练输出分别在 [无 HER 日志](no_her.log) 和 [有 HER 日志](her_078.log)，各自目录保存中期和最终权重。

HER 由 Marcin Andrychowicz 等在 NeurIPS 2017 的《Hindsight Experience Replay》中提出。论文第 4.5 节比较了用同一局未来状态重标目标的 `future` 采样策略。 [论文](https://proceedings.neurips.cc/paper_files/paper/2017/hash/453fadbd8a1a3af50a9df4df899537b5-Abstract.html)
