<p align="center">
  <img src="readme-cover.png" alt="小机器人在笔记本前学习，身旁是学习笔记、张量积木和机械臂的温暖插画" width="100%">
</p>

<h1 align="center">🌱 从零开始的强化学习</h1>

<p align="center">
  <strong>一个新手的学习笔记，欢迎批评、纠错和一起交流。</strong><br>
  从看懂一个张量开始，慢慢走向自己的算法与机器人。
</p>

<p align="center">
  <img src="https://img.shields.io/badge/Python-Learning-3776AB?style=flat&amp;logo=python&amp;logoColor=white" alt="Python：学习中">
  <img src="https://img.shields.io/badge/PyTorch-Practicing-EE4C2C?style=flat&amp;logo=pytorch&amp;logoColor=white" alt="PyTorch：练习中">
  <img src="https://img.shields.io/badge/Stage-Discrete_PPO-14B8A6?style=flat" alt="当前阶段：离散PPO起步">
  <img src="https://img.shields.io/badge/Feedback-Welcome-F59E0B?style=flat" alt="欢迎批评与建议">
</p>

<p align="center">
  <a href="first_step/">📚 查看练习</a> ·
  <a href="强化学习指南.txt">🧭 完整学习路线</a> ·
  <a href="https://github.com/ZzxArthur/rl-learning/issues">💬 给我提个建议</a>
</p>

---

## 👋 你好，我还是一名新手

这个仓库记录我学习强化学习的过程：写过的练习、遇到的问题，以及通过实验得到的一点点理解，也方便我在不同电脑上继续学习。

之前主要跟着教程手敲代码。现在希望逐步弄清楚每一步为什么这样写，自己完成实现和验证，并且能解释程序的行为和实验结果。

部分练习参考教程完成，学习中也会查阅官方文档、借助 AI 解释概念和检查代码。我会继续练习脱离参考独立实现，把自己的理解和问题都记录下来。

> 📍 **当前进度（2026-11-12）：离散 PPO 起步。**
> 已完成引导式 REINFORCE/DQN 和用户主导的 LunarLander DQN 基础项目；PPO clip 小题已通过，下一步在空的 `twice_step/ppo_cartpole/networks.py` 实现 ActorCritic。完整 PPO、SAC 和机器人项目仍待完成。最新任务以 [学习交接](学习交接.md) 和源码为准。

## 🧩 现在在练什么？

| 主题 | 当前状态与重点 |
| --- | --- |
| 🔢 Python / PyTorch 与基础 RL | 已有练习和项目实践，在新问题中继续巩固。 |
| 🧠 REINFORCE / DQN | 基础项目闭环完成，有教师审查、纠错和部分工具补写。 |
| 🎲 离散 PPO | ratio/clip 固定数据小题通过；网络、采样和完整更新尚待实现。 |
| 🛠️ 项目工程 | 继续练职责、梯度流向、实验记录与独立验证。 |
| 🦾 SAC / 机器人 / 视觉 | 后续方向，尚未完成相应独立项目。 |

## 🧭 一步步往前走

```mermaid
flowchart LR
    A["🌱 基础练习<br/>已有实践"] --> B["🧠 算法项目<br/>当前：离散PPO → 连续PPO / SAC"]
    B --> C["🦾 机械臂仿真<br/>状态输入 → 双相机视觉"]
    C --> D["🌉 Sim2Real<br/>逐步走向真机"]
    style A fill:#DCFCE7,stroke:#86EFAC,color:#14532D
    style B fill:#EFF6FF,stroke:#2563EB,stroke-width:2px,color:#1E3A8A
    style C fill:#FFF7ED,stroke:#FDBA74,color:#7C2D12
    style D fill:#F5F3FF,stroke:#C4B5FD,color:#4C1D95
```

接下来复用表格方法、REINFORCE、DQN 和简化 Actor-Critic/GAE 的经验，逐步独立实现 PPO、SAC。之后继续学习机械臂仿真、腕部与全局双相机，以及 Sim2Real。代码能运行、一个项目闭环和多条件稳定表现分别记录，不用一次高奖励代替完整能力证明。

这是长期路线，我会根据实际掌握情况逐步推进。

## 📂 仓库里有什么？

| 入口 | 内容 |
| --- | --- |
| [📚 first_step/](first_step/) | 按学习顺序积累的基础练习，部分文件保留了调试说明和实验结果。 |
| [🛠️ twice_step/](twice_step/) | 引导式项目、用户主导的DQN迁移和正在开始的PPO项目。 |
| [🧭 强化学习指南.txt](强化学习指南.txt) | 长期学习路线、教学方式和阶段验收要求，会随学习情况调整。 |
| [📍 学习交接.md](学习交接.md) | 当前任务、学习证据边界与未来约十步。 |
| [🗂️ 历史记录.md](历史记录.md) | 已发生的实现、检查、贡献和失败记录。 |
| [🧵 强化学习发展主线](强化学习发展主线_逐页要点与讲稿.md) | 问题驱动的算法总览、讲稿与个人教学定位。 |

## 💬 欢迎批评，也欢迎一起学习

我目前经验有限，代码和理解中可能有不准确、考虑不周的地方。**如果你发现问题，很欢迎直接指出来。**

无论是公式理解、张量形状、代码结构、命名习惯，还是测试和实验结论，都欢迎通过 [Issue](https://github.com/ZzxArthur/rl-learning/issues) 交流。如果能说明问题所在、解释原因，或者推荐适合初学者的资料，我会非常感谢。

对收到的反馈，我会认真查证、学习，并结合自己的实验逐步改进。也欢迎同样从基础开始的朋友一起交流。

<p align="center">
  <strong>感谢愿意花时间阅读、纠错和分享经验的你。🌻</strong><br>
  <sub>封面为 AI 辅助生成的学习主题插画。</sub>
</p>

<!--
Cover asset: readme-cover.png (2172 × 724)
Generation mode: built-in image_gen, using the imagegen skill.
Final generation prompt:
Use case: illustration-story. Asset type: wide panoramic GitHub README cover, approximately 3:1 aspect ratio. Primary request: an original, friendly illustration for a beginner's personal reinforcement-learning and robotics learning journal. Scene/backdrop: a warm off-white, uncluttered study space. Subject: a curious little robot learning at a laptop beside an open notebook, with a small tabletop robot arm exploring colorful blocks; subtle floating neural-node and tensor-grid motifs connected by a gently dotted learning path. Style/medium: polished editorial illustration with soft three-dimensional forms, rounded shapes, tactile matte surfaces, carefully drawn playful details. Composition/framing: wide horizontal banner with balanced spacing, complete subjects visible, generous breathing room and clean margins; readable when displayed at GitHub README width. Lighting/mood: soft warm light, curious, lively, modest, welcoming, beginner-friendly. Color palette: warm off-white ground with teal, periwinkle, and coral accents. Constraints: conceptual illustration of learning aspirations, not a technical achievement claim; opaque background; absolutely no text, no letters, no numbers, no logos, no watermarks.
-->
