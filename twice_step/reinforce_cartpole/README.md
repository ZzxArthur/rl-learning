# CartPole REINFORCE

使用 PyTorch 和 Gymnasium 实现 REINFORCE，在 `CartPole-v1` 中学习通过选择左右动作让杆保持平衡。策略网络接收四维状态，输出两个动作的 logits；训练时从动作分布采样，使用整局的折扣回报作为策略梯度的权重。

## 文件说明

| 文件 | 作用 |
| --- | --- |
| `policy.py` | 定义输出动作 logits 的策略网络。 |
| `rollout.py` | 与环境交互，采集一整局轨迹。 |
| `returns.py` | 根据轨迹奖励计算每一步的折扣回报。 |
| `train.py` | 计算策略损失，多轮采样并更新模型，保存训练后的权重。 |
| `config.py` | 保存训练环境、种子、轮数、折扣因子和学习率。 |
| `checkpoint.py` | 保存模型参数，或加载参数并创建策略网络。 |
| `evaluate.py` | 使用贪心动作评估已保存模型。 |
| `experiments.py` | 用多个训练种子分别训练，再用相同的评估种子评估。 |
| `plot_rewards.py` | 绘制每局训练奖励及滑动平均曲线。 |
| `check_returns.py` | 检查折扣回报的计算结果。 |

## 运行

需要安装 Python、PyTorch、Gymnasium、Matplotlib 和 Pygame（用于评估画面）。先进入项目目录：

```bash
cd twice_step/reinforce_cartpole
```

训练并保存模型参数到 `policy_weights.pt`，训练曲线保存为 `train_rewards.png`：

```bash
python train.py
```

评估已保存的模型，并显示 CartPole 画面（需要先有与当前网络结构匹配的 `policy_weights.pt`）：

```bash
python evaluate.py
```

分别用多个训练种子训练和评估，曲线保存为 `experiment_rewards.png`：

```bash
python experiments.py
```

多种子实验只输出结果，不保存各次训练的模型。训练参数在 `config.py` 中修改；实验的训练种子与评估种子在 `experiments.py` 中修改。`evaluate.py` 的独立运行入口目前单独指定环境和评估种子。

曲线中的浅色线是每局训练时的原始奖励，深色线是最近 50 局的平均奖励（前 50 局取已有局数的平均）。训练奖励来自采样动作的策略，与评估时使用贪心动作得到的奖励不同。曲线可以帮助观察变化，但单条曲线不能证明训练稳定收敛。

检查折扣回报：

```bash
python check_returns.py
```

训练按动作概率采样；评估时选择 logits 最大的动作。轨迹因终止或时间限制结束时，当前实现都只计算已采集到的奖励，不对截断后的回报进行价值估计。
