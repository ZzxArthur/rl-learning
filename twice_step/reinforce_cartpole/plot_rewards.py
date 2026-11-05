"""绘制每局训练奖励及其滑动平均。"""
from itertools import accumulate

import matplotlib.pyplot as plt


def plot_rewards(rewards_by_seed, save_path, window=50):
    if window < 1:
        raise ValueError('window must be positive')

    fig, ax = plt.subplots()
    for seed, rewards in rewards_by_seed.items():
        if not rewards:
            continue
        episodes = range(1, len(rewards) + 1)
        cumulative = [0, *accumulate(rewards)]
        moving_average = [
            (cumulative[end] - cumulative[max(0, end - window)]) / min(end, window)
            for end in episodes
        ]
        line, = ax.plot(episodes, moving_average, label=f'Seed {seed}: moving average')
        ax.plot(episodes, rewards, color=line.get_color(), alpha=0.15, linewidth=0.7)

    ax.set_xlabel('Training episode')
    ax.set_ylabel('Episode reward')
    ax.set_title(f'Training rewards (moving average: {window} episodes)')
    ax.grid(alpha=0.25)
    if rewards_by_seed and any(rewards_by_seed.values()):
        ax.legend()
    fig.tight_layout()
    fig.savefig(save_path, dpi=150)
    plt.close(fig)
    print(f'训练奖励曲线已保存: {save_path}')
