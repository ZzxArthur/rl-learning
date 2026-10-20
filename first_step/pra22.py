# 练习22：用状态转移表实现简化step接口，并采集按时间排列的五元组轨迹。
# 每次从S0开始，保存终止转移后停止；终止状态继续执行动作时抛出异常。
import torch
# 轨迹需要按时间保留每一次交互。字典用状态作键，同一个状态再次出现时会覆盖旧记录，因此这里使用列表。
def collect_trajectory(actions):
    trajectory = []
    initial_state = 'S0'
    for i in actions:
        next_state, reward, terminated = step(initial_state, i)
        if terminated == False:
            trajectory.append((initial_state, i, reward, next_state, terminated))
            initial_state = next_state
        else:
            trajectory.append((initial_state, i, reward, next_state, terminated))
            break

    return trajectory

def step(state, action):
    """根据当前状态和动作，返回下一状态、即时奖励和是否结束"""
    if state == "T":
        raise ValueError("已到达终止状态，无法继续执行动作。")
    else:
        return transitions[(state, action)]
# 键：(当前状态, 动作)
# 值：(下一状态, 即时奖励, 是否结束)
transitions = {
    ("S0", 0): ("S1", 1.0, False),
    ("S0", 1): ("T", 0.0, True),
    ("S1", 0): ("T", 0.0, True),
    ("S1", 1): ("T", 2.0, True),
}

initial_state = "S0"
actions_a = [0, 1]  # 本次使用 Python 整数动作编号
state1_a = step(initial_state, actions_a[0])  # 执行动作0，返回下一状态、即时奖励和是否结束
state2_a = step(state1_a[0], actions_a[1])  # 执行动作1，返回下一状态、即时奖励和是否结束
trajectory_a = collect_trajectory(actions_a)
actions_b = [1, 0]  # 本次使用 Python 整数动作编号
state1_b = step(initial_state, actions_b[0])  # 执行动作0，返回下一状态、即时奖励和是否结束

trajectory_b = collect_trajectory(actions_b)

print(trajectory_a)
print(trajectory_b)
