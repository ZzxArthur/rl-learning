# 练习53：用采样回报训练状态价值模型，并检查状态与回报的逐项对应。
import torch

# 6 个状态，形状 (6, 2)，float32
states = torch.tensor([
    [1.0, 0.0],  # 第 1 局第 1 步
    [0.0, 1.0],  # 第 1 局第 2 步
    [1.0, 0.0],  # 第 2 局第 1 步
    [0.0, 1.0],  # 第 2 局第 2 步
    [1.0, 0.0],  # 第 3 局第 1 步
    [0.0, 1.0],  # 第 3 局第 2 步
], dtype=torch.float32)

# 对应每一步的折扣回报，形状 (6,)，float32 采样回报
returns = torch.tensor(
    [3.7, 3.0, 5.6, 4.0, 4.6, 4.0],
    dtype=torch.float32,
)

torch.manual_seed(42)
value_model = torch.nn.Linear(2, 1)
optimizer = torch.optim.SGD(value_model.parameters(), lr=0.05)
num_iterations = 200

for _ in range(num_iterations):
    states_predict_values = value_model(states).squeeze(-1)  #形状与returns匹配
    print(states_predict_values)
    loss = torch.mean((states_predict_values - returns) ** 2)
    print(loss)
    loss.backward()
    with torch.no_grad():
        optimizer.step()
        optimizer.zero_grad()
with torch.no_grad():
    states_predict_values = value_model(states).squeeze(-1)
    loss = torch.mean((states_predict_values - returns) ** 2)
    print('最终 MSE：', loss.item())
    print('两个状态的预测价值：', states_predict_values[:2])
