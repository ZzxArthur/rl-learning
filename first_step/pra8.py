# 练习8：使用两个输入特征训练线性回归，手写带形状检查的均方误差。
# 使用SGD更新参数并报告停止原因，在no_grad中验证新样本及逐项绝对误差。

import torch


def mse_loss(y, t):
    if y.shape == t.shape:
        squared_error = (t-y)**2
        num_elements = y.numel()
        loss = squared_error.sum() / num_elements
        return loss
    else:
        raise ValueError("y与t的形状必须一样")


x = torch.tensor([[0, 0], [1, 0], [0, 1], [1, 1]], dtype=torch.float32)
t = torch.tensor([[1], [3], [-2], [0]], dtype=torch.float32)
step = 0
max_step = 400
model = torch.nn.Linear(in_features = 2, out_features = 1)
optimizer = torch.optim.SGD(model.parameters(), lr = 0.2)

y = model(x)
loss = mse_loss(y, t)

while loss > 1e-6 and step < max_step:
    loss.backward()
    with torch.no_grad():
        optimizer.step()
        optimizer.zero_grad()
    y = model(x)
    loss = mse_loss(y, t)
    step += 1
    if step >= max_step:
        print("达到更新次数上线")
    elif loss <= 1e-6:
        print("损失达标")

x1 = torch.tensor([[2, 1], [-1, 1]], dtype = torch.float32)
t1 = torch.tensor([[2], [-4]], dtype = torch.float32)
with torch.no_grad():

    y1 = model(x1)
    loss1 = mse_loss(y1, t1)
print(loss1)
print(abs(t1 - y1))



