# 练习6：用共享的权重和偏置拟合一元线性关系，手写前向计算与均方误差。
# 使用SGD更新参数、清理梯度，并同步重算预测和损失以判断停止条件。
# 训练后在no_grad中复用参数预测新样本，比较预测值与目标值。

import torch
import math


def f(x, w, b):
    return x * w + b


def mse_loss(y, t):
    a = (t-y)**2
    len = y.flatten().shape[0]
    loss = a.sum() / len
    return loss


x = torch.tensor([[0], [1], [2], [3]], dtype = torch.float32)
t = torch.tensor([[1], [3], [5], [7]], dtype = torch.float32)
w = torch.randn(1, 1, requires_grad = True)
b = torch.randn(1, 1, requires_grad = True)
print(x.shape)
optimizer = torch.optim.SGD([w, b], lr = 0.02)  # 优化器 是用来改参数的
step = 0
y = f(x, w, b)
loss = mse_loss(y, t)
while loss > 1e-6 and step < 1000:
    loss.backward()
    print(loss)
    with torch.no_grad():
        optimizer.step()
        optimizer.zero_grad()
    y = f(x, w, b)
    loss = mse_loss(y, t)
    step += 1

new_x = torch.tensor([[4], [5]], dtype=torch.float32)
new_t = torch.tensor([[9], [11]], dtype=torch.float32)
with torch.no_grad():
    new_y = f(new_x, w, b)
    new_loss = mse_loss(new_y, new_t)

print(new_y)
print(new_loss)







