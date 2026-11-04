# 练习3：使用autograd计算平方损失的梯度，区分参数值和存储的梯度。
# 观察反向传播前后的.grad，以及清零后重新计算梯度的结果。

import torch
def f(x):
    return (x-3)**2

def grad(x):
    return 2*(x-3)

x = torch.tensor([1.0], dtype=torch.float32, requires_grad=True)
print(x.grad)
loss = f(x)
print(loss)
loss.backward()
print(x.grad)
print(x)
x.grad.zero_()  # 清零梯度
print(x.grad)

loss = f(x)
loss.backward()
print(x.grad)   # 梯度值会累加

