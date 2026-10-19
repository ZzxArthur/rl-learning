# 练习7：使用nn.Linear和SGD训练一元线性回归，手写带形状检查的均方误差。
# 验证新输入预测和已知MSE结果，并用形状不匹配的输入检查ValueError。

import torch


def mse_loss(y, t):
    if y.shape == t.shape:
        squared_error = (t-y)**2
        num_elements = y.numel()
        loss = squared_error.sum() / num_elements
        return loss
    else:
        raise ValueError("y 与 t 的形状必须一致")
        

x = torch.tensor([[0], [1], [2], [3]], dtype = torch.float32)
t = torch.tensor([[1], [3], [5], [7]], dtype = torch.float32)
step = 0
model = torch.nn.Linear(in_features=1, out_features=1)
optimizer = torch.optim.SGD(model.parameters(), lr = 0.2)
y = model(x)
loss = mse_loss(y, t)
while loss > 1e-6 and step < 100:
    loss.backward()
    with torch.no_grad():
        optimizer.step()
        optimizer.zero_grad()
    y = model(x)
    loss = mse_loss(y, t)
    step += 1

print(loss)
print(y)
x1 = torch.tensor([[4], [5]], dtype=torch.float32)
t1 = torch.tensor([[9], [11]], dtype=torch.float32)
with torch.no_grad():
    y1 = model(x1)
    loss1 = mse_loss(y1, t1)

print(y1)
print(loss1)

y2 = torch.tensor([[1], [3]], dtype=torch.float32)
t2 = torch.tensor([[2], [1]], dtype=torch.float32)

print(mse_loss(y2, t2))

y3 = torch.randn([4, 1])
t3 = torch.randn([4, ])

print(mse_loss(y3, t3))












