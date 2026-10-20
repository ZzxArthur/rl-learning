# 练习12：用nn.Module和nn.Parameter注册线性模型的权重与偏置。
# 通过parameters()将参数交给SGD管理，使用MSE训练双特征线性回归。

import torch
import torch.nn as nn
import torch.nn.functional as F

class EzNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.w = nn.Parameter(torch.randn(2, 1))
        self.b = nn.Parameter(torch.randn(1, 1))

    def forward(self, x):
        y = x @ self.w + self.b
        return y 


x = torch.tensor(
    [[0, 0], [1, 0], [0, 1], [1, 1]],
    dtype=torch.float32,
)

t = torch.tensor(
    [[1], [3], [-2], [0]],
    dtype=torch.float32,
)
step = 1
max_step = 400
lr = 0.04
eznet = EzNet()
optimizer = torch.optim.SGD(eznet.parameters(), lr=lr)
y = eznet.forward(x)
train_loss = F.mse_loss(y, t)
while train_loss > 1e-6 and step < max_step:
    train_loss.backward()
    with torch.no_grad():
        optimizer.step()
        optimizer.zero_grad()
    y = eznet.forward(x)
    train_loss = F.mse_loss(y, t)
    step += 1

print(train_loss)
print(y)
