# 练习13：固定初始化，对照2→8→1网络保留或绕过ReLU时的XOR拟合效果。
# 使用MSE和SGD训练；当前前向绕过ReLU，文件末尾记录两组结果。
import torch
import torch.nn as nn


class EzNet(nn.Module):
    def __init__(self):
        super().__init__()
        torch.manual_seed(42)
        self.w1 = nn.Parameter(torch.randn(2, 8))
        self.b1 = nn.Parameter(torch.randn(1, 8))
        self.w2 = nn.Parameter(torch.randn(8, 1))
        self.b2 = nn.Parameter(torch.randn(1, 1))

    def forward(self, x):
        a1 = x @ self.w1 + self.b1 
        a2 = nn.ReLU()(a1)
        a2 = a1
        y1 = a2 @ self.w2 + self.b2  # 曾经因为a2 = nn.ReLU(a1)报错, 因为nn.ReLU是一个类, 需要调用它才能生效.TypeError: unsupported operand type(s) for @: 'ReLU' and 'Parameter'

        return y1


x = torch.tensor(
    [[0, 0], [1, 0], [0, 1], [1, 1]],
    dtype=torch.float32,
)

t = torch.tensor(
    [[0], [1], [1], [0]],
    dtype=torch.float32,
)

step = 1
max_step = 5000
lr = 0.02

eznet = EzNet()
optimizer = torch.optim.SGD(eznet.parameters(), lr=lr)

y1 = eznet(x)
train_loss = torch.nn.functional.mse_loss(y1, t)
while train_loss.max() > 1e-6 and step < max_step:
    train_loss.backward()
    with torch.no_grad():
        optimizer.step()
        optimizer.zero_grad()
    y = eznet(x)
    train_loss = torch.nn.functional.mse_loss(y, t)
    step += 1
    if step == max_step:
        print("达到更新次数上线")
    elif train_loss.max() <= 1e-6:
        print("损失达标")
        print(step)


print(train_loss)
print(y)
# tensor(9.7663e-07, grad_fn=<MseLossBackward0>)
# tensor([[-8.0697e-04],
#         [ 1.0012e+00],
#         [ 1.0007e+00],
#         [-1.0781e-03]], grad_fn=<AddBackward0>)
# tensor(0.2500, grad_fn=<MseLossBackward0>)
# tensor([[0.5000],
#         [0.5000],
#         [0.5000],
#         [0.5000]], grad_fn=<AddBackward0>)
