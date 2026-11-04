# 练习4：使用自动求导和手动参数更新，使x逐步接近目标值3。
# 在no_grad中更新参数并清理梯度，按损失阈值或100次更新上限结束。

import torch
def f(x):  #即损失函数 任务即调整x,让他尽量接近3,使得损失函数的值尽量小
    return (x-3)**2

def grad(x):
    return 2*(x-3)


x = torch.tensor([1], dtype=torch.float32, requires_grad=True)
step = 0
while loss := f(x) > 1e-6 and step < 100:  # 当损失函数的值大于1e-6时,继续迭代或者达到最大迭代次数100次,停止迭代
    loss = f(x)
    loss.backward()  # 计算梯度
    print(f"x: {x}, loss: {loss}, grad: {x.grad}")  # 打印当前x的值,损失函数的值,梯度的值
    with torch.no_grad():  # 不需要计算梯度,不需要记录梯度
        x -= 0.1 * x.grad  # 梯度下降法更新x的值,学习率为0.1
        x.grad.zero_()  # 清零梯度,否则梯度会累加
    step += 1  # 每次迭代

print(step)
