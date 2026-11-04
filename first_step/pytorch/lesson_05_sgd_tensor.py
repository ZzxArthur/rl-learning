# 练习5：使用SGD优化单个可训练张量，使x逐步接近目标值3。
# 观察损失、参数和梯度，练习backward、step与zero_grad的分工。

import torch 

#optimizer = torch.optim.SGD(model.parameters(), lr=learning_rate)
#第一个参数 params 是要优化的参数张量集合。
def f(x):  #即损失函数 任务即调整x,让他尽量接近3,使得损失函数的值尽量小
    return (x-3)**2


x = torch.tensor([1], dtype=torch.float32, requires_grad=True)
step = 0
optimizer = torch.optim.SGD([x], lr=0.1)
while loss := f(x) > 1e-6 and step < 100:
    loss = f(x)
    loss.backward()  # 计算梯度
    print(f"loss:{loss}, x:{x}, x.grad:{x.grad}, step:{step}")
    with torch.no_grad():
        optimizer.step()
        optimizer.zero_grad()
    step += 1
