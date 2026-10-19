# 练习9：将双特征线性回归的完整训练流程封装为train函数。
# 显式传入模型和优化器，练习参数副作用、停止判断及最新预测与损失的返回。
# 使用no_grad评估新样本的误差，并通过权重、偏置快照检查参数不变。

import torch


def mse_loss(y, t):
    if y.shape == t.shape:
        squared_error = (t-y)**2
        num_elements = y.numel()
        loss = squared_error.sum() / num_elements
        return loss
    else:
        raise ValueError("y与t的形状必须一样")

    
def upgrade(train_loss, optimizer):
    # 做一次反向传播更新梯度,然后根据梯度更新需要优化的参数
    train_loss.backward()
    with torch.no_grad():
        optimizer.step()
        optimizer.zero_grad()


def train(x, model, t, optimizer):
    y = model(x)
    train_loss = mse_loss(y, t)
    step = 0
    max_step = 400
    while train_loss > 1e-6 and step < max_step:
        upgrade(train_loss, optimizer)
        y = model(x)
        train_loss = mse_loss(y, t)
        step += 1
        if step == max_step:
            print("达到更新次数上线")
        elif train_loss <= 1e-6:
            print("损失达标")
            print(step)
    return y, train_loss


def abs_loss(y, t):
    return abs(t - y)


x = torch.tensor([[0, 0], [1, 0], [0, 1], [1, 1]], dtype=torch.float32)
t = torch.tensor([[1], [3], [-2], [0]], dtype=torch.float32) 

learning_rate = 0.08
model = torch.nn.Linear(in_features=2, out_features=1)
y = model(x)
optimizer = torch.optim.SGD(model.parameters(), lr=learning_rate)
y, train_loss = train(x, model, t, optimizer)


print(train_loss)
print(abs(t - y))

x1 = torch.tensor([[2, 1], [-1, 1]], dtype=torch.float32)
t1 = torch.tensor([[2], [-4]], dtype=torch.float32)

w = model.weight
b = model.bias

w1 = w.detach().clone()  # detach()表示脱离计算图,但仍共享原来的数据
b1 = b.detach().clone()  # clone()表示复制数据 得到独立副本
with torch.no_grad():
    y1 = model(x1)
    print(abs_loss(y1, t1))
    print(mse_loss(y1, t1))

w2 = w.detach().clone() + 1
print(w)
print(w2)
print(torch.equal(w, w1), torch.equal(b, b1), torch.equal(w, w2))  # equal()比较形状和元素是否相同
