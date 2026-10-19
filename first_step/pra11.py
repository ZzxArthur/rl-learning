# 练习11：单样本双特征线性模型的前向预测与手写均方误差梯度。
# 输入x为(1, 2)、目标t为(1, 1)，backward返回梯度，不更新参数。
import torch
import torch.nn.functional as F


class EzNet:
    def __init__(self):
        self.w1 = torch.randn([1, 1], requires_grad=True)
        self.w2 = torch.randn([1, 1], requires_grad=True)
        self.b  = torch.randn([1, 1], requires_grad=True)


    def predict(self, x):
        y = x[:, 0] * self.w1 + x[:, 1] * self.w2 + self.b
        return y

    def loss(self, x, t):
        y = self.predict(x)
        return F.mse_loss(y, t)

    def backward(self, x, y, t):
        dw1 = 2 * (y - t) * x[:, 0]
        dw2 = 2 * (y - t) * x[:, 1]
        db  = 2 * (y - t) * 1
        return [dw1, dw2, db]


eznet = EzNet()
x = torch.tensor([[1, 2]], dtype=torch.float32)
t = torch.tensor([[3]], dtype=torch.float32)
y = eznet.predict(x)
dw1, dw2, db = eznet.backward(x, y, t)
optimizer = torch.optim.SGD([eznet.w1, eznet.w2, eznet.b], lr=1)
train_loss = eznet.loss(x, t)
train_loss.backward()

print(eznet.w1.grad, eznet.w2.grad, eznet.b.grad)
print(torch.allclose(dw1, eznet.w1.grad), torch.allclose(dw2, eznet.w2.grad), torch.allclose(db, eznet.b.grad))



