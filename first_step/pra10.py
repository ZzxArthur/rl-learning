# 练习10：模型类保存权重和偏置并预测，类外函数负责损失计算和参数更新。
# 使用双特征单样本练习自动求导训练，并对新输入执行预测。
import torch


class NetWork:
    def __init__(self):

        self.w = torch.randn([2, 1], requires_grad=True)
        self.b = torch.randn([1, 1], requires_grad=True)

    def f(self, x):
        y = x @ self.w + self.b
        return y


def mse_loss(y, t):
    if y.shape == t.shape:
        squard_error = (t - y)**2
        num = y.numel()
        train_loss = squard_error.sum()/num
        return train_loss
    else:
        raise ValueError("t与y的维度需要保持一致!") 


def update(train_loss, optim):
    train_loss.backward()
    with torch.no_grad():
        optim.step()
        optim.zero_grad()

      
if __name__ == "__main__":
    # 这里只在直接运行文件时执行
    x = torch.tensor([[1, 2]], dtype=torch.float32)
    t = torch.tensor([[2]], dtype=torch.float32)
    learning_rate = 0.04
    step = 0
    max_step = 400
    predict = NetWork()
    y = predict.f(x)
    train_loss = mse_loss(y, t)
    optimizer = torch.optim.SGD([predict.w, predict.b], lr=learning_rate)
    while train_loss > 1e-6 and step < max_step:
        update(train_loss, optimizer)
        y = predict.f(x)
        train_loss = mse_loss(y, t)
        step += 1

    print(train_loss)
    print(y)

    eval_x = torch.tensor([[3, 4]], dtype=torch.float32)
    eval_y = predict.f(eval_x)
    print(eval_y)





