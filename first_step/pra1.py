# 练习1：手写平方损失、解析梯度和梯度下降更新，观察参数向目标值3靠近。
# 当前初值为3，损失已经达标，因此不会进入更新循环。

def f(x):
    return (x-3)**2


def grad(x):
    return 2*(x-3)


def update(x, learning_rate=0.1):
    return x - learning_rate * grad(x)


x = 3
while f(x) > 1e-6:
    x = update(x)
    print(f"x: {x}, f(x): {f(x)}")


