# 练习2：观察张量的内容、shape和ndim，学习逐元素相加的广播规则。
# 当前形状(3, 1)与(4, 1)不兼容，执行相加时会产生广播报错。

import torch
# tensor用法  tensor([1, 2, 3]) 只有一个列表相当于只有一个轴, [[1,2], [2,3]] 内层有两个列表相当于有俩行,每个列表有俩个数 相当于2列
a = torch.tensor([[1], [2], [3]])  #相当于3行1列
b = torch.tensor([[1], [2], [3], [4]])  #相当于一个轴,可以是行可以是列
print(a)
print(b)
print(a.shape)
print(b.shape)
print(a.ndim)
print(b.ndim)
print(a + b)     # 广播会自动补齐维度 
print(b + a)
