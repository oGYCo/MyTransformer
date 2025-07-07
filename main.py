import torch # 引入 PyTorch 库

# 1. 创建张量 (就像在C里声明和初始化一个多维数组)
# 相当于 C 的 int arr[2][3] = {{1,2,3}, {4,5,6}};
x = torch.tensor([[1, 2, 3],
                  [4, 5, 6]])
print("张量 x:\n",x)

# 2. 查看形状和类型 (非常重要的调试手段)
print("x 的形状:", x.shape)   # 输出 torch.Size([2, 3])
print("x 的数据类型:", x.dtype) # 输出 torch.int64

# 3. 创建特定形状的张量
zeros_tensor = torch.zeros(3, 4) # 创建一个 3x4 的全零张量
ones_tensor = torch.ones(3, 4)   # 创建一个 3x4 的全一张量
rand_tensor = torch.rand(3, 4)   # 创建一个 3x4 的随机张量

# 4. 核心：向量化计算 (这是与C最大的不同)
# 在C里，两个矩阵相加需要写嵌套 for 循环。
# 在 PyTorch 里，直接用 + 即可，它会自动在底层用高效代码(或GPU)完成。
a = torch.tensor([[1, 1], [1, 1]])
b = torch.tensor([[2, 2], [2, 2]])
c = a + b # 元素级相加
print("a + b:\n", c)

# 矩阵乘法 (在 Transformer 中无处不在)
d = torch.matmul(a, b) # 或者用 @ 符号: d = a @ b
print("a @ b:\n", d)