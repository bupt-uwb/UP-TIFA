import numpy as np
import torch

# a = np.array([[np.random.randint(1, 4) for j in range(1, 5)] for i in range(1, 11)])
# print(a.shape)
# b = np.array([[np.random.randint(1, 4) for j in range(1, 5)] for i in range(1, 11)])
# print(b.shape)
# c = np.array([a,b]).reshape((-1, 2, 4))
# print(c.shape)
# a = torch.randn(16,1,500)
# # print(a.shape)
# b = torch.sum(torch.pow(a,2),dim=(2,), keepdim=True)
# # print(b.shape)
# # c = torch.sum(b,dim=2)
# # print(c.shape)

# A是array([ 0,  1,  2,  3,  4,  5,  6,  7,  8,  9, 10, 11, 12, 13, 14, 15])
# A = np.arange(24)
#
# # 将A变换为三维矩阵
# A = A.reshape(2, 3, 4)
# print(A)
# B = A.transpose((1,0,2))
# print(B[:,1,:])
#
# a = [1]
# b = a
# print(a,b)
# a.append(2)
# print(a,b)
a=2
b='+'
c='2'
print(f'{a}{b}{c}')
# print(eval(a b c))