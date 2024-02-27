# PyTorch introduction

import torch
import numpy as np

x = torch.tensor([[1,2,3], [4,5,6]])
y = torch.tensor([[7,8,9], [10,11,12]])
r = x + y
print(r)

shape = [2,3]
xzeros = torch.zeros(shape)
xones = torch.ones(shape)
xrnd = torch.rand(shape)

print(xzeros)
print(xones)
print(xrnd)

torch.manual_seed(42)
print(torch.rand([2,3]))

#Converting between tensors and NumPy arrays

xnp = np.array([[1,2,3],[4,5,6]])
'''f2 = xnp + y
print(f2)
f2.type()'''

xtensor = torch.from_numpy(xnp)
print(xtensor)
print(xtensor.type())

print()

