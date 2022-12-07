import torch
import numpy as np

x = torch.rand((4, 20))

print(x.min(axis=0)[0])

arr_1 = torch.Tensor(np.array([2, 3, 4, 5]))
arr_2 = torch.Tensor(np.array([3, 4, 5, 6]))

print(arr_1*arr_2)



