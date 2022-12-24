import torch
import numpy as np


dicty = {"obs": torch.rand(12, 1), "values": torch.rand(12, 1), "actions": torch.rand(12, 1)}


for key in dicty:
    print(f"{key}: {dicty[key]}")

x = {key: tensor[1:] for key, tensor in dicty.items()}

print("Equivalente al batch:")
print(x)

x = {key: tensor[:-1] for key, tensor in dicty.items()}

print("Equivalente al learner:")
print(x)


print("Dones:")
x = [False, False, False, False, True]
x = torch.from_numpy(np.array(x))
print(x)
x = (~x).float() * 0.99
print(x)


