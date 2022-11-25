import torch
import numpy as np



"""proto_dones = [[0, 0, 0, 1], [0, 0, 0, 1]]
proto_dones = np.array(proto_dones)
# Tensor random de 2 filas y 10 columnas, unroll_size = 10 (columnas)
unroll_size = 4
gamma = 0.99
advantages = torch.zeros((2, unroll_size))
reward = torch.randint(0, 10, (2, unroll_size))
value = torch.randint(0, 10, (2, unroll_size))
done = torch.from_numpy(proto_dones)
print("rewards:", reward)
print("values:", value)

# Por cada paso en el batch    WORKING!
for t in range(unroll_size - 1):
    discount = 1
    advantage_t = 0
    for k in range(t, unroll_size - 1):
        advantage_t += discount*(reward[:, k:k+1] + gamma * value[:, k+1:k+2] * (1 - done[:, k:k+1]) - value[:, k:k+1])
        discount *= gamma
    advantages[:, t:t+1] = advantage_t

print("advantages:", advantages)"""

x = torch.randint(0, 10, (1, 1, 10, 10))
print("x shape:", x.size())
print(x)
print("x sliced shape:", x[0, 0, 0:5, :].size())
print(x[0, 0, 0:5, :])
