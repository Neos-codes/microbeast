import numpy as np


array = np.array([1, 1, 0, 1, 0, 0])

# np,where retorna una lista de indices donde se cumple la condicion
array[np.where(array == 0)] = 2
print(array)   # Imprime solo los 1's
