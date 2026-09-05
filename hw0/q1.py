import numpy as np

A = np.array([[0, 2, 4], [2, 4, 2], [3, 3, 1]])
b = np.array([-2, -2, -4])
c = np.array([1, 1, 1])

A_inv = np.linalg.inv(A)

print(A_inv)
print(A_inv @ b)
print(A @ c)