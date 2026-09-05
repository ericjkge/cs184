import math
import matplotlib.pyplot as plt
import numpy as np

n = math.ceil(1/(4 * 0.0025))

print(n)

rng = np.random.default_rng()
F_hat = np.arange(1, n + 1) / n

for k in [1, 8, 64, 512]:
    B = rng.choice([-1, 1], size=(n, k))
    Y = B.sum(axis=1) / np.sqrt(k)
    Y_sorted = np.sort(Y)
    plt.step(Y_sorted, F_hat, where="post", label=str(k))

Z = rng.standard_normal(n)
Z_sorted = np.sort(Z)
plt.step(Z_sorted, F_hat, where="post", label="Gaussian")

plt.xlim(-3, 3)
plt.ylim(0, 1)
plt.xlabel("Observations")
plt.ylabel("Probability")
plt.legend()
plt.show()