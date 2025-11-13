import numpy as np

A = np.array([[3, -1],
              [-1, 3]])

x = np.array([1., 0.])  # initial vector

print("k | x_k (vector) | approx ratio")
print("-" * 40)

for k in range(1, 11):
    x = A @ x                     # multiply without normalization
    ratio = x[0] / (x[1] if x[1] != 0 else np.nan)
    print(f"{k:2d} | [{x[0]:.0f}, {x[1]:.0f}] | ratio ≈ {ratio:.6f}")

