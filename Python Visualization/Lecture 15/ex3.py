import numpy as np

A = np.array([[3., -1.],
              [-1., 3.]])

x = np.array([1., 0.])                 # initial vector
x = x / np.linalg.norm(x)

for k in range(1, 8):
    y = A @ x
    lam = (x @ y) / (x @ x)            # Rayleigh quotient using current x
    x = y / np.linalg.norm(y)          # normalize to avoid overflow
    print(f"iter {k}: λ≈{lam:.10f}, x≈{x}")
