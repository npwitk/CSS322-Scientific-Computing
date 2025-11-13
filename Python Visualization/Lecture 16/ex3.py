import numpy as np
import matplotlib.pyplot as plt

# Given coefficients and constants from the centered difference system
A_centered = np.array([
    [-14.25, 1.5, 0],
    [7, -17, 1],
    [0, 7.5, -20.25]
], dtype=float)

b_centered = np.array([13, 0, 0.5], dtype=float)
y_centered = np.linalg.solve(A_centered, b_centered)

# For backward difference (alternative method)
A_backward = np.array([
    [-17.25, 1, 0],
    [7, -18, 1],
    [0, 7.5, -21.25]
], dtype=float)

b_backward = np.array([7, 0, 0.5], dtype=float)
y_backward = np.linalg.solve(A_backward, b_backward)

# Combine with boundary conditions
y0, y4 = 2, -1
t = np.array([2, 2.5, 3, 3.5, 4])
u_centered = np.array([y0, *y_centered, y4])
u_backward = np.array([y0, *y_backward, y4])

# Plot results
plt.figure(figsize=(8,5))
plt.plot(t, u_centered, 'o-', label='Centered Difference (Recommended)', color='tab:blue')
plt.plot(t, u_backward, 's--', label='Backward Difference (For Illustration)', color='tab:orange')
plt.title("Finite Difference Method (More Complex BVP)")
plt.xlabel("$t$")
plt.ylabel("$u(t)$")
plt.grid(True)
plt.legend()
plt.show()
