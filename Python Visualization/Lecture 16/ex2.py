import numpy as np
import matplotlib.pyplot as plt

# Mesh points
t_fd = np.array([0, 0.5, 1.0])
y0, y2 = 0, 1
y1 = 1/8  # From finite difference calculation
u_fd = np.array([y0, y1, y2])

# Continuous (exact) solution
t_real = np.linspace(0, 1, 200)
u_real = t_real**3  # Exact analytical solution

# Plot both
plt.figure(figsize=(7,5))
plt.plot(t_real, u_real, 'k--', label='Exact Solution $u(t) = t^3$')
plt.plot(t_fd, u_fd, 'o-', color='tab:blue', label='Finite Difference (1 interior point)')
plt.scatter(0.5, y1, color='red', s=60, label=f'Approx at $t=0.5$: {y1:.3f}')
plt.title('Finite Difference vs Real Solution')
plt.xlabel('$t$')
plt.ylabel('$u(t)$')
plt.legend()
plt.grid(True)
plt.show()
