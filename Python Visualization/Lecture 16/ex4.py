import numpy as np
import matplotlib.pyplot as plt

# Collocation polynomial coefficients
x1, x2, x3 = 0, -0.5, 1.5

# Approximate polynomial solution
t = np.linspace(0, 1, 50)
u_collocation = x1 + x2 * t + x3 * t**2

# Exact analytical solution: u'' = 6t -> u = t^3 + C1*t + C2
# Apply boundary conditions: u(0)=0, u(1)=1
# => C2=0, C1=0 => u = t^3
u_exact = t**3

# Plot both
plt.figure(figsize=(7,5))
plt.plot(t, u_exact, 'k--', label='Exact Solution $u(t)=t^3$')
plt.plot(t, u_collocation, 'o-', color='tab:blue', label='Collocation Approximation $u(t)=-0.5t+1.5t^2$')
plt.scatter([0, 0.5, 1], [-0.5*0+1.5*0**2, -0.5*0.5+1.5*0.5**2, -0.5*1+1.5*1**2], color='red', zorder=5)
plt.title("Collocation Method with Polynomial Basis (1 Interior Point)")
plt.xlabel("$t$")
plt.ylabel("$u(t)$")
plt.legend()
plt.grid(True)
plt.show()
