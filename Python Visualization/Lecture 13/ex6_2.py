import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import solve_ivp

# ------------------------------
# Define the system dy/dt = f(t, y)
# ------------------------------
def f(t, y):
    y1, y2 = y
    dy1 = t * y2 - y1
    dy2 = 2 * (y1**2) * y2 + t**2
    return [dy1, dy2]

# ------------------------------
# Euler's Method implementation
# ------------------------------
def euler_system(f, y0, t0, h, steps):
    t = [t0]
    y = [y0]
    for _ in range(steps):
        y_next = y[-1] + h * np.array(f(t[-1], y[-1]))
        t.append(t[-1] + h)
        y.append(y_next)
    return np.array(t), np.array(y)

# ------------------------------
# Parameters
# ------------------------------
y0 = np.array([-2.0, 1.0])
t0 = 0.0
h = 0.2
steps = 2
t_end = t0 + steps * h

# ------------------------------
# High-accuracy reference solution (for "true" curve)
# ------------------------------
sol = solve_ivp(f, [t0, t_end], y0, t_eval=np.linspace(t0, t_end, 200))

# ------------------------------
# Run Euler method
# ------------------------------
t_vals, y_vals = euler_system(f, y0, t0, h, steps)

# ------------------------------
# Print intermediate results
# ------------------------------
print("Euler Steps:")
for i, (t, y) in enumerate(zip(t_vals, y_vals)):
    print(f"Step {i}: t = {t:.2f}, y1 = {y[0]:.4f}, y2 = {y[1]:.4f}")

# ------------------------------
# Visualization
# ------------------------------
fig, axes = plt.subplots(1, 2, figsize=(12, 5))

# ---- Plot 1: Phase Plane (y1 vs y2)
axes[0].plot(sol.y[0], sol.y[1], 'g--', label='True Solution (solve_ivp)')
axes[0].plot(y_vals[:,0], y_vals[:,1], 'ro-', label="Euler's Method")
for i in range(len(y_vals)):
    axes[0].text(y_vals[i,0], y_vals[i,1], f'{i}', fontsize=9, ha='left', va='bottom')
axes[0].set_xlabel('$y_1$')
axes[0].set_ylabel('$y_2$')
axes[0].set_title('Phase Plane: $y_2$ vs $y_1$')
axes[0].grid(True, linestyle='--', alpha=0.6)
axes[0].legend()

# ---- Plot 2: Time evolution
axes[1].plot(sol.t, sol.y[0], 'g--', label='True $y_1(t)$')
axes[1].plot(sol.t, sol.y[1], 'b--', label='True $y_2(t)$')
axes[1].plot(t_vals, y_vals[:,0], 'ro-', label="Euler $y_1$")
axes[1].plot(t_vals, y_vals[:,1], 'ms--', label="Euler $y_2$")
axes[1].set_xlabel('t')
axes[1].set_ylabel('Values')
axes[1].set_title('Euler vs. True Solution (Time Evolution)')
axes[1].legend()
axes[1].grid(True, linestyle='--', alpha=0.6)

plt.tight_layout()
plt.show()
