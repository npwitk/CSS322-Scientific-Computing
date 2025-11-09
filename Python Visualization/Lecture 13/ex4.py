import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation

# ---------------------------------------------
# Define the ODE dy/dt = t / y, and true solution
# ---------------------------------------------
def f(t, y):
    return t / y

def true_solution(t):
    # Analytical solution: y^2 = t^2/2 + 1
    return np.sqrt(t**2 / 2 + 1)

# ---------------------------------------------
# Heun's Method Implementation
# ---------------------------------------------
def heun_method(f, y0, t0, h, n):
    t_values = [t0]
    y_values = [y0]
    s1_list, s2_list = [], []

    for k in range(n):
        t_k = t_values[-1]
        y_k = y_values[-1]
        s1 = f(t_k, y_k)
        s2 = f(t_k + h, y_k + h*s1)
        y_next = y_k + (h/2)*(s1 + s2)
        y_values.append(y_next)
        t_values.append(t_k + h)
        s1_list.append(s1)
        s2_list.append(s2)

    return np.array(t_values), np.array(y_values), s1_list, s2_list

# ---------------------------------------------
# Parameters
# ---------------------------------------------
y0 = 1
t0 = 0
h = 0.5
t_end = 2
n = int((t_end - t0) / h)

# Compute Heun results
t_vals, y_vals, s1_vals, s2_vals = heun_method(f, y0, t0, h, n)

# Compute true solution
t_true = np.linspace(t0, t_end, 200)
y_true = true_solution(t_true)

# ---------------------------------------------
# Set up figure
# ---------------------------------------------
fig, ax = plt.subplots(figsize=(7,5))
ax.plot(t_true, y_true, 'g--', label='True Solution')
num_line, = ax.plot([], [], 'ro-', label="Heun's Method")
ax.set_xlim(t0, t_end)
ax.set_ylim(0.8, 1.6)
ax.set_xlabel('t')
ax.set_ylabel('y(t)')
ax.set_title("Heun's Method (Improved Euler / RK2)\nODE: dy/dt = t / y")
ax.legend()
ax.grid(True, linestyle='--', alpha=0.5)

# Text box to display iteration info
text_iter = ax.text(0.05, 1.52, '', fontsize=10)

# ---------------------------------------------
# Animation update
# ---------------------------------------------
def update(frame):
    # Plot all points up to current iteration
    ax.plot(t_vals[:frame+1], y_vals[:frame+1], 'ro-', linewidth=2)
    text_iter.set_text(f"Step {frame}\n"
                       f"t = {t_vals[frame]:.2f}, y = {y_vals[frame]:.4f}\n"
                       f"s₁ = {s1_vals[frame-1]:.4f}  s₂ = {s2_vals[frame-1]:.4f}"
                       if frame > 0 else "")
    return ax,

ani = FuncAnimation(fig, update, frames=len(t_vals),
                    interval=1200, blit=False, repeat=False)

plt.tight_layout()
plt.show()
