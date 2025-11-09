import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation

# ---------------------------------------------------
# Define the system of ODEs: y' = f(t, y)
# ---------------------------------------------------
def f(t, y):
    y1, y2 = y
    dy1 = t * y2 - y1
    dy2 = 2 * y1**2 * y2 + t**2
    return np.array([dy1, dy2])

# ---------------------------------------------------
# Euler’s method for systems
# ---------------------------------------------------
def euler_system(f, y0, t0, h, n_steps):
    t_vals = [t0]
    y_vals = [y0]

    for k in range(n_steps):
        t_k = t_vals[-1]
        y_k = y_vals[-1]
        y_next = y_k + h * f(t_k, y_k)
        t_vals.append(t_k + h)
        y_vals.append(y_next)

    return np.array(t_vals), np.array(y_vals)

# ---------------------------------------------------
# Parameters
# ---------------------------------------------------
t0 = 0.0
y0 = np.array([-2.0, 1.0])
h = 0.2
n_steps = 5  # perform a few steps for clarity

t_vals, y_vals = euler_system(f, y0, t0, h, n_steps)

# ---------------------------------------------------
# Plot setup
# ---------------------------------------------------
fig, ax = plt.subplots(1, 2, figsize=(10, 4))

# Left: y1(t) and y2(t) vs time
ax[0].set_title("Euler’s Method for System of ODEs")
ax[0].set_xlabel("t")
ax[0].set_ylabel("y₁(t), y₂(t)")
ax[0].grid(True, linestyle="--", alpha=0.5)
ax[0].set_xlim(0, t_vals[-1])
ax[0].set_ylim(min(y_vals[:,0].min(), y_vals[:,1].min()) - 0.5,
               max(y_vals[:,0].max(), y_vals[:,1].max()) + 0.5)
line_y1, = ax[0].plot([], [], "r-o", label="y₁(t)")
line_y2, = ax[0].plot([], [], "b-o", label="y₂(t)")
ax[0].legend()

# Right: phase plot y₁ vs y₂
ax[1].set_title("Phase Plot (y₂ vs y₁)")
ax[1].set_xlabel("y₁")
ax[1].set_ylabel("y₂")
ax[1].grid(True, linestyle="--", alpha=0.5)
ax[1].set_xlim(-2.5, 0)
ax[1].set_ylim(0.5, y_vals[:,1].max() + 1)
phase_line, = ax[1].plot([], [], "m-o")

# Iteration text
text_box = ax[0].text(0.05, 0.9, "", transform=ax[0].transAxes, fontsize=10)

# ---------------------------------------------------
# Animation function
# ---------------------------------------------------
def update(frame):
    line_y1.set_data(t_vals[:frame+1], y_vals[:frame+1,0])
    line_y2.set_data(t_vals[:frame+1], y_vals[:frame+1,1])
    phase_line.set_data(y_vals[:frame+1,0], y_vals[:frame+1,1])
    text_box.set_text(f"Step {frame}\n"
                      f"t = {t_vals[frame]:.2f}\n"
                      f"y₁ = {y_vals[frame,0]:.4f}\n"
                      f"y₂ = {y_vals[frame,1]:.4f}")
    return line_y1, line_y2, phase_line, text_box

ani = FuncAnimation(fig, update, frames=len(t_vals),
                    interval=1000, blit=False, repeat=False)

plt.tight_layout()
plt.show()
