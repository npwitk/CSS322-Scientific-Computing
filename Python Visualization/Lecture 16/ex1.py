import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation

# ODE: u'' = 6t  -> y1' = y2,  y2' = 6t
def f(t, y):
    y1, y2 = y
    return np.array([y2, 6*t])

# RK4 integrator
def rk4(f, y0, t0, tf, h):
    t = np.arange(t0, tf + h, h)
    y = np.zeros((len(t), len(y0)))
    y[0] = y0
    for i in range(len(t)-1):
        k1 = f(t[i], y[i])
        k2 = f(t[i] + h/2, y[i] + h*k1/2)
        k3 = f(t[i] + h/2, y[i] + h*k2/2)
        k4 = f(t[i] + h, y[i] + h*k3)
        y[i+1] = y[i] + (h/6)*(k1 + 2*k2 + 2*k3 + k4)
    return t, y

# Parameters
t0, tf, h = 0, 1, 0.05
target = 1
slopes = np.linspace(-1, 1, 60)

fig, ax = plt.subplots(figsize=(6,4))
ax.set_xlim(0, 1)
ax.set_ylim(-0.5, 2.2)
ax.set_xlabel("t")
ax.set_ylabel("u(t)")
ax.set_title("Shooting Method Animation")
ax.grid(True)

# Target and fixed boundary points
ax.axhline(target, color='gray', ls='--', label='Target u(1)=1')
ax.scatter([0,1],[0,1], color='black', zorder=5)
ax.text(0, -0.1, r"$\alpha$", fontsize=12)
ax.text(1.02, 1.02, r"$\beta$", fontsize=12)

# Lines for animation
line, = ax.plot([], [], 'b-', lw=2, alpha=0.8)
dot, = ax.plot([], [], 'ro', alpha=0.8)
final_line, = ax.plot([], [], 'g-', lw=3, alpha=0)  # will appear at the end

def init():
    line.set_data([], [])
    dot.set_data([], [])
    final_line.set_data([], [])
    final_line.set_alpha(0)
    return line, dot, final_line

def update(i):
    slope = slopes[i]
    t, y = rk4(f, np.array([0.0, slope]), t0, tf, h)
    line.set_data(t, y[:,0])
    dot.set_data([t[-1]], [y[-1,0]])

    # Fade lines in/out
    alpha = 1 - abs(slope)/1.2
    line.set_alpha(alpha)
    dot.set_alpha(alpha)

    ax.set_title(f"Try y₂(0) = {slope:.2f} → u(1) = {y[-1,0]:.2f}")

    # If slope is close to 0, highlight final green curve
    if abs(slope) < 0.05:
        final_line.set_data(t, y[:,0])
        final_line.set_alpha(1.0)
        ax.set_title(f"✅ Correct slope y₂(0) = 0 → u(1) = 1.00")

    return line, dot, final_line

ani = FuncAnimation(
    fig, update, frames=len(slopes),
    init_func=init, blit=True, interval=100, repeat=True
)

plt.legend()
plt.tight_layout()
plt.show()
