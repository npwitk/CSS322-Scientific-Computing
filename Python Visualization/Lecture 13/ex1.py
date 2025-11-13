import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation

# ============================================
# Define the ODE and analytical solution
# ============================================
def solution(t, y0, lam):
    return y0 * np.exp(lam * t)

# ============================================
# User Input
# ============================================
print("=== Example 1.2: Stability Visualization ===")
print("ODE: y' = λy  →  Solution: y(t) = y₀ e^{λt}")
lam = float(input("Enter λ (e.g. -1 for asymptotic stability, 0.5 for instability): "))
y0_center = float(input("Enter central initial y₀ (e.g. 1): "))

# Nearby perturbations
epsilons = [-0.2, -0.1, 0, 0.1, 0.2]
y0_values = [y0_center + e for e in epsilons]

# Time range
t_max = 5 if lam >= 0 else 8
t = np.linspace(0, t_max, 200)

# ============================================
# Set up the plot
# ============================================
fig, ax = plt.subplots(figsize=(7,5))
ax.set_xlim(0, t_max)
ax.set_ylim(min(y0_values) - 0.5, max(y0_values) * (3 if lam > 0 else 1.5))
ax.set_xlabel("t")
ax.set_ylabel("y(t)")
ax.set_title(f"Stability of y' = λy   (λ = {lam})")
ax.grid(True, linestyle="--", alpha=0.5)

# Lines for each trajectory
lines = []
colors = plt.cm.viridis(np.linspace(0,1,len(y0_values)))
for i, y0 in enumerate(y0_values):
    (line,) = ax.plot([], [], color=colors[i], lw=2, label=f"y₀ = {y0:.2f}")
    lines.append(line)
ax.legend()

# Annotation text
text_box = ax.text(0.05, 0.9, "", transform=ax.transAxes, fontsize=10)

# ============================================
# Animation update
# ============================================
def update(frame):
    for i, y0 in enumerate(y0_values):
        y_vals = solution(t[:frame], y0, lam)
        lines[i].set_data(t[:frame], y_vals)
    text_box.set_text(f"Time = {t[frame]:.2f}s")
    return lines + [text_box]

ani = FuncAnimation(fig, update, frames=len(t), interval=60, blit=False, repeat=False)

plt.tight_layout()
plt.show()
