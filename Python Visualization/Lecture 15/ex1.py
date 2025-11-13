import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation

# --- Define matrix ---
A = np.array([[3, -1],
              [-1, 3]])

# Eigen decomposition
eigvals, eigvecs = np.linalg.eig(A)

# --- Create grid ---
x = np.linspace(-2, 2, 15)
y = np.linspace(-2, 2, 15)
X, Y = np.meshgrid(x, y)
points = np.vstack([X.flatten(), Y.flatten()])

# --- Animation setup ---
fig, ax = plt.subplots(figsize=(7,7))
ax.set_xlim(-5, 5)
ax.set_ylim(-5, 5)
ax.set_aspect('equal')
ax.axhline(0, color='black', lw=1)
ax.axvline(0, color='black', lw=1)
ax.set_title("Matrix Transformation Animation")

# Quiver for initial arrows (light blue)
quiver = ax.quiver(points[0], points[1],
                   np.zeros_like(points[0]), np.zeros_like(points[1]),
                   angles='xy', scale_units='xy', scale=1, color='lightblue', alpha=0.6)

# Plot eigenvectors (red arrows)
for i in range(len(eigvals)):
    v = eigvecs[:, i]
    lam = eigvals[i]
    ax.quiver(0, 0, v[0]*lam, v[1]*lam, color='red', scale_units='xy', scale=1, width=0.015)
    ax.text(v[0]*lam*1.1, v[1]*lam*1.1, rf"$\lambda={lam:.1f}$", color='red', fontsize=10)

# --- Animation function ---
frames = 50
def animate(i):
    t = i / frames
    M = (1 - t) * np.eye(2) + t * A  # gradual interpolation
    new_points = M @ points
    quiver.set_UVC(new_points[0] - points[0], new_points[1] - points[1])
    ax.set_title(f"Transformation Progress: {t*100:.0f}%")
    return quiver,

anim = FuncAnimation(fig, animate, frames=frames+1, interval=100, blit=False)
plt.show()
