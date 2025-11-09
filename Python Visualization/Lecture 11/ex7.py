import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation

# --- Define the nonlinear system ---
def f(x):
    x1, x2 = x
    return np.array([
        x1 + 2*x2 - 2,
        x1**2 + 4*x2**2 - 4
    ])

def J_true(x):
    """True Jacobian (used only for the initial B0)"""
    x1, x2 = x
    return np.array([
        [1, 2],
        [2*x1, 8*x2]
    ])

# --- Broyden's Method ---
def broyden(f, J0, x0, tol=1e-6, max_iter=20):
    x = np.array(x0, dtype=float)
    B = np.array(J0, dtype=float)
    history = [x.copy()]
    B_history = [B.copy()]

    for k in range(max_iter):
        Fx = f(x)
        try:
            h = np.linalg.solve(B, -Fx)
        except np.linalg.LinAlgError:
            print("Jacobian approximation singular — stopping.")
            break

        x_new = x + h
        F_new = f(x_new)

        # Broyden update
        y = F_new - Fx
        B = B + np.outer((y - B @ h), h) / np.dot(h, h)

        x = x_new
        history.append(x.copy())
        B_history.append(B.copy())

        if np.linalg.norm(h) < tol:
            break

    return np.array(history), B_history

# --- Initial setup ---
print("Broyden's Method for f(x) = [x1 + 2x2 - 2,  x1^2 + 4x2^2 - 4]")
x0 = np.array([1.0, 2.0])
B0 = J_true(x0)
print(f"Initial guess: {x0}")
print(f"Initial Jacobian:\n{B0}")

# --- Run method ---
history, B_hist = broyden(f, B0, x0)

print("\nIteration results:")
for i, x in enumerate(history):
    print(f"x({i}) = [{x[0]:.6f}, {x[1]:.6f}]")

# --- Visualization setup ---
x1_vals = np.linspace(-2, 2, 400)
x2_vals = np.linspace(0, 2.5, 400)
X1, X2 = np.meshgrid(x1_vals, x2_vals)
F1 = X1 + 2*X2 - 2
F2 = X1**2 + 4*X2**2 - 4

fig, ax = plt.subplots(figsize=(7,6))
ax.contour(X1, X2, F1, levels=[0], colors='blue', linewidths=1.5, linestyles='solid')
ax.contour(X1, X2, F2, levels=[0], colors='green', linewidths=1.5, linestyles='dashed')

# Legend
f1_proxy = plt.Line2D([0], [0], color='blue', lw=2, label='f1=0')
f2_proxy = plt.Line2D([0], [0], color='green', lw=2, linestyle='--', label='f2=0')
ax.legend(handles=[f1_proxy, f2_proxy])

ax.set_xlabel(r"$x_1$")
ax.set_ylabel(r"$x_2$")
ax.set_title("Broyden's Method for Nonlinear System")
ax.grid(True)

# Path line and iteration info
path_line, = ax.plot([], [], 'ro-', linewidth=2)
text_iter = ax.text(0.05, 0.95, "", transform=ax.transAxes, fontsize=10, va="top")

# --- Animation update ---
def update(frame):
    path_line.set_data(history[:frame+1,0], history[:frame+1,1])
    xk = history[frame]
    text_iter.set_text(f"Iteration {frame}\n"
                       f"x1 = {xk[0]:.4f}\n"
                       f"x2 = {xk[1]:.4f}")
    return path_line, text_iter

ani = FuncAnimation(
    fig, update, frames=len(history),
    interval=1200, blit=True, repeat=False
)

plt.tight_layout()
plt.show()
