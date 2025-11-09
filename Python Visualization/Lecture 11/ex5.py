import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation

# --- Define system of nonlinear equations ---
def f(x):
    x1, x2 = x
    return np.array([
        x1**2 * x2 - 1,           # f1(x1, x2)
        x1 + 3*x1*x2 - 4          # f2(x1, x2)
    ])

def J(x):
    x1, x2 = x
    return np.array([
        [2*x1*x2, x1**2],
        [1 + 3*x2, 3*x1]
    ])

# --- Newton's Method for systems ---
def newton_system(f, J, x0, tol=1e-6, max_iter=20):
    x = np.array(x0, dtype=float)
    history = [x.copy()]
    for k in range(max_iter):
        Fx = f(x)
        Jx = J(x)
        try:
            h = np.linalg.solve(Jx, -Fx)
        except np.linalg.LinAlgError:
            print("Jacobian singular — stopping.")
            break
        x = x + h
        history.append(x.copy())
        if np.linalg.norm(h) < tol:
            break
    return np.array(history)

# --- Interactive input ---
print("Newton's Method for Systems Example")
x1_0 = float(input("Enter initial guess for x1: "))
x2_0 = float(input("Enter initial guess for x2: "))
x0 = np.array([x1_0, x2_0])
print(f"Initial guess: x0 = {x0}")

# --- Run the algorithm ---
history = newton_system(f, J, x0)

print("\nIteration results:")
for i, x in enumerate(history):
    print(f"x({i}) = [{x[0]:.6f}, {x[1]:.6f}]")

# --- Prepare visualization domain ---
x1_vals = np.linspace(0.5, 4, 400)
x2_vals = np.linspace(-0.5, 1.5, 400)
X1, X2 = np.meshgrid(x1_vals, x2_vals)

# Compute f1=0 and f2=0 contour data
F1 = X1**2 * X2 - 1
F2 = X1 + 3*X1*X2 - 4

# --- Plot setup ---
fig, ax = plt.subplots(figsize=(7,6))

# Draw contour lines (f1=0, f2=0)
f1_contour = ax.contour(X1, X2, F1, levels=[0], colors='blue', linewidths=1.5, linestyles='solid')
f2_contour = ax.contour(X1, X2, F2, levels=[0], colors='green', linewidths=1.5, linestyles='dashed')

# Add legend manually (since contour() ignores "label")
f1_proxy = plt.Line2D([0], [0], color='blue', lw=2, label='f1=0')
f2_proxy = plt.Line2D([0], [0], color='green', lw=2, linestyle='--', label='f2=0')
ax.legend(handles=[f1_proxy, f2_proxy])

# Plot labels and grid
ax.set_xlabel(r"$x_1$")
ax.set_ylabel(r"$x_2$")
ax.set_title("Newton's Method for a Nonlinear System (2D)")
ax.grid(True)

# --- Trajectory line and iteration text ---
path_line, = ax.plot([], [], 'ro-', linewidth=2)
text_iter = ax.text(0.05, 0.95, "", transform=ax.transAxes, fontsize=10, va="top")

# --- Animation update function ---
def update(frame):
    path_line.set_data(history[:frame+1, 0], history[:frame+1, 1])
    xk = history[frame]
    text_iter.set_text(f"Iteration {frame}\n"
                       f"x1 = {xk[0]:.4f}\n"
                       f"x2 = {xk[1]:.4f}")
    return path_line, text_iter

# --- Animate the Newton iterations ---
ani = FuncAnimation(
    fig, update, frames=len(history),
    interval=1200, blit=True, repeat=False
)

plt.tight_layout()
plt.show()
