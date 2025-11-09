import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from matplotlib.animation import FuncAnimation

# --- Define function and derivatives ---
def f(x1, x2):
    return 0.5*x1**2 + 2.5*x2**2

def grad_f(x):
    x1, x2 = x
    return np.array([x1, 5*x2])

def hessian_f(x):
    # Constant Hessian for this quadratic
    return np.array([[1, 0], [0, 5]])

# --- Newton's Method ---
def newton_2d(x0, tol=1e-6, max_iter=10):
    history = [x0]
    x = x0
    for _ in range(max_iter):
        g = grad_f(x)
        H = hessian_f(x)
        h = -np.linalg.solve(H, g)
        x_new = x + h
        history.append(x_new)
        if np.linalg.norm(h) < tol:
            break
        x = x_new
    return np.array(history)

# --- Interactive input ---
print("Newton's Method in 2D: f(x1, x2) = 0.5x1^2 + 2.5x2^2")
x1_0 = float(input("Enter initial x1: "))
x2_0 = float(input("Enter initial x2: "))
x0 = np.array([x1_0, x2_0])
print(f"Starting point: {x0}")

# --- Run method ---
history = newton_2d(x0)
print("\nIteration Results:")
for i, x in enumerate(history):
    print(f"{i}: x = [{x[0]:.4f}, {x[1]:.4f}], f(x) = {f(x[0], x[1]):.6f}")

# --- 3D Surface Plot Setup ---
x1 = np.linspace(-6, 6, 100)
x2 = np.linspace(-3, 3, 100)
X1, X2 = np.meshgrid(x1, x2)
Z = f(X1, X2)

fig = plt.figure(figsize=(8,6))
ax = fig.add_subplot(111, projection='3d')
ax.plot_surface(X1, X2, Z, cmap='viridis', alpha=0.7, linewidth=0)
ax.set_xlabel("$x_1$")
ax.set_ylabel("$x_2$")
ax.set_zlabel("$f(x_1, x_2)$")
ax.set_title("Newton's Method for 2D Quadratic Optimization")

# --- Trajectory line and points ---
path_line, = ax.plot([], [], [], 'r-', linewidth=2, label='Newton path')
current_point, = ax.plot([], [], [], 'ro', markersize=6)
text_iter = ax.text2D(0.05, 0.9, "", transform=ax.transAxes, fontsize=10)

# --- Animation update ---
def update(frame):
    x_path = history[:frame+1]
    xs, ys = x_path[:,0], x_path[:,1]
    zs = f(xs, ys)
    path_line.set_data(xs, ys)
    path_line.set_3d_properties(zs)
    current_point.set_data(xs[-1:], ys[-1:])
    current_point.set_3d_properties(zs[-1:])
    text_iter.set_text(f"Iteration {frame}\n"
                       f"x=({xs[-1]:.3f},{ys[-1]:.3f})\n"
                       f"f(x)={zs[-1]:.4f}")
    return path_line, current_point, text_iter

ani = FuncAnimation(fig, update, frames=len(history),
                    interval=1500, blit=False, repeat=False)

ax.legend()
plt.tight_layout()
plt.show()
