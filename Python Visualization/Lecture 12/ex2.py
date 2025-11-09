import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation

# --- Define function and derivatives ---
def f(x):
    return 0.5 - x * np.exp(-x**2)

def f_prime(x):
    return (2*x**2 - 1) * np.exp(-x**2)

def f_doubleprime(x):
    return 2*x * (3 - 2*x**2) * np.exp(-x**2)

# --- Newton optimization method ---
def newton_minimize(f, f_prime, f_doubleprime, x0, tol=1e-6, max_iter=20):
    history = [x0]
    for _ in range(max_iter):
        fp = f_prime(x0)
        fpp = f_doubleprime(x0)
        if abs(fpp) < 1e-12:
            print("Zero second derivative — stopping.")
            break
        x1 = x0 - fp / fpp
        history.append(x1)
        if abs(x1 - x0) < tol:
            break
        x0 = x1
    return np.array(history)

# --- Interactive input ---
print("Newton's Method for Minimization of f(x) = 0.5 - x e^{-x^2}")
x0 = float(input("Enter initial guess x0: "))
print(f"Starting from x0 = {x0}")

# --- Run method ---
history = newton_minimize(f, f_prime, f_doubleprime, x0)
print("\nIteration Results:")
for i, x in enumerate(history):
    print(f"{i:2d}: x = {x:.6f}, f(x) = {f(x):.6f}")

# --- Visualization setup ---
x_vals = np.linspace(-1, 2, 400)
y_vals = f(x_vals)
fig, ax = plt.subplots(figsize=(8,5))
ax.plot(x_vals, y_vals, 'b', label=r"$f(x)=0.5 - xe^{-x^2}$")
ax.set_xlabel("x")
ax.set_ylabel("f(x)")
ax.set_title("Newton's Method for Optimization")
ax.grid(True)

points, = ax.plot([], [], 'ro-', linewidth=2)
tangent_line, = ax.plot([], [], 'r--', linewidth=1.2)
text_iter = ax.text(0.02, 0.95, "", transform=ax.transAxes, fontsize=10, va="top")

# --- Tangent line at point ---
def tangent_at_point(xk):
    slope = f_prime(xk)
    yk = f(xk)
    x_line = np.linspace(xk - 0.3, xk + 0.3, 20)
    y_line = yk + slope * (x_line - xk)
    return x_line, y_line

# --- Animation update ---
def update(frame):
    xk = history[frame]
    points.set_data(history[:frame+1], [f(x) for x in history[:frame+1]])
    x_line, y_line = tangent_at_point(xk)
    tangent_line.set_data(x_line, y_line)
    text_iter.set_text(f"Iteration {frame}\n"
                       f"x={xk:.4f}\n"
                       f"f(x)={f(xk):.4f}")
    return points, tangent_line, text_iter

ani = FuncAnimation(fig, update, frames=len(history),
                    interval=1000, blit=True, repeat=False)

plt.legend()
plt.tight_layout()
plt.show()
