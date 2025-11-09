import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation

# --- Function definition ---
def f(x):
    return x**2 - 3*x - 4

# --- Secant method producing history ---
def secant(f, x0, x1, tol=1e-6, max_iter=20):
    history = [(x0, f(x0)), (x1, f(x1))]
    for k in range(max_iter):
        xk, fk = history[-1]
        xkm1, fkm1 = history[-2]
        if abs(fk - fkm1) < 1e-14:
            print("Denominator too small — stopping.")
            break
        x_next = xk - fk*(xk - xkm1)/(fk - fkm1)
        f_next = f(x_next)
        history.append((x_next, f_next))
        if abs(x_next - xk) < tol:
            break
    return history

# --- User input ---
print("Secant Method for f(x) = x^2 - 3x - 4")
x0 = float(input("Enter x(0): "))
x1 = float(input("Enter x(1): "))

# --- Run method ---
history = secant(f, x0, x1)
print("\nIteration results:")
for k, (xk, fk) in enumerate(history):
    print(f"x({k}) = {xk:.6f},  f(x({k})) = {fk:.6f}")

# --- Prepare plot ---
xmin, xmax = min(x0, x1) - 1, max(x0, x1) + 1
x = np.linspace(xmin, xmax, 400)
y = f(x)

fig, ax = plt.subplots(figsize=(8,4))
ax.plot(x, y, 'b', label=r"$f(x)=x^2-3x-4$")
ax.axhline(0, color="black", linewidth=0.8)
ax.set_xlabel("x")
ax.set_ylabel("f(x)")
ax.set_title("Secant Method — Chord Convergence")
ax.legend()

# Initialize graphic elements
secant_line, = ax.plot([], [], "r--", linewidth=1.5)
points, = ax.plot([], [], "ko", markersize=6)
text_iter = ax.text(0.02, 0.95, "", transform=ax.transAxes, fontsize=10, va="top")

# --- Animation update function ---
def update(frame):
    if frame < 1:
        xk, fk = history[frame]
        secant_line.set_data([], [])
        points.set_data([xk], [fk])
        text_iter.set_text(f"Iteration {frame}\nx({frame}) = {xk:.6f}")
        return secant_line, points, text_iter

    xkm1, fkm1 = history[frame-1]
    xk, fk = history[frame]
    x_line = np.linspace(min(xkm1, xk)-1, max(xkm1, xk)+1, 100)
    # secant (chord) line through (x_{k-1}, f_{k-1}) and (x_k, f_k)
    slope = (fk - fkm1) / (xk - xkm1)
    y_line = slope*(x_line - xk) + fk

    secant_line.set_data(x_line, y_line)
    points.set_data([xkm1, xk], [fkm1, fk])
    text_iter.set_text(f"Iteration {frame}\nx({frame}) = {xk:.6f}")
    return secant_line, points, text_iter

ani = FuncAnimation(fig, update, frames=len(history),
                    interval=1000, blit=True, repeat=False)

plt.tight_layout()
plt.show()
