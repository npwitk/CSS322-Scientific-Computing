import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation

# --- Function definition and derivative ---
def f(x):
    return x**2 - 3*x - 4

def df(x):
    return 2*x - 3

# --- Newton iteration producing (x_k, f(x_k)) history ---
def newton(f, df, x0, tol=1e-6, max_iter=20):
    history = [x0]
    for k in range(max_iter):
        fx, dfx = f(history[-1]), df(history[-1])
        if abs(dfx) < 1e-12:
            print("Derivative too small — stopping.")
            break
        x_next = history[-1] - fx/dfx
        history.append(x_next)
        if abs(x_next - history[-2]) < tol:
            break
    return history

# --- Input ---
print("Newton's Method for f(x) = x^2 - 3x - 4")
x0 = float(input("Enter initial guess x0: "))

# --- Run method ---
history = newton(f, df, x0)
print("\nIteration results:")
for k, xk in enumerate(history):
    print(f"x({k}) = {xk:.6f}")

# --- Prepare plot ---
x = np.linspace(min(history[0]-1, -2), max(history[0]+1, 6), 400)
y = f(x)

fig, ax = plt.subplots(figsize=(8,4))
ax.plot(x, y, 'b', label=r"$f(x)=x^2-3x-4$")
ax.axhline(0, color="black", linewidth=0.8)
ax.set_xlabel("x")
ax.set_ylabel("f(x)")
ax.set_title("Newton's Method — Tangent Line Convergence")
ax.legend()

# Plot object placeholders
tangent_line, = ax.plot([], [], "r--", linewidth=1.5)
point, = ax.plot([], [], "ko", markersize=6)
text_iter = ax.text(0.02, 0.95, "", transform=ax.transAxes, fontsize=10, va="top")

# --- Animation update ---
def update(frame):
    xk = history[frame]
    fxk = f(xk)
    dfxk = df(xk)
    # Tangent line: y = f'(xk)*(x - xk) + f(xk)
    x_line = np.linspace(xk-2, xk+2, 100)
    y_line = dfxk*(x_line - xk) + fxk
    tangent_line.set_data(x_line, y_line)
    point.set_data([xk], [fxk])
    text_iter.set_text(f"Iteration {frame}\nx({frame}) = {xk:.6f}")
    return tangent_line, point, text_iter

ani = FuncAnimation(fig, update, frames=len(history),
                    interval=1000, blit=True, repeat=False)

plt.tight_layout()
plt.show()
