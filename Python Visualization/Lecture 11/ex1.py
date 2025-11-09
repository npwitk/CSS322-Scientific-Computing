import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation

# --- Function definition ---
def f(x):
    """Example function"""
    return x**2 - 3*x - 4

# --- Bisection method returning iteration history ---
def bisection(f, a, b, tol=1e-6, max_iter=30):
    fa, fb = f(a), f(b)
    if fa * fb > 0:
        raise ValueError("f(a) and f(b) must have opposite signs.")
    history = []
    for k in range(max_iter):
        m = a + (b - a) / 2
        fm = f(m)
        history.append((a, b, m, fa, fb, fm))
        if abs(fm) < tol or (b - a) / 2 < tol:
            break
        if np.sign(fa) != np.sign(fm):
            b, fb = m, fm
        else:
            a, fa = m, fm
    return history

# --- User input ---
print("Bisection Method for f(x) = x^2 - 3x - 4")
print("Enter your initial interval [a, b]:")
a = float(input("a = "))
b = float(input("b = "))

# --- Check sign condition ---
fa, fb = f(a), f(b)
print(f"f(a) = {fa:.6f}, f(b) = {fb:.6f}")

if fa * fb > 0:
    print("\nf(a) and f(b) have the SAME sign.")
    print("Try again with an interval where f(a) and f(b) differ in sign.")
    exit()
else:
    print("\nGood! f(a) and f(b) have opposite signs — a root lies within.\n")

# --- Run algorithm ---
history = bisection(f, a, b, tol=1e-6)

# --- Plot setup ---
x = np.linspace(a - 1, b + 1, 400)
y = f(x)

fig, ax = plt.subplots(figsize=(8, 4))
ax.plot(x, y, label=r"$f(x)=x^2-3x-4$")
ax.axhline(0, color="black", linewidth=0.8)
ax.set_xlabel("x")
ax.set_ylabel("f(x)")
ax.set_title("Bisection Method Animation")

# Fixed axis limits for clear reference (no zooming)
x_margin = (b - a) * 0.5
ax.set_xlim(a - x_margin, b + x_margin)
ax.set_ylim(min(y) - 2, max(y) + 2)

# Initialize graphics elements
interval_line, = ax.plot([], [], "r-", linewidth=8, alpha=0.5)
midpoint_dot, = ax.plot([], [], "bo", markersize=6)
text_iter = ax.text(0.02, 0.95, "", transform=ax.transAxes, fontsize=10, va="top")

# --- Animation update function ---
def update(frame):
    a, b, m, fa, fb, fm = history[frame]
    interval_line.set_data([a, b], [0, 0])
    midpoint_dot.set_data([m], [0])
    text_iter.set_text(f"Iteration {frame+1}\n[a,b]=[{a:.4f},{b:.4f}]\nmid={m:.4f}")
    return interval_line, midpoint_dot, text_iter

# --- Animate ---
ani = FuncAnimation(fig, update, frames=len(history),
                    interval=1000, blit=True, repeat=False)

plt.legend()
plt.tight_layout()
plt.show()
