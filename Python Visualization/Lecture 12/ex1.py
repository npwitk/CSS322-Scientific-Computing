import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation

# --- Function definition ---
def f(x):
    return 0.5 - x * np.exp(-x**2)

# --- Successive Parabolic Interpolation algorithm ---
def successive_parabolic_interpolation(f, x0, x1, x2, tol=1e-6, max_iter=20):
    history = [(x0, f(x0)), (x1, f(x1)), (x2, f(x2))]
    for k in range(max_iter):
        u, v, w = [x for x, _ in history[-3:]]
        fu, fv, fw = f(u), f(v), f(w)

        # Denominator check
        denom = (v - u) * (fv - fw) - (v - w) * (fv - fu)
        if abs(denom) < 1e-12:
            print("Denominator too small — stopping.")
            break

        # Parabolic interpolation formula
        x_new = v - 0.5 * (
            ((v - u)**2 * (fv - fw) - (v - w)**2 * (fv - fu))
            / denom
        )

        f_new = f(x_new)
        history.append((x_new, f_new))

        # Replace oldest point (keep last three)
        history = history[-3:]

        # Check convergence
        if abs(x_new - v) < tol:
            break

    return np.array(history)

# --- Interactive input ---
print("Successive Parabolic Interpolation")
x0 = float(input("Enter initial x0: "))
x1 = float(input("Enter initial x1: "))
x2 = float(input("Enter initial x2: "))
print(f"Initial points: {x0}, {x1}, {x2}")

# --- Run method ---
history = [(x0, f(x0)), (x1, f(x1)), (x2, f(x2))]
full_history = [(x0, f(x0)), (x1, f(x1)), (x2, f(x2))]

for i in range(20):
    u, v, w = [x for x, _ in history[-3:]]
    fu, fv, fw = f(u), f(v), f(w)
    denom = (v - u) * (fv - fw) - (v - w) * (fv - fu)
    if abs(denom) < 1e-12:
        break
    x_new = v - 0.5 * (((v - u)**2 * (fv - fw) - (v - w)**2 * (fv - fu)) / denom)
    f_new = f(x_new)
    history.append((x_new, f_new))
    full_history.append((x_new, f_new))
    history = history[-3:]
    if abs(x_new - v) < 1e-6:
        break

xs = np.array([x for x, _ in full_history])
fs = np.array([f(x) for x in xs])
print("\nIteration Results:")
for i, (xv, fv) in enumerate(full_history):
    print(f"{i:2d}: x = {xv:.6f}, f(x) = {fv:.6f}")

# --- Visualization setup ---
x_vals = np.linspace(-1, 2, 400)
y_vals = f(x_vals)

fig, ax = plt.subplots(figsize=(8,5))
ax.plot(x_vals, y_vals, 'b', label=r"$f(x)=0.5 - xe^{-x^2}$")
ax.set_title("Successive Parabolic Interpolation")
ax.set_xlabel("x")
ax.set_ylabel("f(x)")
ax.grid(True)

points, = ax.plot([], [], 'ro', markersize=6)
interp_curve, = ax.plot([], [], 'r--', linewidth=1.5)
text_iter = ax.text(0.02, 0.95, "", transform=ax.transAxes, fontsize=10, va="top")

# --- Parabola fitting helper ---
def parabola_through_points(u, fu, v, fv, w, fw, n=200):
    # Fit a quadratic polynomial
    A = np.array([
        [u**2, u, 1],
        [v**2, v, 1],
        [w**2, w, 1]
    ])
    b = np.array([fu, fv, fw])
    a, b_, c = np.linalg.solve(A, b)
    xx = np.linspace(min(u,v,w)-0.2, max(u,v,w)+0.2, n)
    yy = a*xx**2 + b_*xx + c
    return xx, yy

# --- Animation update ---
def update(frame):
    if frame < 2:
        return points, interp_curve, text_iter
    u, v, w = xs[max(0, frame-2)], xs[max(0, frame-1)], xs[frame]
    fu, fv, fw = f(u), f(v), f(w)
    xx, yy = parabola_through_points(u, fu, v, fv, w, fw)
    interp_curve.set_data(xx, yy)
    points.set_data([u, v, w], [fu, fv, fw])
    text_iter.set_text(f"Iteration {frame}\n"
                       f"x={w:.4f}\n"
                       f"f(x)={fw:.4f}")
    return points, interp_curve, text_iter

ani = FuncAnimation(fig, update, frames=len(xs),
                    interval=1200, blit=True, repeat=False)

plt.legend()
plt.tight_layout()
plt.show()
