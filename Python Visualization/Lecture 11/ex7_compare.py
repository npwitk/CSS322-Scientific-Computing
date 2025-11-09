import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation

# --- Define system ---
def f(x):
    x1, x2 = x
    return np.array([
        x1 + 2*x2 - 2,
        x1**2 + 4*x2**2 - 4
    ])

def J_true(x):
    x1, x2 = x
    return np.array([
        [1, 2],
        [2*x1, 8*x2]
    ])

# --- Newton's Method for Systems ---
def newton_system(f, J, x0, tol=1e-6, max_iter=20):
    x = np.array(x0, dtype=float)
    history = [x.copy()]
    for k in range(max_iter):
        Fx = f(x)
        Jx = J(x)
        try:
            h = np.linalg.solve(Jx, -Fx)
        except np.linalg.LinAlgError:
            print("Jacobian singular — stopping Newton.")
            break
        x = x + h
        history.append(x.copy())
        if np.linalg.norm(h) < tol:
            break
    return np.array(history)

# --- Broyden's Method ---
def broyden(f, J0, x0, tol=1e-6, max_iter=20):
    x = np.array(x0, dtype=float)
    B = np.array(J0, dtype=float)
    history = [x.copy()]
    for k in range(max_iter):
        Fx = f(x)
        try:
            h = np.linalg.solve(B, -Fx)
        except np.linalg.LinAlgError:
            print("Jacobian approximation singular — stopping Broyden.")
            break
        x_new = x + h
        F_new = f(x_new)
        y = F_new - Fx
        # Rank-1 update
        B = B + np.outer((y - B @ h), h) / np.dot(h, h)
        x = x_new
        history.append(x.copy())
        if np.linalg.norm(h) < tol:
            break
    return np.array(history)

# --- Interactive input ---
print("Broyden vs Newton's Method for f(x) = [x1 + 2x2 - 2,  x1^2 + 4x2^2 - 4]")
x1_0 = float(input("Enter initial guess for x1: "))
x2_0 = float(input("Enter initial guess for x2: "))
x0 = np.array([x1_0, x2_0])
print(f"Initial guess: x0 = {x0}")

# --- Run both methods ---
newton_hist = newton_system(f, J_true, x0)
broyden_hist = broyden(f, J_true(x0), x0)

# --- Print iteration summary ---
print("\nIteration results (Newton vs Broyden):")
for i in range(max(len(newton_hist), len(broyden_hist))):
    xn = newton_hist[i] if i < len(newton_hist) else [np.nan, np.nan]
    xb = broyden_hist[i] if i < len(broyden_hist) else [np.nan, np.nan]
    print(f"Iter {i:2d}: Newton [{xn[0]:.6f}, {xn[1]:.6f}] | Broyden [{xb[0]:.6f}, {xb[1]:.6f}]")

# --- Prepare contour plot data ---
x1_vals = np.linspace(-2, 2, 400)
x2_vals = np.linspace(0, 2.5, 400)
X1, X2 = np.meshgrid(x1_vals, x2_vals)
F1 = X1 + 2*X2 - 2
F2 = X1**2 + 4*X2**2 - 4

# --- Create two side-by-side subplots ---
fig, (axN, axB) = plt.subplots(1, 2, figsize=(12, 6))
plt.tight_layout()

def setup_axes(ax, title):
    ax.contour(X1, X2, F1, levels=[0], colors='blue', linewidths=1.5)
    ax.contour(X1, X2, F2, levels=[0], colors='green', linestyles='--', linewidths=1.5)
    f1_proxy = plt.Line2D([0], [0], color='blue', lw=2, label='f1=0')
    f2_proxy = plt.Line2D([0], [0], color='green', lw=2, linestyle='--', label='f2=0')
    ax.legend(handles=[f1_proxy, f2_proxy])
    ax.set_xlabel(r"$x_1$")
    ax.set_ylabel(r"$x_2$")
    ax.set_title(title)
    ax.grid(True)

setup_axes(axN, "Newton's Method")
setup_axes(axB, "Broyden's Method")

# Initialize paths and text
lineN, = axN.plot([], [], 'ro-', lw=2)
lineB, = axB.plot([], [], 'ro-', lw=2)
textN = axN.text(0.05, 0.95, "", transform=axN.transAxes, fontsize=10, va="top")
textB = axB.text(0.05, 0.95, "", transform=axB.transAxes, fontsize=10, va="top")

# --- Update function for animation ---
def update(frame):
    # Newton
    if frame < len(newton_hist):
        lineN.set_data(newton_hist[:frame+1, 0], newton_hist[:frame+1, 1])
        xn = newton_hist[frame]
        textN.set_text(f"Iter {frame}\n"
                       f"x1={xn[0]:.3f}\n"
                       f"x2={xn[1]:.3f}")
    # Broyden
    if frame < len(broyden_hist):
        lineB.set_data(broyden_hist[:frame+1, 0], broyden_hist[:frame+1, 1])
        xb = broyden_hist[frame]
        textB.set_text(f"Iter {frame}\n"
                       f"x1={xb[0]:.3f}\n"
                       f"x2={xb[1]:.3f}")
    return lineN, lineB, textN, textB

ani = FuncAnimation(
    fig, update, frames=max(len(newton_hist), len(broyden_hist)),
    interval=1200, blit=True, repeat=False
)

plt.show()
