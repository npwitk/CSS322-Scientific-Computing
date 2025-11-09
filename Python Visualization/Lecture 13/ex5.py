import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation

# ------------------------------
# Define ODE and true solution
# ------------------------------
def f(t, y):
    return t / y

def true_solution(t):
    # Analytic solution: y^2 = t^2/2 + 1
    return np.sqrt(t**2 / 2 + 1)

# ------------------------------
# Backward Euler Step Solver (Newton’s Method)
# ------------------------------
def backward_euler_step(f, t_k, y_k, h, tol=1e-8, max_iter=20):
    # Initial guess: start from y_k (common practice)
    y_guess = y_k
    t_next = t_k + h
    
    for i in range(max_iter):
        g = y_guess - y_k - h * f(t_next, y_guess)        # g(y) = 0
        g_prime = 1 - h * (-t_next / (y_guess**2))        # derivative wrt y
        y_new = y_guess - g / g_prime
        if abs(y_new - y_guess) < tol:
            return y_new
        y_guess = y_new
    return y_guess

# ------------------------------
# Backward Euler full solver
# ------------------------------
def backward_euler(f, y0, t0, h, n):
    t_vals = [t0]
    y_vals = [y0]

    for k in range(n):
        t_k = t_vals[-1]
        y_k = y_vals[-1]
        y_next = backward_euler_step(f, t_k, y_k, h)
        t_vals.append(t_k + h)
        y_vals.append(y_next)
    
    return np.array(t_vals), np.array(y_vals)

# ------------------------------
# Parameters
# ------------------------------
y0 = 1
t0 = 0
h = 0.5
t_end = 2
n = int((t_end - t0) / h)

# Run solver
t_vals, y_vals = backward_euler(f, y0, t0, h, n)
t_true = np.linspace(t0, t_end, 200)
y_true = true_solution(t_true)

# ------------------------------
# Visualization setup
# ------------------------------
fig, ax = plt.subplots(figsize=(7,5))
ax.plot(t_true, y_true, 'g--', label='True Solution')
ax.plot(t_vals, y_vals, 'ro-', label='Backward Euler Points')
ax.set_xlim(t0, t_end)
ax.set_ylim(0.9, 2.5)
ax.set_xlabel("t")
ax.set_ylabel("y(t)")
ax.set_title("Backward Euler Method (Implicit)\nODE: dy/dt = t / y")
ax.legend()
ax.grid(True, linestyle="--", alpha=0.5)
text_iter = ax.text(0.05, 2.35, "", fontsize=10)

# ------------------------------
# Animation update function
# ------------------------------
def update(frame):
    ax.plot(t_vals[:frame+1], y_vals[:frame+1], 'ro-', linewidth=2)
    text_iter.set_text(f"Step {frame}\n"
                       f"t = {t_vals[frame]:.2f}, y = {y_vals[frame]:.4f}")
    return ax,

ani = FuncAnimation(fig, update, frames=len(t_vals),
                    interval=1200, blit=False, repeat=False)

plt.tight_layout()
plt.show()
