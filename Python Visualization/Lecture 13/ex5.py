import numpy as np
import matplotlib.pyplot as plt

# ------------------------------
# Define ODE and true solution
# ------------------------------
def f(t, y):
    return t / y

def true_solution(t):
    # Analytic solution: y^2 = t^2/2 + 1
    return np.sqrt(t**2 / 2 + 1)

# ------------------------------
# Backward Euler exact-step formula for this ODE
# ------------------------------
def backward_euler_step_exact(t_k, y_k, h, positive=True):
    t_next = t_k + h
    # Equation: y_{k+1} = y_k + h * (t_{k+1}/y_{k+1})
    # => y_{k+1}^2 - y_k*y_{k+1} - h*t_{k+1} = 0
    # Quadratic: a=1, b=-y_k, c=-h*t_{k+1}
    a, b, c = 1, -y_k, -h*t_next
    disc = np.sqrt(b**2 - 4*a*c)
    if positive:
        y_next = (-b + disc) / (2*a)
    else:
        y_next = (-b - disc) / (2*a)
    return y_next

# ------------------------------
# Full solver
# ------------------------------
def backward_euler_interactive(f, y0, t0, h, n, positive=True):
    t_vals = [t0]
    y_vals = [y0]
    for k in range(n):
        t_k, y_k = t_vals[-1], y_vals[-1]
        y_next = backward_euler_step_exact(t_k, y_k, h, positive)
        t_vals.append(t_k + h)
        y_vals.append(y_next)
    return np.array(t_vals), np.array(y_vals)

# ------------------------------
# Ask user choice
# ------------------------------
choice = input("Choose root type (positive/negative): ").strip().lower()
use_positive = True if choice.startswith('p') else False

# ------------------------------
# Parameters
# ------------------------------
y0 = 1
t0 = 0
h = 0.5
t_end = 2
n = int((t_end - t0) / h)

# Run solver
t_vals, y_vals = backward_euler_interactive(f, y0, t0, h, n, positive=use_positive)
t_true = np.linspace(t0, t_end, 200)
y_true = true_solution(t_true)

# ------------------------------
# Plot
# ------------------------------
plt.figure(figsize=(7,5))
plt.plot(t_true, y_true, 'g--', label='True Solution (Positive branch)')
plt.plot(t_vals, y_vals, 'ro-', label=f'Backward Euler ({choice.title()} root)')
plt.xlabel("t")
plt.ylabel("y(t)")
plt.title("Backward Euler Method (Implicit)\nODE: dy/dt = t / y")
plt.legend()
plt.grid(True, linestyle="--", alpha=0.5)
plt.tight_layout()
plt.show()
