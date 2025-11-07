import numpy as np
import matplotlib.pyplot as plt

# Given function
def f(t, y):
    return t / y

# Euler's Method
def euler(f, y0, t0, h, steps):
    t = [t0]
    y = [y0]
    for _ in range(steps):
        y_next = y[-1] + h * f(t[-1], y[-1])
        t_next = t[-1] + h
        t.append(t_next)
        y.append(y_next)
    return np.array(t), np.array(y)

# Parameters
h = 0.5
t0, y0 = 0, 1
steps = 3  # up to t=1.5

# Euler approximation
t_euler, y_euler = euler(f, y0, t0, h, steps)

# Analytical solution
t_exact = np.linspace(0, 1.5, 100)
y_exact = np.sqrt(t_exact**2 + 1)

# Plot
plt.figure(figsize=(7, 5))
plt.plot(t_exact, y_exact, 'g-', label='Exact Solution $y=\\sqrt{t^2+1}$')
plt.plot(t_euler, y_euler, 'ro--', label="Euler's Approximation")
plt.title("Euler’s Method vs Analytical Solution")
plt.xlabel("t")
plt.ylabel("y")
plt.legend()
plt.grid(True)
plt.show()
