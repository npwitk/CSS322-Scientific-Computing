import numpy as np
import matplotlib.pyplot as plt

# Function
def f(t, y):
    return t / y

# Euler’s Method
def euler(f, y0, t0, h, steps):
    t = [t0]
    y = [y0]
    for _ in range(steps):
        y_next = y[-1] + h * f(t[-1], y[-1])
        t_next = t[-1] + h
        t.append(t_next)
        y.append(y_next)
    return np.array(t), np.array(y)

# Ask user for steps
steps = int(input("Enter number of Euler steps (e.g. 4, 10, 15): "))

# Parameters
t0, y0 = 0, 1
t_end = 1.5
h = (t_end - t0) / steps

# Euler approximation
t_euler, y_euler = euler(f, y0, t0, h, steps)

# Exact solution
t_exact = np.linspace(0, t_end, 400)
y_exact = np.sqrt(t_exact**2 + 1)

# Plot
plt.figure(figsize=(7, 5))
plt.plot(t_exact, y_exact, 'g-', lw=2, label='Exact Solution $y=\\sqrt{t^2+1}$')
plt.plot(t_euler, y_euler, 'ro--', lw=1.5, label=f"Euler (steps={steps})")
plt.title(f"Euler’s Method vs Analytical Solution ({steps} steps)")
plt.xlabel("t")
plt.ylabel("y")
plt.legend()
plt.grid(True)
plt.xlim(0, 1.6)
plt.ylim(0.9, 2.4)
plt.show()
