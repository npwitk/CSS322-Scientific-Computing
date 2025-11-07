import numpy as np
import matplotlib.pyplot as plt

# Given differential equation
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

# Adams–Bashforth 2nd Order Method
def euler_first_point(f, t0, y0, h):
    return y0 + h * f(t0, y0)

def ab2(f, t0, y0, h, steps):
    t = [t0]
    y = [y0]
    # Use Euler for first step
    t.append(t0 + h)
    y.append(euler_first_point(f, t0, y0, h))
    # AB2 formula
    for k in range(1, steps):
        t_next = t[k] + h
        y_next = y[k] + (3*h/2)*f(t[k], y[k]) - (h/2)*f(t[k-1], y[k-1])
        t.append(t_next)
        y.append(y_next)
    return np.array(t), np.array(y)

# Parameters
h = 0.5
t0, y0 = 0, 1
steps = 3  # up to t=1.5

# Compute results
t_euler, y_euler = euler(f, y0, t0, h, steps)
t_ab2, y_ab2 = ab2(f, t0, y0, h, steps)
t_exact = np.linspace(0, 1.5, 100)
y_exact = np.sqrt(t_exact**2 + 1)

# Plot (3 lines)
plt.plot(t_exact, y_exact, 'g-', label='Exact Solution')
plt.plot(t_euler, y_euler, 'ro--', label='Euler’s Method')
plt.plot(t_ab2, y_ab2, 'bo--', label='Adams-Bashforth 2nd Order (AB2)')
plt.xlabel('t')
plt.ylabel('y')
plt.title('Euler vs AB2 vs Exact Solution')
plt.legend()
plt.grid(True)
plt.show()
