import numpy as np
import matplotlib.pyplot as plt

# define ODE system
# y1' = y2
# y2' = 6t
def f(t, y):
    y1, y2 = y
    return np.array([y2, 6*t])

# Runge-Kutta 4th order step
def rk4_step(f, t, y, h):
    k1 = f(t, y)
    k2 = f(t + h/2, y + h*k1/2)
    k3 = f(t + h/2, y + h*k2/2)
    k4 = f(t + h, y + h*k3)
    return y + (h/6)*(k1 + 2*k2 + 2*k3 + k4)

# Integrate the system
def integrate(f, t0, y0, h, n):
    t_values = [t0]
    y_values = [y0]
    for i in range(n):
        y0 = rk4_step(f, t0, y0, h)
        t0 += h
        t_values.append(t0)
        y_values.append(y0)
    return np.array(t_values), np.array(y_values)

# setup
a, b = 0, 1
h = 0.05
n = int((b - a) / h)

guesses = [1, -1, 0]   # different guesses for u'(0)
colors = ['r', 'b', 'g']

plt.figure(figsize=(8,5))

for guess, color in zip(guesses, colors):
    y0 = np.array([0, guess])  # [u(0), u'(0)]
    t, y = integrate(f, a, y0, h, n)
    plt.plot(t, y[:,0], color, label=f"u'(0)={guess}")
    print(f"u'(0)={guess:>3},  u(1)={y[-1,0]:.4f}")

plt.axhline(1, color='k', linestyle='--', label='Target u(1)=1')
plt.xlabel('t')
plt.ylabel('u(t)')
plt.title("Shooting Method Visualization for u''=6t, u(0)=0, u(1)=1")
plt.legend()
plt.grid(True)
plt.show()
