import numpy as np
import matplotlib.pyplot as plt

# Function definition
def f(x): return x**2 + 3*x

# Given
x0 = 1
h = 0.01
true_slope = 2*x0 + 3

# Function values
f_x0 = f(x0)
f_forward = f(x0 + h)
f_backward = f(x0 - h)

# Finite differences
forward = (f_forward - f_x0) / h
backward = (f_x0 - f_backward) / h
central = (f_forward - f_backward) / (2*h)

# Second derivative
second = (f_forward - 2*f_x0 + f_backward) / (h**2)

print(f"Forward diff: {forward:.5f}")
print(f"Backward diff: {backward:.5f}")
print(f"Central diff: {central:.5f}")
print(f"Second derivative (central): {second:.5f}")

# Plot setup
X = np.linspace(0.5, 1.5, 200)
plt.figure(figsize=(7,5))
plt.plot(X, f(X), 'b', lw=2, label='$f(x)=x^2+3x$')

# Tangent line
tangent = true_slope*(X - x0) + f_x0
plt.plot(X, tangent, 'k--', label=f'True tangent slope = {true_slope}')

# Secant lines
fwd_line = forward*(X - x0) + f_x0
bwd_line = backward*(X - x0) + f_x0
cent_line = central*(X - x0) + f_x0

plt.plot(X, fwd_line, 'r--', label=f'Forward slope = {forward:.2f}')
plt.plot(X, bwd_line, 'g--', label=f'Backward slope = {backward:.2f}')
plt.plot(X, cent_line, 'orange', ls='--', label=f'Central slope = {central:.2f}')

# Points
plt.scatter([x0-h, x0, x0+h],
            [f_backward, f_x0, f_forward],
            color='black', zorder=5)
plt.text(x0, f_x0+0.1, "x=1", ha='center')

plt.title("Forward, Backward, and Central Difference Approximation at x=1")
plt.xlabel("x")
plt.ylabel("f(x)")
plt.legend()
plt.grid(True)
plt.show()
