import numpy as np
import matplotlib.pyplot as plt

# Function to integrate
f = lambda x: x**2

# Original interval [0, 2]
a, b = 0, 2

# Standard Gaussian nodes for n=2
t1, t2 = -1/np.sqrt(3), 1/np.sqrt(3)
w1 = w2 = 1

# Transform nodes to [a,b]
x1 = (b - a) / 2 * t1 + (b + a) / 2
x2 = (b - a) / 2 * t2 + (b + a) / 2

# Evaluate f
f1, f2 = f(x1), f(x2)

# Plot setup
x = np.linspace(-0.5, 2.5, 400)
y = f(x)

plt.figure(figsize=(8,5))
plt.plot(x, y, label=r"$f(x)=x^2$", color="royalblue", linewidth=2)
plt.fill_between(x, y, where=(x>=0)&(x<=2), alpha=0.2, color="lightgray", label="Area on [0,2]")

# Gaussian sample points
plt.scatter([x1, x2], [f1, f2], color="crimson", zorder=5, label="Mapped Gaussian nodes")

# Visual guides
plt.axvline(0, color='gray', linestyle='--', linewidth=1)
plt.axvline(2, color='gray', linestyle='--', linewidth=1)
plt.axvline(x1, color='orange', linestyle=':', linewidth=1)
plt.axvline(x2, color='orange', linestyle=':', linewidth=1)

# Labels
plt.text(x1, f1 + 0.3, f"$x_1={x1:.2f}$", ha='center')
plt.text(x2, f2 + 0.3, f"$x_2={x2:.2f}$", ha='center')
plt.text(1, 3.8, r"Transformed from $t=\pm\frac{1}{\sqrt{3}}$", ha='center', color='gray')

# Titles and legend
plt.title("Transforming Gaussian Quadrature from [-1,1] → [0,2]", fontsize=13, weight='bold')
plt.xlabel("x")
plt.ylabel("f(x)")
plt.legend(frameon=False, fontsize=10)
plt.grid(alpha=0.3)
plt.xlim(-0.5, 2.5)
plt.ylim(0, 5)
plt.show()
