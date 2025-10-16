import numpy as np
import matplotlib.pyplot as plt

# Define function
f = lambda x: np.exp(-x**2)

# Domain for plotting
x = np.linspace(-1.2, 1.2, 400)
y = f(x)

# Gaussian nodes and weights (2-point rule)
x1, x2 = -1/np.sqrt(3), 1/np.sqrt(3)
w1 = w2 = 1

# Evaluate f at nodes
f1, f2 = f(x1), f(x2)

# Plot function
plt.figure(figsize=(8,5))
plt.plot(x, y, label=r"$f(x)=e^{-x^2}$", color='royalblue', linewidth=2)

# Shade actual area under curve
plt.fill_between(x, y, color='lightgray', alpha=1, label=r"True area $\int_{-1}^{1} e^{-x^2}dx$")

# Vertical guide lines at x1, x2
plt.axvline(x1, color='gray', linestyle='--', linewidth=1)
plt.axvline(x2, color='gray', linestyle='--', linewidth=1)

# Mark quadrature sample points
plt.scatter([x1, x2], [f1, f2], color='crimson', zorder=5, label=r"Quadrature nodes $x_1, x_2$")

# Draw rectangles showing sampled height × weight
plt.bar(x1, f1, width=1, color='orange', alpha=0.3, edgecolor='darkorange', label="Sample areas")
plt.bar(x2, f2, width=1, color='orange', alpha=0.3, edgecolor='darkorange')

# Labels
plt.text(x1, f1 + 0.05, r"$x_1=-\frac{1}{\sqrt{3}}$", ha='center', fontsize=11)
plt.text(x2, f2 + 0.05, r"$x_2=\frac{1}{\sqrt{3}}$", ha='center', fontsize=11)

# Aesthetics
plt.title("2-Point Gaussian Quadrature on [-1,1]", fontsize=14, weight='bold')
plt.xlabel("x")
plt.ylabel("f(x)")
plt.xlim(-1.2, 1.2)
plt.ylim(0, 1.1)
plt.legend(frameon=False, fontsize=10)
plt.grid(alpha=0.3)
plt.show()
