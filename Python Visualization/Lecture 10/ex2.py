import numpy as np
import matplotlib.pyplot as plt

# ======================================================
# Given data
# ======================================================
x = np.array([0, 1, 2, 3])
y = np.array([1, -1, 4, 2])

# Construct matrix A for y = c1*x + c2*e^x
A = np.column_stack([x, np.exp(x)])
b = y

# Solve using normal equations: (A^T A)c = A^T b
ATA = A.T @ A
ATb = A.T @ b
c = np.linalg.solve(ATA, ATb)
c1, c2 = c
print(f"Best fit function: y = {c1:.4f}x + {c2:.4f}e^x")

# Predicted values and smooth curve
x_smooth = np.linspace(0, 3, 200)
y_fit = c1 * x_smooth + c2 * np.exp(x_smooth)
y_pred = c1 * x + c2 * np.exp(x)
residuals = y - y_pred

# ======================================================
# Plotting
# ======================================================
fig, ax = plt.subplots(figsize=(7,5))
ax.scatter(x, y, color='red', zorder=3, label='Data points')
ax.plot(x_smooth, y_fit, color='blue', lw=2, label=f'Fit: y={c1:.2f}x {c2:+.2f}e^x')

# Plot residuals (vertical dashed lines)
for xi, yi, ypi in zip(x, y, y_pred):
    ax.plot([xi, xi], [ypi, yi], 'k--', lw=1)
    ax.text(xi + 0.05, yi, f"{yi - ypi:+.2f}", fontsize=9, color='gray')

# Labels and aesthetics
ax.set_xlabel("x")
ax.set_ylabel("y")
ax.set_title("Least Squares Fit: y = c₁x + c₂eˣ")
ax.grid(True, linestyle='--', alpha=0.5)
ax.legend()
plt.tight_layout()
plt.show()
 