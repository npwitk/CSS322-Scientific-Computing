import numpy as np
import matplotlib.pyplot as plt

# ======================================================
# Given data
# ======================================================
x = np.array([0, 1, 2, 3])
y = np.array([1, -1, 4, 2])

# Construct A matrix for y = mx + c
A = np.vstack([x, np.ones_like(x)]).T  # shape (4, 2)
b = y

# Solve using normal equations
ATA = A.T @ A
ATb = A.T @ b
params = np.linalg.solve(ATA, ATb)
m, c = params
print(f"Best fit line: y = {m:.3f}x + {c:.3f}")

# Predicted values
y_pred = m * x + c
residuals = y - y_pred

# ======================================================
# Plotting
# ======================================================
fig, ax = plt.subplots(figsize=(7,5))
ax.scatter(x, y, color='red', label='Data points', zorder=3)
ax.plot(x, y_pred, color='blue', label=f'Best-fit line: y={m:.2f}x+{c:.2f}', zorder=2)

# Plot residuals as dashed lines
for xi, yi, ypi in zip(x, y, y_pred):
    ax.plot([xi, xi], [ypi, yi], 'k--', lw=1)
    ax.text(xi + 0.05, yi, f"{yi - ypi:+.2f}", fontsize=9, color='darkgray')

# Labeling
ax.set_xlabel("x")
ax.set_ylabel("y")
ax.set_title("Linear Least Squares Fit: y = mx + c")
ax.legend()
ax.grid(True, linestyle='--', alpha=0.5)
plt.tight_layout()
plt.show()
