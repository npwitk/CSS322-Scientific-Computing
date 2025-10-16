import numpy as np
import matplotlib.pyplot as plt

# Define the function and true value
f = lambda x: x**2
a, b = 0, 2
true_value = 8/3

# Composite setup
k = 3
h = (b - a) / k
x_points = np.linspace(a, b, k+1)
midpoints = (x_points[:-1] + x_points[1:]) / 2

# Compute approximations
M3 = h * np.sum(f(midpoints))
T3 = h * (0.5*f(a) + np.sum(f(x_points[1:-1])) + 0.5*f(b))

# --- Plot setup ---
x = np.linspace(a, b, 400)
y = f(x)
fig, axes = plt.subplots(1, 2, figsize=(12, 5))

# ---------- Composite Midpoint ----------
axes[0].plot(x, y, color="royalblue", lw=2, label=r"$f(x)=x^2$")
axes[0].fill_between(x, y, color="lightgray", alpha=0.3)
for m in midpoints:
    axes[0].bar(m, f(m), width=h, align='center', alpha=0.4, color='orange', edgecolor='darkorange')
axes[0].set_title(f"Composite Midpoint (M₃ = {M3:.4f})", fontsize=12)
axes[0].set_xlabel("x")
axes[0].set_ylabel("f(x)")
axes[0].grid(alpha=0.3)
axes[0].legend(frameon=False)

# ---------- Composite Trapezoid ----------
axes[1].plot(x, y, color="royalblue", lw=2, label=r"$f(x)=x^2$")
axes[1].fill_between(x, y, color="lightgray", alpha=0.3)
for i in range(k):
    xs = [x_points[i], x_points[i+1]]
    ys = [f(x_points[i]), f(x_points[i+1])]
    axes[1].fill_between(xs, [0, 0], ys, alpha=0.3, color='mediumseagreen', edgecolor='seagreen')
axes[1].set_title(f"Composite Trapezoid (T₃ = {T3:.4f})", fontsize=12)
axes[1].set_xlabel("x")
axes[1].grid(alpha=0.3)
axes[1].legend(frameon=False)

# ---------- Overall Figure ----------
fig.suptitle(
    "Exercise 7 — Comparing Composite Midpoint and Trapezoid Rules\n"
    r"True Value: $\int_0^2 x^2 dx = 8/3 = 2.6667$",
    fontsize=14, weight='bold'
)
plt.tight_layout()
plt.show()
