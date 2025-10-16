import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import quad

# Function and true integral
f = lambda x: np.exp(-x) * np.sin(x)
a, b = 0, 2*np.pi
true_value, _ = quad(f, a, b)  # high-precision reference

# Midpoint rule implementation
def midpoint_integral(f, a, b, n):
    xs = np.linspace(a, b, n+1)
    mids = (xs[:-1] + xs[1:]) / 2
    widths = (b - a) / n
    return np.sum(f(mids) * widths)

# Compute errors for increasing n
n_values = np.array([1, 2, 4, 8, 16, 32, 64])
approx_values = np.array([midpoint_integral(f, a, b, n) for n in n_values])
errors = np.abs(approx_values - true_value)

# Plot function + rectangles for a sample n
def plot_midpoint(n, ax):
    xs = np.linspace(a, b, n+1)
    mids = (xs[:-1] + xs[1:]) / 2
    widths = (b - a)/n
    ax.plot(np.linspace(a, b, 400), f(np.linspace(a, b, 400)), 'royalblue', lw=2)
    for m in mids:
        ax.bar(m, f(m), width=widths, align="center", alpha=0.3, color="orange", edgecolor="darkorange")
    ax.fill_between(np.linspace(a, b, 400), f(np.linspace(a, b, 400)), color="lightgray", alpha=0.2)
    ax.set_xlim(a, b)
    ax.set_title(f"{n} subintervals → Approx = {midpoint_integral(f,a,b,n):.5f}")
    ax.grid(alpha=0.3)

# Create figure with two parts
fig, axes = plt.subplots(1, 2, figsize=(13,4.5))

# (1) Show rectangles for n=4
plot_midpoint(4, axes[0])
axes[0].set_xlabel("x")
axes[0].set_ylabel("f(x)")

# (2) Error vs n plot
axes[1].plot(n_values, errors, marker="o", color="crimson", lw=2)
axes[1].set_xscale("log")
axes[1].set_yscale("log")
axes[1].set_xlabel("Number of subintervals (n)")
axes[1].set_ylabel("Absolute Error")
axes[1].set_title("Error decreases as we subdivide")
axes[1].grid(True, which="both", ls="--", alpha=0.5)

fig.suptitle("Improving Accuracy by Subdividing — Midpoint Rule Example", fontsize=14, weight="bold")
plt.tight_layout()
plt.show()
