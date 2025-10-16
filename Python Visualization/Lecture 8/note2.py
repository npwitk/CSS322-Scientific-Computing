import numpy as np
import matplotlib.pyplot as plt

# Define functions
f_x = lambda x: x**2            # Original function on [0,2]
f_t = lambda t: (t + 1)**2      # Transformed function on [-1,1]

# Ranges
x = np.linspace(-0.5, 2.5, 400)
t = np.linspace(-1.5, 1.5, 400)

# Compute corresponding y values
y_x = f_x(x)
y_t = f_t(t)

# Create figure
fig, axes = plt.subplots(1, 2, figsize=(11, 4.5))

# --- Left plot: Original integral [0,2] ---
axes[0].plot(x, y_x, color="royalblue", linewidth=2, label=r"$f(x)=x^2$")
axes[0].fill_between(x, y_x, where=(x>=0)&(x<=2), color="lightgray", alpha=0.4)
axes[0].axvline(0, color="gray", linestyle="--")
axes[0].axvline(2, color="gray", linestyle="--")
axes[0].text(1, 4.2, r"$\int_0^2 x^2 dx$", ha="center", fontsize=12)
axes[0].set_xlim(-0.5, 2.5)
axes[0].set_ylim(0, 5)
axes[0].set_xlabel("x")
axes[0].set_ylabel("f(x)")
axes[0].set_title("Original integral on [0, 2]")
axes[0].legend(frameon=False)

# --- Right plot: Transformed integral [-1,1] ---
axes[1].plot(t, y_t, color="orange", linewidth=2, label=r"$f(t)=(t+1)^2$")
axes[1].fill_between(t, y_t, where=(t>=-1)&(t<=1), color="lightgray", alpha=0.4)
axes[1].axvline(-1, color="gray", linestyle="--")
axes[1].axvline(1, color="gray", linestyle="--")
axes[1].text(0, 4.2, r"$\int_{-1}^{1} (t+1)^2 dt$", ha="center", fontsize=12)
axes[1].set_xlim(-1.5, 1.5)
axes[1].set_ylim(0, 5)
axes[1].set_xlabel("t")
axes[1].set_ylabel("f(t)")
axes[1].set_title("Transformed integral on [-1, 1]")
axes[1].legend(frameon=False)

# --- Overall title ---
fig.suptitle("Same Area, Different Variable: Transforming [0,2] → [-1,1]", fontsize=14, weight="bold")

plt.tight_layout()
plt.show()
