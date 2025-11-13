import numpy as np
import matplotlib.pyplot as plt
from matplotlib.widgets import Slider

# ======================================================
# Generate a larger structured matrix (e.g., 15×15)
# ======================================================
np.random.seed(1)
rows, cols = 15, 15
A = np.add.outer(np.linspace(0, 10, rows), np.sin(np.linspace(0, 3*np.pi, cols)))  # smooth pattern
A += 0.8 * np.random.randn(rows, cols)  # add some noise

# SVD decomposition
U, S, VT = np.linalg.svd(A, full_matrices=False)
ranks = np.arange(1, len(S) + 1)

# ======================================================
# Function to compute rank-k approximation
# ======================================================
def compute_Ak(k):
    return (U[:, :k] * S[:k]) @ VT[:k, :]

# ======================================================
# Initial setup
# ======================================================
fig, axes = plt.subplots(1, 2, figsize=(12, 6))
plt.subplots_adjust(bottom=0.25)

# Initial rank
k_init = 2
A_k = compute_Ak(k_init)

# Original Matrix
im0 = axes[0].imshow(A, cmap='viridis', aspect='auto')
axes[0].set_title("Original Matrix A")
axes[0].set_xlabel("Columns")
axes[0].set_ylabel("Rows")

# Rank-k Approximation
im1 = axes[1].imshow(A_k, cmap='viridis', aspect='auto', vmin=np.min(A), vmax=np.max(A))
error = np.linalg.norm(A - A_k, 'fro')
axes[1].set_title(f"Rank-{k_init} Approximation\nError = {error:.3f}")
axes[1].set_xlabel("Columns")
axes[1].set_ylabel("Rows")

# ======================================================
# Add colorbars with meaning labels
# ======================================================
cbar0 = fig.colorbar(im0, ax=axes[0], fraction=0.046, pad=0.04)
cbar1 = fig.colorbar(im1, ax=axes[1], fraction=0.046, pad=0.04)
cbar0.set_label("Value Magnitude (lighter = larger)", fontsize=9)
cbar1.set_label("Value Magnitude (lighter = larger)", fontsize=9)

# ======================================================
# Slider setup
# ======================================================
ax_slider = plt.axes([0.25, 0.1, 0.5, 0.03])
rank_slider = Slider(ax_slider, 'Rank k', 1, len(S), valinit=k_init, valstep=1)

# ======================================================
# Update function
# ======================================================
def update(val):
    k = int(rank_slider.val)
    A_k = compute_Ak(k)
    im1.set_data(A_k)
    error = np.linalg.norm(A - A_k, 'fro')
    axes[1].set_title(f"Rank-{k} Approximation\nError = {error:.3f}")
    fig.canvas.draw_idle()

rank_slider.on_changed(update)

# ======================================================
# Main title and layout
# ======================================================
plt.suptitle("Interactive SVD Low-Rank Approximation", fontsize=12, y=0.97)
plt.show()
