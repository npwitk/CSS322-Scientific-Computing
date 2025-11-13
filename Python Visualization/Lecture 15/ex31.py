import numpy as np
import matplotlib.pyplot as plt

# Matrix and initial vector
A = np.array([[3, -1],
              [-1, 3]])
x = np.array([1., 0.])

# Normalize the first vector
x /= np.linalg.norm(x)

# Store all iterations
points = [x.copy()]
lambdas = []

# Perform normalized Power Method
for k in range(10):
    y = A @ x
    lam = (x @ y) / (x @ x)        # Rayleigh quotient
    lambdas.append(lam)
    x = y / np.linalg.norm(y)      # normalize
    points.append(x.copy())

points = np.array(points)

# Eigenvectors for reference
eigvals, eigvecs = np.linalg.eig(A)
v_dom = eigvecs[:, np.argmax(eigvals)]
v_dom /= np.linalg.norm(v_dom)

# --- Plot ---
plt.figure(figsize=(7,7))
plt.axhline(0, color='black', lw=1)
plt.axvline(0, color='black', lw=1)
plt.gca().set_aspect('equal')

# Unit circle for visualization
circle = plt.Circle((0, 0), 1, color='gray', fill=False, linestyle=':')
plt.gca().add_patch(circle)

# Dominant eigenvector direction
plt.plot([-v_dom[0], v_dom[0]], [-v_dom[1], v_dom[1]], 'r--', label="Eigenvector (λ=4)")

# Iteration points
plt.scatter(points[:,0], points[:,1], color='blue', zorder=3, label='x^(k)')
plt.quiver(points[:-1,0], points[:-1,1],
           points[1:,0]-points[:-1,0], points[1:,1]-points[:-1,1],
           angles='xy', scale_units='xy', scale=1, color='blue', alpha=0.6)

# Annotate iterations
for i, (px, py) in enumerate(points):
    plt.text(px*1.08, py*1.08, f"k={i}", fontsize=9)

plt.title("Normalized Power Method (Direction Convergence)")
plt.xlabel("x₁")
plt.ylabel("x₂")
plt.legend()
plt.show()

# --- Eigenvalue convergence ---
plt.figure(figsize=(6,3))
plt.plot(range(1, len(lambdas)+1), lambdas, 'o-')
plt.axhline(4, color='r', linestyle='--', label='True λ₁=4')
plt.title("Rayleigh Quotient Convergence to Dominant Eigenvalue")
plt.xlabel("Iteration")
plt.ylabel("λ estimate")
plt.legend()
plt.show()
