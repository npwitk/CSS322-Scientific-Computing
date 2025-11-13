import numpy as np
import matplotlib.pyplot as plt

# Matrix and initial vector
A = np.array([[3, -1],
              [-1, 3]])
x = np.array([1., 0.])

# For storing results
points = [x.copy()]

# Perform Power Method (un-normalized)
for k in range(10):
    x = A @ x
    points.append(x.copy())

# Convert to numpy array
points = np.array(points)

# --- Plot ---
plt.figure(figsize=(7,7))
plt.axhline(0, color='black', lw=1)
plt.axvline(0, color='black', lw=1)
plt.gca().set_aspect('equal')

# Plot the direction of the dominant eigenvector (for λ=4)
eigvals, eigvecs = np.linalg.eig(A)
v = eigvecs[:, np.argmax(eigvals)]
plt.plot([-v[0]*10, v[0]*10], [-v[1]*10, v[1]*10], 'r--', label="Eigenvector direction (λ=4)")

# Plot each iteration point
plt.scatter(points[:,0], points[:,1], color='blue', label='x^(k) points')
plt.quiver(points[:-1,0], points[:-1,1],
           points[1:,0]-points[:-1,0], points[1:,1]-points[:-1,1],
           angles='xy', scale_units='xy', scale=1, color='blue', alpha=0.6)

# Annotate iterations
for i, (px, py) in enumerate(points):
    plt.text(px*1.05, py*1.05, f"k={i}", fontsize=9)

# Labels and limits
plt.xlim(-6e5, 6e5)
plt.ylim(-6e5, 6e5)
plt.title("Power Method (Unnormalized) Trajectory in 2D Space")
plt.xlabel("x₁ component")
plt.ylabel("x₂ component")
plt.legend()
plt.show()
