import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

# ----------------------------------------
# Given data
# ----------------------------------------
A = np.array([[0, 1.6],
              [0, 0],
              [0, -1.2]])
b = np.array([-1, 0, 1])

# SVD
U, S, VT = np.linalg.svd(A, full_matrices=True)

# Least squares solution (manual)
sigma1 = S[0]
u1 = U[:, 0]
v1 = VT.T[:, 0]
x = (u1.T @ b / sigma1) * v1

# Reconstruct Ax (projection)
Ax = A @ x

# Generate all points along the line span(A[:,1])
t = np.linspace(-1.5, 1.5, 50)
line_points = np.outer(A[:, 1], t)

# ----------------------------------------
# 3D Visualization
# ----------------------------------------
fig = plt.figure(figsize=(7,6))
ax = fig.add_subplot(111, projection='3d')

# Plot the column space line (span of A)
ax.plot(line_points[0], line_points[1], line_points[2], 'gray', lw=2, label='col(A)')

# Plot vectors
origin = np.zeros(3)
ax.quiver(*origin, *b, color='k', lw=2, label='b')
ax.quiver(*origin, *Ax, color='r', lw=2, label='Ax (projection)')
ax.quiver(*Ax, *(b - Ax), color='b', linestyle='dashed', lw=1.5, label='error: b - Ax')

# Annotations
ax.text(*b, 'b', color='k')
ax.text(*Ax, 'Ax', color='r')

# Axes labels and settings
ax.set_xlabel("x₁")
ax.set_ylabel("x₂")
ax.set_zlabel("x₃")
ax.set_title("Least Squares Projection via SVD")
ax.legend()
ax.view_init(elev=25, azim=-45)
plt.tight_layout()
plt.show()

# ----------------------------------------
# Print summary
# ----------------------------------------
print("Computed least-squares solution:")
print("x =", np.round(x, 4))
print("A x =", np.round(Ax, 4))
print("Residual (b - Ax) =", np.round(b - Ax, 4))
