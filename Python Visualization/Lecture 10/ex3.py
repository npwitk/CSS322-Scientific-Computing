import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

# ======================================================
# Given data
# ======================================================
A = np.array([
    [1, 2],
    [0, -2.4],
    [0, 1.8]
])
b = np.array([-1, 0, 1])

# Compute least squares solution using QR factorization
Q, R = np.linalg.qr(A)
x = np.linalg.solve(R[:2, :2], Q.T[:2, :] @ b)
Ax = A @ x
r = b - Ax

# ======================================================
# Generate plane for column space (span of A)
# ======================================================
# Create a grid of linear combinations of A[:,0] and A[:,1]
u = np.linspace(-1.5, 1.5, 20)
v = np.linspace(-1.5, 1.5, 20)
U, V = np.meshgrid(u, v)
plane_points = np.outer(U.ravel(), A[:,0]) + np.outer(V.ravel(), A[:,1])
X, Y, Z = plane_points[:,0].reshape(U.shape), plane_points[:,1].reshape(U.shape), plane_points[:,2].reshape(U.shape)

# ======================================================
# 3D Plot
# ======================================================
fig = plt.figure(figsize=(8,6))
ax = fig.add_subplot(111, projection='3d')

# Column space plane
ax.plot_surface(X, Y, Z, color='lightgray', alpha=0.4, label='col(A)')

# Vectors
origin = np.zeros(3)
ax.quiver(*origin, *b, color='k', lw=2, label='b')
ax.quiver(*origin, *Ax, color='r', lw=2, label='Projection Ax')
ax.quiver(*Ax, *(r), color='b', lw=1.5, linestyle='dashed', label='Residual (b - Ax)')

# Points
ax.scatter(*Ax, color='red', s=40)
ax.scatter(*b, color='black', s=40)

# Labels
ax.text(*b, " b", color='k', fontsize=10)
ax.text(*Ax, " Ax", color='r', fontsize=10)
ax.text(*(Ax + 0.5*r), " Residual", color='b', fontsize=10)

# Axes settings
ax.set_xlabel("x₁")
ax.set_ylabel("x₂")
ax.set_zlabel("x₃")
ax.set_title("Least Squares via QR Factorization")
ax.legend()
ax.view_init(elev=25, azim=-50)
plt.tight_layout()
plt.show()

# ======================================================
# Print numerical results
# ======================================================
print("Least Squares Solution (QR):")
print("x =", np.round(x, 4))
print("A x =", np.round(Ax, 4))
print("Residual =", np.round(r, 4))
print("||b - Ax||₂ =", np.linalg.norm(r))
