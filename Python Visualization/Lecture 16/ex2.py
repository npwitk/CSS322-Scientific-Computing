import numpy as np
import matplotlib.pyplot as plt

# -------------------------------------------------------
# Exact solution u(t) = t^3  for the BVP u''=6t, u(0)=0,u(1)=1
# -------------------------------------------------------
def exact_solution(t):
    return t**3


# -------------------------------------------------------
# Finite Difference Solver
# -------------------------------------------------------
def finite_difference(n=None, h=None):
    # Determine n or h
    if h is not None:
        n = int(1/h) - 1
        h = 1/(n+1)
    else:
        h = 1/(n+1)

    t = np.linspace(0, 1, n+2)

    # Build tridiagonal system A y = b
    A = np.zeros((n, n))
    b = np.zeros(n)

    for i in range(n):
        A[i,i] = -2
        if i > 0:
            A[i,i-1] = 1
        if i < n-1:
            A[i,i+1] = 1

        t_i = (i+1) * h
        b[i] = 6 * t_i * h*h   # multiply both sides by h²

    # Boundary condition at the right endpoint: y_(n+1) = 1
    b[-1] -= 1

    # Solve the linear system
    y_internal = np.linalg.solve(A, b)
    y = np.concatenate(([0], y_internal, [1]))

    return t, y


# -------------------------------------------------------
# Program entry: user input
# -------------------------------------------------------
print("Finite Difference Visualization for u'' = 6t, u(0)=0, u(1)=1")
print("You can enter:")
print("  - n = number of interior points")
print("  - h = step size")
print()

mode = input("Enter mode (n / h): ").strip().lower()

if mode == "n":
    n = int(input("Enter number of interior points (e.g., 5, 10, 20): "))
    t_fd, y_fd = finite_difference(n=n)
    title = f"Finite Difference (n = {n} interior points)"

elif mode == "h":
    h = float(input("Enter step size h (e.g., 0.1, 0.05): "))
    t_fd, y_fd = finite_difference(h=h)
    title = f"Finite Difference (h = {h})"

else:
    print("Invalid input. Please enter 'n' or 'h'.")
    exit()


# -------------------------------------------------------
# Plot results
# -------------------------------------------------------
t_exact = np.linspace(0, 1, 300)
y_exact = exact_solution(t_exact)

plt.figure(figsize=(7,5))

# Different colors because you requested it ✔️
plt.plot(t_exact, y_exact, label="Exact Solution  $u(t)=t^3$")
plt.scatter(t_fd, y_fd, color='red', label="Finite Difference Points")

plt.title(title)
plt.xlabel("t")
plt.ylabel("u(t)")
plt.grid(True)
plt.legend()
plt.show()
