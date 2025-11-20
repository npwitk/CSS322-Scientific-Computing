import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import quad

# Target function
def f(x):
    return np.exp(-x**2)

# Composite Midpoint Rule
def midpoint_rule(a, b, n):
    h = (b - a) / n
    mids = a + h*(np.arange(n) + 0.5)
    return h * np.sum(f(mids)), mids

# Composite Trapezoid Rule
def trapezoid_rule(a, b, n):
    h = (b - a) / n
    x = np.linspace(a, b, n+1)
    return h * (0.5*f(x[0]) + np.sum(f(x[1:-1])) + 0.5*f(x[-1])), x

# Composite Simpson Rule
def simpson_rule(a, b, n):
    if n % 2 == 1:
        raise ValueError("Simpson's rule requires EVEN n.")
    h = (b - a) / n
    x = np.linspace(a, b, n+1)
    S = f(x[0]) + f(x[-1]) + 4*np.sum(f(x[1:-1:2])) + 2*np.sum(f(x[2:-2:2]))
    return (h/3) * S, x


# ---- Visualization ----
def plot_visualization(rule, approx, nodes, a=0, b=1):
    # High resolution curve for true area
    X = np.linspace(a, b, 1000)
    Y = f(X)

    plt.figure(figsize=(10, 6))

    # --- TRUE AREA SHADED ---
    plt.fill_between(X, Y, color="lightgray", alpha=0.6, label="True Area")

    # --- APPROX AREA SHADED ---
    if rule == "midpoint":
        h = (b - a) / len(nodes)
        for m in nodes:
            x_left, x_right = m - h/2, m + h/2
            plt.fill_between([x_left, x_right], f(m), color="skyblue", alpha=0.55)
        plt.scatter(nodes, f(nodes), color="blue", label="Midpoints", zorder=3)

    elif rule == "trapezoid":
        for i in range(len(nodes)-1):
            xs = np.linspace(nodes[i], nodes[i+1], 20)
            ys = np.interp(xs, [nodes[i], nodes[i+1]], [f(nodes[i]), f(nodes[i+1])])
            plt.fill_between(xs, ys, color="lightgreen", alpha=0.55)
        plt.scatter(nodes, f(nodes), color="green", label="Trapezoid Nodes", zorder=3)

    elif rule == "simpson":
        for i in range(0, len(nodes)-1, 2):
            xs = np.linspace(nodes[i], nodes[i+2], 200)

            # Quadratic interpolation for Simpson’s rule
            x0, x1, x2 = nodes[i], nodes[i+1], nodes[i+2]
            y0, y1, y2 = f(x0), f(x1), f(x2)

            A = np.array([
                [x0**2, x0, 1],
                [x1**2, x1, 1],
                [x2**2, x2, 1]
            ])
            coef = np.linalg.solve(A, np.array([y0, y1, y2]))
            poly = lambda x: coef[0]*x**2 + coef[1]*x + coef[2]

            plt.fill_between(xs, poly(xs), color="violet", alpha=0.45)

        plt.scatter(nodes, f(nodes), color="purple", label="Simpson Nodes", zorder=3)

    # --- True function curve ---
    plt.plot(X, Y, 'k', linewidth=2, label="f(x) = e^{-x^2}")

    # --- Compute and format errors ---
    true_val, _ = quad(f, a, b)
    abs_error = abs(approx - true_val)
    percent_error = abs_error / true_val * 100

    # --- Title with exact values & errors ---
    plt.title(
        f"{rule.capitalize()} Rule\n"
        f"Approx = {approx:.6f}   |   True = {true_val:.6f}\n"
        f"Abs Error = {abs_error:.6e}   |   Percent Error = {percent_error:.4f}%"
    )

    plt.xlabel("x")
    plt.ylabel("f(x)")
    plt.grid(alpha=0.25)
    plt.legend()
    plt.show()


# ---- Main Program ----
if __name__ == "__main__":
    print("\n=== Numerical Integration Visual Comparison ===")
    print("Available rules: midpoint, trapezoid, simpson\n")

    rule = input("Choose a rule: ").strip().lower()
    n = int(input("Number of subintervals n: "))

    a, b = 0, 1

    if rule == "m":
        approx, mids = midpoint_rule(a, b, n)
        plot_visualization("midpoint", approx, mids)

    elif rule == "t":
        approx, nodes = trapezoid_rule(a, b, n)
        plot_visualization("trapezoid", approx, nodes)

    elif rule == "s":
        approx, nodes = simpson_rule(a, b, n)
        plot_visualization("simpson", approx, nodes)

    else:
        print("Invalid rule.")
