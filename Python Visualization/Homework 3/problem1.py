import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle, Polygon
from matplotlib.animation import FuncAnimation, PillowWriter

# Define function and interval
def f(x):
    return 2 * x

a, b = 2.0, 3.0
m = (a + b) / 2.0
fa, fm, fb = f(a), f(m), f(b)

# Check exact integral value
true_area = b**2 - a**2
print(f"True integral value = {true_area}")

# Create plot
fig, ax = plt.subplots(figsize=(7, 4.5))
X = np.linspace(1.8, 3.2, 400)
ax.plot(X, f(X), lw=2, label="$f(x)=2x$")
x_fill = np.linspace(a, b, 200)
ax.fill_between(x_fill, f(x_fill), 0, alpha=0.15, label="True area")

# Shapes for rules
mid_rect = Rectangle((a, 0), width=b - a, height=fm, alpha=0.35, visible=False)
ax.add_patch(mid_rect)
trap_poly = Polygon([[a, 0], [a, fa], [b, fb], [b, 0]], closed=True, alpha=0.35, visible=False)
ax.add_patch(trap_poly)

# Simpson polynomial
coeffs = np.polyfit([a, m, b], [fa, fm, fb], deg=2)
p2 = np.poly1d(coeffs)
x_sim = np.linspace(a, b, 400)
sim_fill = [None]  # to store artist

# Text
method_text = ax.text(0.02, 0.95, "", transform=ax.transAxes, va="top")
formula_text = ax.text(0.02, 0.82, "", transform=ax.transAxes, va="top")
value_text = ax.text(0.02, 0.68, "", transform=ax.transAxes, va="top")
ax.text(0.02, 0.55, "True value: $\\int_2^3 2x\\,dx = 5$", transform=ax.transAxes, va="top")

# Axis styling
ax.set_xlim(1.9, 3.1)
ax.set_ylim(0, 7)
ax.set_xlabel("x")
ax.set_ylabel("y")
ax.set_title("Approximating $\\int_2^3 2x\\,dx$")
ax.legend(loc="lower right")

# Rule results
M = (b - a) * f(m)
T = (b - a) / 2 * (fa + fb)
S = (b - a) / 6 * (fa + 4 * fm + fb)

frames = 90
phase_len = frames // 3

def init():
    mid_rect.set_visible(False)
    trap_poly.set_visible(False)
    if sim_fill[0] is not None:
        sim_fill[0].remove()
        sim_fill[0] = None
    return (mid_rect, trap_poly)

def update(frame):
    phase = frame // phase_len
    alpha = min(1.0, (frame % phase_len) / (phase_len * 0.6))
    mid_rect.set_visible(False)
    trap_poly.set_visible(False)
    if sim_fill[0] is not None:
        sim_fill[0].remove()
        sim_fill[0] = None

    if phase == 0:
        mid_rect.set_alpha(alpha * 0.35 + 0.01)
        mid_rect.set_visible(True)
        method_text.set_text("Midpoint Rule")
        formula_text.set_text(r"$M(f)=(b-a)f((a+b)/2)$")
        value_text.set_text(f"$M(f)=1\\times f(2.5)={M:.1f}$")
    elif phase == 1:
        trap_poly.set_alpha(alpha * 0.35 + 0.01)
        trap_poly.set_visible(True)
        method_text.set_text("Trapezoid Rule")
        formula_text.set_text(r"$T(f)=\frac{b-a}{2}(f(a)+f(b))$")
        value_text.set_text(f"$T(f)=0.5(4+6)={T:.1f}$")
    else:
        sim_fill[0] = ax.fill_between(x_sim, p2(x_sim), 0, alpha=alpha * 0.35 + 0.01)
        method_text.set_text("Simpson's Rule")
        formula_text.set_text(r"$S(f)=\frac{b-a}{6}(f(a)+4f((a+b)/2)+f(b))$")
        value_text.set_text(f"$S(f)=1/6(4+20+6)={S:.1f}$")
    return (mid_rect, trap_poly)

# Animate
anim = FuncAnimation(fig, update, frames=frames, init_func=init, interval=60)
writer = PillowWriter(fps=15)
anim.save("integration_rules_2x_2to3.gif", writer=writer)
plt.show()
