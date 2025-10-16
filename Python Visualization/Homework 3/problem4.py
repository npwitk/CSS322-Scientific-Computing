import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Polygon, Rectangle
from matplotlib.animation import FuncAnimation
import matplotlib.cm as cm

# --- Interactive setup ---
print("Approximate ∫_{-2}^2 x dx using composite midpoint & trapezoid rules.")
k = int(input("Enter number of subintervals (e.g., 2, 4, 6): "))

# --- Function and parameters ---
def f(x): return x
a, b = -2, 2
h = (b - a) / k
x_points = np.linspace(a, b, k + 1)
midpoints = (x_points[:-1] + x_points[1:]) / 2

# --- Figure setup ---
fig, ax = plt.subplots(figsize=(7, 4.5))
X = np.linspace(a - 0.5, b + 0.5, 400)
ax.plot(X, f(X), 'b', lw=2, label="$f(x)=x$")
ax.fill_between(np.linspace(a, b, 400), f(np.linspace(a, b, 400)), 0, alpha=0.1, color='blue')

ax.axhline(0, color='black', lw=1)
ax.axvline(0, color='black', lw=0.8)

# --- Midpoint rectangles (blue) ---
rects = []
for m in midpoints:
    rect = Rectangle((m - h / 2, 0), h, f(m), color='royalblue', alpha=0.35, visible=False)
    ax.add_patch(rect)
    rects.append(rect)

# --- Trapezoids (different colors) ---
colors = cm.plasma(np.linspace(0.1, 0.9, k))
trap_patches = []
for i in range(k):
    trap = Polygon([[x_points[i], 0], [x_points[i], f(x_points[i])],
                    [x_points[i+1], f(x_points[i+1])], [x_points[i+1], 0]],
                   closed=True, color=colors[i], alpha=0.4, visible=False)
    ax.add_patch(trap)
    trap_patches.append(trap)

# --- Text annotations ---
method_text = ax.text(0.02, 0.95, "", transform=ax.transAxes, va="top")
formula_text = ax.text(0.02, 0.82, "", transform=ax.transAxes, va="top")
value_text = ax.text(0.02, 0.70, "", transform=ax.transAxes, va="top")
ax.text(0.02, 0.57, "True value: $\\int_{-2}^2 x\\,dx = 0$", transform=ax.transAxes, va="top")

# --- Style ---
ax.set_xlim(a - 0.5, b + 0.5)
ax.set_ylim(-2.5, 2.5)
ax.set_xlabel("x")
ax.set_ylabel("y")
ax.set_title(f"Composite Midpoint & Trapezoid Rules for $\\int_{{-2}}^2 x\\,dx$ (k={k})")
ax.legend(loc="lower right")

frames = 60
half = frames // 2

# --- Animation setup ---
def init():
    for rect in rects: rect.set_visible(False)
    for trap in trap_patches: trap.set_visible(False)
    method_text.set_text("")
    formula_text.set_text("")
    value_text.set_text("")
    return rects + trap_patches

def update(frame):
    if frame < half:
        alpha = min(1, frame / (half * 0.6)) * 0.35
        for rect in rects:
            rect.set_visible(True)
            rect.set_alpha(alpha)
        for trap in trap_patches:
            trap.set_visible(False)
        method_text.set_text("Composite Midpoint Rule ($M_k$)")
        formula_text.set_text(r"$M_k(f) = h\sum f((x_{j-1}+x_j)/2)$")
        value_text.set_text("$M_k(f) = 0$")
    else:
        alpha = min(1, (frame - half) / (half * 0.6)) * 0.45
        for rect in rects:
            rect.set_visible(False)
        for trap in trap_patches:
            trap.set_visible(True)
            trap.set_alpha(alpha)
        method_text.set_text("Composite Trapezoid Rule ($T_k$)")
        formula_text.set_text(r"$T_k(f) = h(\frac{1}{2}f(x_0)+\sum f(x_j)+\frac{1}{2}f(x_k))$")
        value_text.set_text("$T_k(f) = 0$")
    return rects + trap_patches

# --- Run animation ---
anim = FuncAnimation(fig, update, frames=frames, init_func=init, interval=80)
plt.show()
