import numpy as np
import matplotlib.pyplot as plt

# ------------------------------
# Define system: Newton's 2nd law (free fall)
# ------------------------------
def f(t, u):
    y, v = u  # position, velocity
    dy = v
    dv = -9.8  # gravity acceleration (m/s^2)
    return np.array([dy, dv])

# ------------------------------
# Euler's Method
# ------------------------------
def euler_system(f, u0, t0, h, steps):
    t = [t0]
    u = [u0]
    for _ in range(steps):
        u_next = u[-1] + h * f(t[-1], u[-1])
        t.append(t[-1] + h)
        u.append(u_next)
    return np.array(t), np.array(u)

# ------------------------------
# Parameters
# ------------------------------
u0 = np.array([0.0, 15.0])  # start at height 0, initial velocity 15 m/s upward
t0 = 0.0
h = 0.05
t_end = 3.5
steps = int((t_end - t0) / h)

# ------------------------------
# Solve using Euler
# ------------------------------
t, u = euler_system(f, u0, t0, h, steps)
y = u[:,0]
v = u[:,1]

# ------------------------------
# True analytical solution for comparison
# ------------------------------
y_true = u0[0] + u0[1]*t - 0.5*9.8*t**2
v_true = u0[1] - 9.8*t

# ------------------------------
# Visualization
# ------------------------------
fig, axes = plt.subplots(1, 2, figsize=(12, 5))

# ---- (1) Position and Velocity vs Time ----
axes[0].plot(t, y_true, 'g--', label='True Position')
axes[0].plot(t, v_true, 'b--', label='True Velocity')
axes[0].plot(t, y, 'ro-', label="Euler Position")
axes[0].plot(t, v, 'ms--', label="Euler Velocity")
axes[0].axhline(0, color='k', lw=0.8)
axes[0].set_xlabel('Time (s)')
axes[0].set_ylabel('Value')
axes[0].set_title("Ball Thrown Upward (Euler vs True Solution)")
axes[0].legend()
axes[0].grid(True, linestyle='--', alpha=0.6)

# ---- (2) Phase Plane (Velocity vs Position) ----
axes[1].plot(y_true, v_true, 'g--', label='True Curve')
axes[1].plot(y, v, 'ro-', label='Euler Points')
axes[1].axhline(0, color='k', lw=0.8)
axes[1].axvline(0, color='k', lw=0.8)
axes[1].set_xlabel('Position y (m)')
axes[1].set_ylabel('Velocity v (m/s)')
axes[1].set_title('Phase Plane (v vs y)')
axes[1].legend()
axes[1].grid(True, linestyle='--', alpha=0.6)

plt.tight_layout()
plt.show()
