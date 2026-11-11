"""Stage 1: the plant with no controller. A 1 degree tilt grows until the rocket falls over."""

from common import DEG, plt, save

from rocketpid import RocketParams, linearize, simulate
import numpy as np

p = RocketParams()
A, _ = linearize(p)
print("open-loop poles:", np.round(np.linalg.eigvals(A), 2))
print("a pole with a positive real part means the plant is unstable")

res = simulate(None, p, t_end=1.5, theta0=1 * DEG)

fig, ax = plt.subplots(figsize=(7, 4))
ax.plot(res.t, res.theta / DEG)
ax.set_ylim(0, 90)
ax.set_xlabel("time (s)")
ax.set_ylabel("pitch (deg)")
ax.set_title("Open loop: 1 deg initial tilt, gimbal fixed at 0")
ax.grid(True, alpha=0.3)
save(fig, "01_open_loop.png")
