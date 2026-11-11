"""Stage 2: P only. It pushes back against the tilt, but with nothing to damp the
motion the rocket swings back and forth and the swings grow. More gain makes it worse."""

import numpy as np
from common import DEG, STEP, plt, save

from rocketpid import PID, pid_controller, simulate

fig, ax = plt.subplots(figsize=(7, 4))
for kp in (0.3, 0.82, 2.0):
    pid = PID(kp)
    res = simulate(pid_controller(pid), setpoint=STEP, t_end=2.0)
    print(f"P, kp={kp}: largest pitch in 2 s = {np.nanmax(np.abs(res.theta)) / DEG:.1f} deg (target 5)")
    ax.plot(res.t, res.theta / DEG, label=f"kp = {kp}")

ax.axhline(STEP / DEG, color="k", ls="--", lw=0.8, label="setpoint")
ax.set_xlabel("time (s)")
ax.set_ylabel("pitch (deg)")
ax.set_ylim(-30, 40)
ax.set_title("P control: 5 deg step")
ax.legend()
ax.grid(True, alpha=0.3)
save(fig, "02_p_control.png")
