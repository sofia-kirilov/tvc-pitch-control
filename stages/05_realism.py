"""Stage 5: four things that go wrong with a textbook PID on real hardware, and the fix for each.

1. Integrator windup when the gimbal hits its travel limit
2. Derivative kick on a setpoint step
3. Sensor noise amplified by the derivative term
4. Overshoot on setpoint steps (fixed with setpoint weighting)
"""

import numpy as np
from common import D_FILTER, DEG, KD, KI, KP, STEP, plt, print_metrics, save

from rocketpid import PID, RocketParams, pid_controller, simulate, step_metrics

p = RocketParams()
fig, axes = plt.subplots(4, 2, figsize=(10, 12), sharex="row")

# 1. windup: a big step keeps the gimbal pinned at its limit for a while
big = 12 * DEG
for label, aw in (("no anti-windup", False), ("anti-windup", True)):
    pid = PID(KP, 6.0, 0.18, out_limit=p.delta_max, anti_windup=aw, d_filter_tau=D_FILTER)
    res = simulate(pid_controller(pid), p, setpoint=big, t_end=4.0)
    fell = np.isnan(res.theta).any()
    axes[0, 0].plot(res.t, res.theta / DEG, label=label)
    axes[0, 1].plot(res.t, res.delta / DEG, label=label)
    print(f"windup, {label:<16} " + ("rocket fell over" if fell else
          f"peak pitch {np.max(res.theta) / DEG:.1f} deg (target 12)"))
axes[0, 0].axhline(big / DEG, color="k", ls="--", lw=0.8)
axes[0, 0].set_ylim(0, 45)
axes[0, 0].set_title("1. Windup: 12 deg step, pitch")
axes[0, 1].set_title("gimbal angle (limit is 5 deg)")

# 2. derivative kick (no output clamp here, so the spike is visible)
for label, on_meas in (("D on error", False), ("D on measurement", True)):
    pid = PID(KP, KI, KD, d_on_measurement=on_meas)
    res = simulate(pid_controller(pid), p, setpoint=lambda t: STEP if t >= 0.2 else 0.0, t_end=1.5)
    axes[1, 0].plot(res.t, res.theta / DEG, label=label)
    axes[1, 1].plot(res.t, res.delta_cmd / DEG, label=label)
    print(f"kick, {label:<18} peak gimbal command {np.max(np.abs(res.delta_cmd)) / DEG:.1f} deg")
axes[1, 0].set_title("2. Derivative kick: 5 deg step, pitch")
axes[1, 1].set_title("gimbal command")

# 3. noise
noise = 0.05 * DEG
for label, tau in (("no D filter", 0.0), (f"D filter, {D_FILTER * 1000:.0f} ms", D_FILTER)):
    pid = PID(KP, KI, KD, out_limit=p.delta_max, d_filter_tau=tau)
    res = simulate(pid_controller(pid), p, setpoint=STEP, angle_noise=noise, t_end=3.0)
    axes[2, 0].plot(res.t, res.theta / DEG, label=label)
    axes[2, 1].plot(res.t, res.delta_cmd / DEG, label=label, lw=0.7)
    late = res.t > 1.5
    print(f"noise, {label:<17} gimbal command jitter {np.std(res.delta_cmd[late]) / DEG:.2f} deg rms")
axes[2, 0].set_title("3. Sensor noise (0.05 deg rms): pitch")
axes[2, 1].set_title("gimbal command")

# 4. setpoint weighting
for weight in (1.0, 0.5):
    label = f"setpoint weight {weight}"
    pid = PID(KP, KI, KD, out_limit=p.delta_max, d_filter_tau=D_FILTER, setpoint_weight=weight)
    res = simulate(pid_controller(pid), p, setpoint=STEP, t_end=3.0)
    print_metrics(label, step_metrics(res.t, res.theta, STEP))
    axes[3, 0].plot(res.t, res.theta / DEG, label=label)
    axes[3, 1].plot(res.t, res.delta_cmd / DEG, label=label)
axes[3, 0].axhline(STEP / DEG, color="k", ls="--", lw=0.8)
axes[3, 0].set_title("4. Overshoot: 5 deg step, pitch")
axes[3, 1].set_title("gimbal command")

for ax in axes.flat:
    ax.grid(True, alpha=0.3)
    ax.legend(fontsize=8)
    ax.set_ylabel("deg")
for ax in axes[3]:
    ax.set_xlabel("time (s)")
save(fig, "05_realism.png")
