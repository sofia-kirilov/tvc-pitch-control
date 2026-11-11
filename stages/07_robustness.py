"""Stage 7: does the tuning still work when the real rocket is not the model?

Part A: Monte Carlo. Randomize thrust, inertia, aero torque and servo speed
        and rerun the same controller 100 times.
Part B: Scale the whole loop gain up and down to find where it goes unstable
        (a rough gain margin).
"""

import numpy as np
from common import DEG, STEP, plt, save, tuned_pid

from rocketpid import RocketParams, pid_controller, simulate, step_metrics

RUNS = 100
rng = np.random.default_rng(1)


def controller(scale=1.0):
    return pid_controller(tuned_pid(RocketParams().delta_max, scale))


# Part A
fig, axes = plt.subplots(1, 2, figsize=(11, 4.2))
overshoots, settles, failed = [], [], 0
for k in range(RUNS):
    p = RocketParams(
        thrust=15.0 * rng.uniform(0.8, 1.2),
        inertia=0.05 * rng.uniform(0.8, 1.2),
        c_aero=0.5 * rng.uniform(0.5, 1.5),
        servo_tau=0.03 * rng.uniform(0.7, 1.5),
    )
    res = simulate(controller(), p, setpoint=STEP, angle_noise=0.05 * DEG,
                   t_end=3.0, dt=2e-3, seed=k)
    m = step_metrics(res.t, res.theta, STEP, band=0.05)
    if np.isnan(m["settle"]):
        failed += 1
    else:
        overshoots.append(m["overshoot"])
        settles.append(m["settle"])
    axes[0].plot(res.t, res.theta / DEG, color="C0", alpha=0.15, lw=0.8)

print(f"Monte Carlo, {RUNS} runs: {failed} failed to settle within 5% in 3 s")
print(f"  overshoot   median {np.median(overshoots):.1f} %   worst {np.max(overshoots):.1f} %")
print(f"  settle time median {np.median(settles):.2f} s   worst {np.max(settles):.2f} s")
axes[0].axhline(STEP / DEG, color="k", ls="--", lw=0.8)
axes[0].set_xlabel("time (s)")
axes[0].set_ylabel("pitch (deg)")
axes[0].set_title(f"{RUNS} runs: thrust/inertia +-20%, aero +-50%, servo -30/+50%")
axes[0].grid(True, alpha=0.3)

# Part B
scales = np.geomspace(0.05, 20, 40)
peak = []
for s in scales:
    res = simulate(controller(s), setpoint=STEP, t_end=4.0, dt=2e-3)
    late = res.theta[res.t > 3.0]
    peak.append(np.inf if np.any(np.isnan(late)) else np.max(np.abs(late - STEP)) / DEG)
peak = np.array(peak)
stable = scales[peak < 0.25]
print(f"loop gain can be scaled from about {stable.min():.2f}x to {stable.max():.1f}x "
      f"before the 5 deg step stops settling")
axes[1].loglog(scales, np.clip(peak, 1e-4, 100), "o-", ms=3)
axes[1].axhline(0.25, color="k", ls="--", lw=0.8)
axes[1].set_xlabel("loop gain multiplier")
axes[1].set_ylabel("worst error after 3 s (deg)")
axes[1].set_title("Gain margin sweep")
axes[1].grid(True, which="both", alpha=0.3)
save(fig, "07_robustness.png")
