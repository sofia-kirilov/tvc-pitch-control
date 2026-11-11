"""Stage 6: tune by sweeping gains and scoring each run, instead of by feel.

Score = ITAE (integral of time * |error|) over a 5 deg step followed by a
side-torque disturbance, with sensor noise and all actuator limits on.

Scoring on the nominal rocket alone rewards ever-higher gains. So each
candidate is scored on three rockets (nominal, strong motor with a slow
servo, weak motor with more aero torque) and the scores are averaged.

Order: sweep ki, then a kp/kd grid, then ki again, then the setpoint weight.
"""

import numpy as np
from common import D_FILTER, DEG, KD, KI, KP, STEP, plt, print_metrics, save

from rocketpid import PID, RocketParams, itae, pid_controller, simulate, step_metrics

PLANTS = [
    RocketParams(),
    RocketParams(thrust=18.0, servo_tau=0.045),
    RocketParams(thrust=12.0, c_aero=0.75),
]


def gust(t):
    return 0.08 if t >= 1.5 else 0.0


def run(kp, ki, kd, weight, p, seed=0):
    pid = PID(kp, ki, kd, out_limit=p.delta_max, d_filter_tau=D_FILTER, setpoint_weight=weight)
    return simulate(
        pid_controller(pid), p, setpoint=STEP, disturbance=gust,
        angle_noise=0.1 * DEG, t_end=3.0, dt=2e-3, seed=seed,
    )


def score(kp, ki, kd, weight):
    total = 0.0
    for k, p in enumerate(PLANTS):
        res = run(kp, ki, kd, weight, p, seed=k)
        total += itae(res.t, res.theta, STEP)
    return total / len(PLANTS)


def best(values, fn):
    scores = [fn(v) for v in values]
    return values[int(np.argmin(scores))]


kis = np.linspace(0.5, 10.0, 12)
weights = np.array([0.2, 0.3, 0.4, 0.5, 0.6, 0.8, 1.0])
kps = np.linspace(0.4, 3.0, 12)
kds = np.linspace(0.05, 0.40, 12)

ki = best(kis, lambda v: score(KP, v, KD, 0.5))
grid = np.array([[score(kp, ki, kd, 0.5) for kp in kps] for kd in kds])
i, j = np.unravel_index(np.argmin(grid), grid.shape)
kp, kd = kps[j], kds[i]
ki = best(kis, lambda v: score(kp, v, kd, 0.5))
weight = best(weights, lambda v: score(kp, ki, kd, v))

print(f"baseline  kp={KP:.3f} ki={KI:.2f} kd={KD:.3f} weight=1.0  score={score(KP, KI, KD, 1.0):.4f}")
print(f"tuned     kp={kp:.3f} ki={ki:.2f} kd={kd:.3f} weight={weight:.1f}  score={score(kp, ki, kd, weight):.4f}")

fig, axes = plt.subplots(1, 2, figsize=(11, 4.2))
im = axes[0].pcolormesh(kps, kds, np.log10(grid), shading="nearest")
axes[0].plot(KP, KD, "wo", label="baseline")
axes[0].plot(kp, kd, "w*", ms=12, label="best")
axes[0].set_xlabel("kp")
axes[0].set_ylabel("kd")
axes[0].set_title("log10(score) over kp and kd")
axes[0].legend()
fig.colorbar(im, ax=axes[0])

for label, gains in (("baseline", (KP, KI, KD, 1.0)), ("tuned", (kp, ki, kd, weight))):
    res = run(*gains, PLANTS[0])
    first = res.t < 1.5
    print_metrics(label, step_metrics(res.t[first], res.theta[first], STEP, band=0.05))
    axes[1].plot(res.t, res.theta / DEG, label=label)
axes[1].axhline(STEP / DEG, color="k", ls="--", lw=0.8)
axes[1].set_xlabel("time (s)")
axes[1].set_ylabel("pitch (deg)")
axes[1].set_title("Nominal rocket: step at 0 s, disturbance at 1.5 s")
axes[1].legend()
axes[1].grid(True, alpha=0.3)
save(fig, "06_tuning.png")
