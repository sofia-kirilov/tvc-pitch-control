"""Stage 4: PID. The I term removes the leftover error and rejects a steady side torque.
The price is more overshoot on the step, which stage 5 deals with."""

from common import D_FILTER, DEG, KD, KI, KP, STEP, plt, print_metrics, save

from rocketpid import PID, pid_controller, simulate, step_metrics

GUST_TIME = 2.0
GUST = 0.08  # N m, e.g. steady wind or a misaligned motor


def gust(t):
    return GUST if t >= GUST_TIME else 0.0


fig, ax = plt.subplots(figsize=(7, 4))
for label, ki in (("PD", 0.0), ("PID", KI)):
    pid = PID(KP, ki, KD, d_filter_tau=D_FILTER)
    res = simulate(pid_controller(pid), setpoint=STEP, disturbance=gust, t_end=5.0)
    before = res.t < GUST_TIME
    print_metrics(label + " (before gust)", step_metrics(res.t[before], res.theta[before], STEP))
    print(f"{label:<28} error at end, with gust: {(STEP - res.theta[-200:].mean()) / DEG:+.3f} deg")
    ax.plot(res.t, res.theta / DEG, label=label)

ax.axhline(STEP / DEG, color="k", ls="--", lw=0.8, label="setpoint")
ax.axvline(GUST_TIME, color="gray", ls=":", lw=0.8)
ax.text(GUST_TIME + 0.05, 0.3, "side torque starts")
ax.set_xlabel("time (s)")
ax.set_ylabel("pitch (deg)")
ax.set_title("PID vs PD: 5 deg step, then a constant disturbance")
ax.legend()
ax.grid(True, alpha=0.3)
save(fig, "04_pid_control.png")
