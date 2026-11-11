"""Stage 9: Kalman filter. With a noisy angle sensor and no gyro, differentiating
the measurement is the weak point of PID. The filter blends the model with the
measurements to estimate angle, rate and the outside disturbance, and the
controller uses the angle and rate estimates.

Both controllers use the same kp, ki, kd. Only the source of angle and rate differs.
"""

import numpy as np
from common import DEG, KD_T, KI_T, KP_T, SP_WEIGHT, STEP, plt, save, tuned_pid

from rocketpid import KalmanFilter, RocketParams, pid_controller, simulate

p = RocketParams()
NOISE = 0.5 * DEG  # a poor angle sensor
CTRL_DT = 0.01


def make_kf_controller(log):
    kf = KalmanFilter(p, CTRL_DT, NOISE)
    state = dict(integral=0.0, delta_hat=0.0)

    def controller(t, r, theta_meas, omega_meas, dt):
        theta_hat, omega_hat, _ = kf.step(state["delta_hat"], theta_meas)
        log.append((t, theta_hat, omega_hat))
        e = r - theta_hat
        u = KP_T * (SP_WEIGHT * r - theta_hat) + state["integral"] - KD_T * omega_hat
        u_sat = max(-p.delta_max, min(p.delta_max, u))
        if u == u_sat:
            state["integral"] += KI_T * e * dt
        state["delta_hat"] += dt * (u_sat - state["delta_hat"]) / (p.servo_tau + dt)
        return u_sat

    return controller


def gust(t):
    return 0.08 if t >= 2.5 else 0.0


log = []
runs = (
    ("PID on raw sensor", pid_controller(tuned_pid(p.delta_max))),
    ("PID on Kalman estimate", make_kf_controller(log)),
)

fig, axes = plt.subplots(1, 3, figsize=(15, 4.2))
for label, ctrl in runs:
    res = simulate(ctrl, p, setpoint=STEP, disturbance=gust, angle_noise=NOISE,
                   t_end=5.0, ctrl_dt=CTRL_DT)
    steady = (res.t > 1.5) & (res.t < 2.5)
    print(f"{label:<24} pitch error {np.std(res.theta[steady] - STEP) / DEG:.3f} deg rms   "
          f"gimbal jitter {np.std(res.delta_cmd[steady]) / DEG:.2f} deg rms")
    axes[0].plot(res.t, res.theta / DEG, label=label)
    axes[1].plot(res.t, res.delta_cmd / DEG, label=label, lw=0.6)

log = np.array(log)
truth = np.interp(log[:, 0], res.t, res.theta)
meas = np.interp(log[:, 0], res.t, res.theta_meas)
print(f"angle error: raw sensor {np.std(meas - truth) / DEG:.3f} deg rms, "
      f"Kalman estimate {np.std(log[:, 1] - truth) / DEG:.3f} deg rms")
axes[2].plot(log[:, 0], meas / DEG, ".", ms=2, color="0.7", label="raw sensor")
axes[2].plot(log[:, 0], truth / DEG, "k", lw=1, label="truth")
axes[2].plot(log[:, 0], log[:, 1] / DEG, lw=1, label="Kalman estimate")

axes[0].axhline(STEP / DEG, color="k", ls="--", lw=0.8)
axes[0].set_title("True pitch")
axes[1].set_title("Gimbal command")
axes[2].set_title("What the filter sees")
for ax in axes:
    ax.set_xlabel("time (s)")
    ax.set_ylabel("deg")
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)
save(fig, "09_kalman.png")
