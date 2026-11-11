"""Stage 8: LQR. Instead of picking three gains, pick how much you care about
pitch error versus gimbal effort and let the math produce the gains.

State x = [integral of error, theta - r, omega, delta]. The integral state
gives the same disturbance rejection the I term gives PID. The controller
does not measure the gimbal angle; it tracks it with a copy of the servo model.
"""

import numpy as np
from common import DEG, STEP, plt, print_metrics, save, tuned_pid
from scipy.linalg import solve_continuous_are

from rocketpid import RocketParams, linearize, pid_controller, simulate, step_metrics

p = RocketParams()
A, B = linearize(p)

# augment with z' = theta - r
Aa = np.zeros((4, 4))
Aa[0, 1] = 1.0
Aa[1:, 1:] = A
Ba = np.vstack([[0.0], B])

Q = np.diag([400.0, 60.0, 0.5, 0.0])  # cost on [integral error, angle error, rate, gimbal]
R = np.array([[10.0]])                # cost on gimbal command
P = solve_continuous_are(Aa, Ba, Q, R)
K = (np.linalg.solve(R, Ba.T @ P)).ravel()
print("LQR gains [integral, theta, omega, delta]:", np.round(K, 3))
print("closed-loop poles:", np.round(np.linalg.eigvals(Aa - Ba @ K[None, :]), 2))


def make_lqr():
    state = dict(z=0.0, delta_hat=0.0)

    def controller(t, r, theta_meas, omega_meas, dt):
        e = theta_meas - r
        u = -(K[0] * state["z"] + K[1] * e + K[2] * omega_meas + K[3] * state["delta_hat"])
        u_sat = max(-p.delta_max, min(p.delta_max, u))
        if u == u_sat:                      # anti-windup: only integrate when not clamped
            state["z"] += e * dt
        state["delta_hat"] += dt * (u_sat - state["delta_hat"]) / (p.servo_tau + dt)
        return u_sat

    return controller


def gust(t):
    return 0.08 if t >= 2.0 else 0.0


noise = dict(angle_noise=0.05 * DEG, gyro_noise=0.5 * DEG)
fig, axes = plt.subplots(1, 2, figsize=(11, 4.2))
runs = (
    ("tuned PID", pid_controller(tuned_pid(p.delta_max))),
    ("LQR", make_lqr()),
)
for label, ctrl in runs:
    res = simulate(ctrl, p, setpoint=STEP, disturbance=gust, t_end=4.0, **noise)
    first = res.t < 2.0
    print_metrics(label, step_metrics(res.t[first], res.theta[first], STEP, band=0.05))
    dip = np.max(np.abs(res.theta[~first] - STEP)) / DEG
    print(f"{label:<28} largest error after disturbance: {dip:.2f} deg")
    axes[0].plot(res.t, res.theta / DEG, label=label)
    axes[1].plot(res.t, res.delta / DEG, label=label)

axes[0].axhline(STEP / DEG, color="k", ls="--", lw=0.8)
axes[0].set_title("Pitch: step at 0 s, disturbance at 2 s")
axes[1].set_title("Gimbal angle")
for ax in axes:
    ax.set_xlabel("time (s)")
    ax.set_ylabel("deg")
    ax.legend()
    ax.grid(True, alpha=0.3)
save(fig, "08_lqr.png")
