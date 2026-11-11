"""Stage 3: PD. The D term adds damping. Gains come from a target natural
frequency and damping ratio instead of guessing.

Ignoring the servo, the closed loop is
    theta'' + (b*kd + c_damp/I) theta' + (b*kp - a) theta = b*kp*r
with a = c_aero/I and b = T*L/I. Match it to s^2 + 2*zeta*wn*s + wn^2.
"""

from common import DEG, STEP, plt, print_metrics, save

from rocketpid import PID, RocketParams, pid_controller, simulate, step_metrics

p = RocketParams()
a = p.c_aero / p.inertia
b = p.thrust * p.arm / p.inertia

wn, zeta = 8.0, 0.7
kp = (wn ** 2 + a) / b
kd = (2 * zeta * wn - p.c_damp / p.inertia) / b
print(f"design: wn = {wn} rad/s, zeta = {zeta}  ->  kp = {kp:.3f}, kd = {kd:.3f}")

fig, ax = plt.subplots(figsize=(7, 4))
for label, gains in (("P only", (kp, 0.0)), ("PD", (kp, kd))):
    pid = PID(gains[0], kd=gains[1], d_filter_tau=0.01)
    res = simulate(pid_controller(pid), p, setpoint=STEP, t_end=2.0)
    if gains[1] > 0:
        print_metrics(label, step_metrics(res.t, res.theta, STEP))
    ax.plot(res.t, res.theta / DEG, label=label)

ax.axhline(STEP / DEG, color="k", ls="--", lw=0.8, label="setpoint")
ax.set_xlabel("time (s)")
ax.set_ylabel("pitch (deg)")
ax.set_ylim(-30, 40)
ax.set_title("PD control: 5 deg step")
ax.legend()
ax.grid(True, alpha=0.3)
save(fig, "03_pd_control.png")
print("PD is damped but settles about 0.8 deg off target. Holding a tilt against the")
print("aero torque takes a constant gimbal angle, and P only produces one from a constant error.")
