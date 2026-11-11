"""Closed-loop simulation: fast physics loop, slower controller loop."""

from dataclasses import dataclass

import numpy as np

from .plant import RocketParams, rk4_step


@dataclass
class Result:
    t: np.ndarray
    theta: np.ndarray       # true pitch angle (rad)
    omega: np.ndarray       # true pitch rate (rad/s)
    delta: np.ndarray       # actual gimbal angle (rad)
    delta_cmd: np.ndarray   # commanded gimbal angle (rad)
    setpoint: np.ndarray    # rad
    theta_meas: np.ndarray  # what the controller saw (rad)


def _as_function(x):
    return x if callable(x) else (lambda t: x)


def pid_controller(pid):
    """Wrap a PID object so it fits the simulate() controller signature."""

    def controller(t, setpoint, theta_meas, omega_meas, dt):
        return pid.update(setpoint, theta_meas, dt)

    return controller


def simulate(
    controller=None,
    params=None,
    t_end=3.0,
    setpoint=0.0,
    disturbance=0.0,
    theta0=0.0,
    omega0=0.0,
    angle_noise=0.0,
    gyro_noise=0.0,
    dt=1e-3,
    ctrl_dt=0.01,
    seed=0,
):
    """Run one simulation.

    controller   function (t, setpoint, theta_meas, omega_meas, dt) -> delta_cmd,
                 or None for open loop (gimbal held at zero)
    setpoint     rad, a number or a function of time
    disturbance  N m of outside torque, a number or a function of time
    angle_noise  std dev of angle sensor noise (rad)
    gyro_noise   std dev of rate sensor noise (rad/s)
    dt           physics step (s)
    ctrl_dt      controller period (s)
    """
    p = params or RocketParams()
    setpoint = _as_function(setpoint)
    disturbance = _as_function(disturbance)
    rng = np.random.default_rng(seed)

    n = int(round(t_end / dt))
    every = max(1, int(round(ctrl_dt / dt)))
    out = np.zeros((7, n))

    state = (theta0, omega0, 0.0)
    cmd = 0.0
    theta_meas = theta0

    for i in range(n):
        t = i * dt
        r = setpoint(t)
        if i % every == 0:
            theta_meas = state[0] + angle_noise * rng.standard_normal()
            omega_meas = state[1] + gyro_noise * rng.standard_normal()
            if controller is not None:
                cmd = controller(t, r, theta_meas, omega_meas, every * dt)
        out[:, i] = (t, state[0], state[1], state[2], cmd, r, theta_meas)
        state = rk4_step(state, cmd, disturbance(t), p, dt)
        if abs(state[0]) > 10.0:  # fell over; stop integrating
            out[:, i + 1:] = np.nan
            out[0, i + 1:] = np.arange(i + 1, n) * dt
            break

    return Result(*out)
