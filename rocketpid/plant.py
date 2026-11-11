"""Pitch dynamics of a small thrust-vector-controlled (TVC) rocket.

State:  theta  pitch angle from vertical (rad)
        omega  pitch rate (rad/s)
        delta  actual gimbal angle (rad)
Input:  delta_cmd  commanded gimbal angle (rad)

    I * theta'' = T * L * sin(delta) + c_aero * theta - c_damp * omega + tau_dist
    delta'      = (clip(delta_cmd) - delta) / servo_tau     (rate limited)

c_aero > 0 means the center of pressure is ahead of the center of mass, so the
rocket tips over on its own. The controller has to hold it up.
"""

from dataclasses import dataclass
import math

import numpy as np


@dataclass
class RocketParams:
    inertia: float = 0.05                      # kg m^2
    thrust: float = 15.0                       # N
    arm: float = 0.30                          # m, gimbal to center of mass
    c_aero: float = 0.5                        # N m / rad, destabilizing
    c_damp: float = 0.02                       # N m s / rad
    servo_tau: float = 0.03                    # s, servo time constant
    delta_max: float = math.radians(5.0)       # rad, gimbal travel limit
    delta_rate_max: float = math.radians(150)  # rad/s, servo speed limit


def derivatives(state, delta_cmd, tau_dist, p):
    theta, omega, delta = state
    cmd = max(-p.delta_max, min(p.delta_max, delta_cmd))
    ddelta = (cmd - delta) / p.servo_tau
    ddelta = max(-p.delta_rate_max, min(p.delta_rate_max, ddelta))
    alpha = (
        p.thrust * p.arm * math.sin(delta)
        + p.c_aero * theta
        - p.c_damp * omega
        + tau_dist
    ) / p.inertia
    return (omega, alpha, ddelta)


def rk4_step(state, delta_cmd, tau_dist, p, dt):
    """One 4th-order Runge-Kutta step with the inputs held constant."""

    def shifted(s, k, h):
        return (s[0] + h * k[0], s[1] + h * k[1], s[2] + h * k[2])

    k1 = derivatives(state, delta_cmd, tau_dist, p)
    k2 = derivatives(shifted(state, k1, dt / 2), delta_cmd, tau_dist, p)
    k3 = derivatives(shifted(state, k2, dt / 2), delta_cmd, tau_dist, p)
    k4 = derivatives(shifted(state, k3, dt), delta_cmd, tau_dist, p)
    return tuple(
        state[i] + dt / 6 * (k1[i] + 2 * k2[i] + 2 * k3[i] + k4[i]) for i in range(3)
    )


def linearize(p):
    """Small-angle model x' = A x + B u, with x = [theta, omega, delta], u = delta_cmd."""
    A = np.array(
        [
            [0.0, 1.0, 0.0],
            [p.c_aero / p.inertia, -p.c_damp / p.inertia, p.thrust * p.arm / p.inertia],
            [0.0, 0.0, -1.0 / p.servo_tau],
        ]
    )
    B = np.array([[0.0], [0.0], [1.0 / p.servo_tau]])
    return A, B
