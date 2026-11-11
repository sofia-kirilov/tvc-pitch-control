import math

import numpy as np

from rocketpid import PID, RocketParams, pid_controller, simulate, step_metrics

DEG = math.pi / 180


def test_proportional_only():
    pid = PID(2.0)
    assert pid.update(1.0, 0.25, 0.01) == 1.5


def test_integral_accumulates():
    pid = PID(0.0, ki=1.0)
    for _ in range(100):
        u = pid.update(1.0, 0.0, 0.01)
    assert abs(u - 1.0) < 1e-9


def test_output_is_clamped():
    pid = PID(10.0, out_limit=0.5)
    assert pid.update(1.0, 0.0, 0.01) == 0.5
    assert pid.update(-1.0, 0.0, 0.01) == -0.5


def test_anti_windup_holds_integral():
    wound = PID(1.0, ki=1.0, out_limit=0.5, anti_windup=False)
    held = PID(1.0, ki=1.0, out_limit=0.5, anti_windup=True)
    for _ in range(500):
        wound.update(1.0, 0.0, 0.01)
        held.update(1.0, 0.0, 0.01)
    assert wound.integral > 4.0
    assert held.integral == 0.0


def test_no_derivative_kick_on_setpoint_step():
    pid = PID(0.0, kd=1.0, d_on_measurement=True)
    pid.update(0.0, 0.0, 0.01)
    assert pid.update(1.0, 0.0, 0.01) == 0.0


def test_open_loop_is_unstable():
    res = simulate(None, t_end=1.5, theta0=1 * DEG)
    assert np.nanmax(np.abs(res.theta)) > 30 * DEG


def test_closed_loop_settles_on_setpoint():
    p = RocketParams()
    pid = PID(0.82, 1.5, 0.124, out_limit=p.delta_max, d_filter_tau=0.01)
    res = simulate(pid_controller(pid), p, setpoint=5 * DEG, t_end=4.0)
    m = step_metrics(res.t, res.theta, 5 * DEG, band=0.05)
    assert m["settle"] < 3.0
    assert abs(m["sse"]) < 0.05 * DEG
