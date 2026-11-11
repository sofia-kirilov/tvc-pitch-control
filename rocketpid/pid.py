"""Discrete PID controller with the usual fixes for real hardware."""


class PID:
    """u = kp*e + ki*integral(e) + kd*de/dt

    out_limit         clamp the output to +/- this value (None = no clamp)
    anti_windup       stop integrating while the output is clamped and the
                      error would push it further into the clamp
    d_on_measurement  differentiate -measurement instead of error, so a
                      setpoint step does not cause a derivative spike
    d_filter_tau      time constant (s) of a low-pass filter on the
                      derivative term (0 = no filter)
    setpoint_weight   fraction of the setpoint the P term sees (0 to 1).
                      Below 1, a setpoint step hits the P term less hard and
                      the I term does the rest, which cuts overshoot without
                      changing how the loop fights disturbances
    """

    def __init__(
        self,
        kp,
        ki=0.0,
        kd=0.0,
        out_limit=None,
        anti_windup=True,
        d_on_measurement=True,
        d_filter_tau=0.0,
        setpoint_weight=1.0,
    ):
        self.kp = kp
        self.ki = ki
        self.kd = kd
        self.out_limit = out_limit
        self.anti_windup = anti_windup
        self.d_on_measurement = d_on_measurement
        self.d_filter_tau = d_filter_tau
        self.setpoint_weight = setpoint_weight
        self.reset()

    def reset(self):
        self.integral = 0.0   # already multiplied by ki
        self._prev = None
        self._d = 0.0

    def update(self, setpoint, measurement, dt):
        error = setpoint - measurement

        # derivative
        d_input = -measurement if self.d_on_measurement else error
        raw_d = 0.0 if self._prev is None else (d_input - self._prev) / dt
        self._prev = d_input
        blend = dt / (self.d_filter_tau + dt)
        self._d += blend * (raw_d - self._d)

        # integral (tentative)
        new_integral = self.integral + self.ki * error * dt

        p_term = self.kp * (self.setpoint_weight * setpoint - measurement)
        u = p_term + new_integral + self.kd * self._d

        if self.out_limit is None:
            self.integral = new_integral
            return u

        u_sat = max(-self.out_limit, min(self.out_limit, u))
        pushing_further = (u > u_sat and error > 0) or (u < u_sat and error < 0)
        if not (self.anti_windup and pushing_further):
            self.integral = new_integral
        return u_sat
