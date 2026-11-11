"""Linear Kalman filter for the pitch axis.

Estimates [theta, omega, d] from a noisy angle measurement, using the known
gimbal angle as the model input. d is an unknown, slowly changing angular
acceleration (wind, motor misalignment, model error). Without it the filter
trusts the model too much and its angle estimate is biased whenever an
outside torque acts on the rocket.
"""

import numpy as np
from scipy.linalg import expm


class KalmanFilter:
    def __init__(self, params, dt, angle_noise_std, accel_noise_std=1.0, d_walk_std=10.0):
        p = params
        A = np.array(
            [
                [0.0, 1.0, 0.0],
                [p.c_aero / p.inertia, -p.c_damp / p.inertia, 1.0],
                [0.0, 0.0, 0.0],
            ]
        )
        B = np.array([[0.0], [p.thrust * p.arm / p.inertia], [0.0]])

        # exact discretization of x' = A x + B u over one controller period
        M = np.zeros((4, 4))
        M[:3, :3] = A
        M[:3, 3:] = B
        Md = expm(M * dt)
        self.F = Md[:3, :3]
        self.G = Md[:3, 3:]

        self.H = np.array([[1.0, 0.0, 0.0]])
        self.R = angle_noise_std ** 2
        # process noise: white angular acceleration, plus a random walk on d
        g = np.array([[dt ** 2 / 2], [dt], [0.0]])
        self.Q = g @ g.T * accel_noise_std ** 2
        self.Q[2, 2] = d_walk_std ** 2 * dt

        self.x = np.zeros((3, 1))
        self.P = np.diag([0.01, 0.1, 1.0])

    def step(self, delta, theta_meas):
        """Predict with the last gimbal angle, then correct with the new measurement.

        Returns the estimates (theta, omega, d).
        """
        self.x = self.F @ self.x + self.G * delta
        self.P = self.F @ self.P @ self.F.T + self.Q

        innovation = theta_meas - self.x[0, 0]
        S = (self.H @ self.P @ self.H.T)[0, 0] + self.R
        K = self.P @ self.H.T / S
        self.x = self.x + K * innovation
        self.P = (np.eye(3) - K @ self.H) @ self.P
        return self.x[0, 0], self.x[1, 0], self.x[2, 0]
