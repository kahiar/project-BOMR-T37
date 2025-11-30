import numpy as np
import utils
from utils import THYMIO_WIDTH_MM


class KalmanFilter:
    """
    Bayesian filtering for robot pose estimation.
    """

    def __init__(self, initial_pose):
        """
        Args:
            initial_pose: np.array [x, y, theta]
            process_noise: Process noise covariance (motion uncertainty)
            measurement_noise: Measurement noise covariance (vision uncertainty)
        """
        self.state = initial_pose.astype(float)  # [x, y, theta]
        self.P = np.eye(3) * 1.0  # Covariance matrix

        # Process noise (model)
        self.Q = np.load("Q.npy") # to have the variance directly

        # Measurement noise (vision)
        self.R = np.load("R.npy") # to have the variance directly

        # Measurement Jacobian (constant identity since h(x) = x)
        self.H = np.eye(3)

        self.L = THYMIO_WIDTH_MM  # Thymio wheel separation


    def predict(self, control_input, dt):
        """
        Predict next state.

        Args:
            control_input: np.array [v_left, v_right] wheel speeds
            dt: float, time step in seconds

        Updates:
            self.state, self.P (predicted state and covariance)
        """
        vL, vR = control_input
        x, y, theta = self.state

        # Differential-drive kinematics
        v = (vR + vL) / 2
        omega = (vR - vL) / self.L

        # ----- MOTION MODEL g(x,u) -----
        x_pred = x + v * dt * np.cos(theta)
        y_pred = y + v * dt * np.sin(theta)
        theta_pred = theta + omega * dt

        # Normalize angle
        theta_pred = (theta_pred + np.pi) % (2*np.pi) - np.pi
        
        self.state = np.array([x_pred, y_pred, theta_pred])

        # ----- JACOBIAN G -----
        G = np.array([
            [1, 0, -v * dt * np.sin(theta)],
            [0, 1,  v * dt * np.cos(theta)],
            [0, 0,  1]
        ])

        # ----- COVARIANCE UPDATE -----
        self.P = G @ self.P @ G.T + self.Q

    def update(self, measurement):
        """
        Correct prediction with vision measurement.

        Args:
            measurement: np.array [x, y, theta] from vision, or None if no detection

        Updates:
            self.state, self.P (corrected state and covariance)
        """
        if measurement is None:
            # TODO: Implement prediction with only robot speed
            return  # No measurement, keep predicted state

        z = measurement.astype(float)

        # Innovation i = z - h(x)
        # and since h(x) = x, H = identity:
        i = z - self.state

        # Normalize the angle component
        i[2] = (i[2] + np.pi) % (2*np.pi) - np.pi

        # Innovation covariance
        S = self.H @ self.P @ self.H.T + self.R

        # Kalman gain
        K = self.P @ self.H.T @ np.linalg.inv(S)

        # State update
        self.state = self.state + K @ i

        # Normalize angle again
        self.state[2] = (self.state[2] + np.pi) % (2*np.pi) - np.pi

        # Covariance update
        I = np.eye(3)
        self.P = (I - K @ self.H) @ self.P

    def get_state(self):
        """
        Current filtered pose estimate.

        Returns:
            np.array: [x, y, theta] filtered pose
        """
        return self.state
