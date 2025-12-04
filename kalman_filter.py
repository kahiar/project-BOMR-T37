import numpy as np
from utils import THYMIO_WIDTH_MM


class KalmanFilter:
    """Extended Kalman Filter for robot pose estimation with differential drive model."""

    def __init__(self, initial_pose, mm2px):
        """
        Initialize the Kalman filter.

        Args:
            initial_pose: np.array [x, y, theta] initial robot pose in pixels/radians
            mm2px: float, conversion factor from millimeters to pixels
        """
        self.state = initial_pose.astype(float)
        self.P = np.eye(3) * 100.0

        self.Q = np.load("Q.npy")  # Process noise covariance
        self.R = np.load("R.npy")  # Measurement noise covariance
        self.H = np.eye(3)  # Measurement matrix (direct observation)

        self.L = THYMIO_WIDTH_MM * mm2px  # Wheel separation in pixels

    def predict(self, control_input, dt):
        """
        Prediction step using differential drive kinematics.

        Args:
            control_input: np.array [v_left, v_right] wheel speeds in px/s
            dt: float, time step in seconds

        Updates:
            self.state: Predicted state [x, y, theta]
            self.P: Predicted covariance matrix
        """
        vL, vR = control_input
        x, y, theta = self.state

        # Differential drive model
        v = (vR + vL) / 2
        omega = (vL - vR) / self.L

        # State prediction
        x_pred = x + v * dt * np.cos(theta)
        y_pred = y + v * dt * np.sin(theta)
        theta_pred = (theta + omega * dt + np.pi) % (2 * np.pi) - np.pi

        self.state = np.array([x_pred, y_pred, theta_pred])

        # Jacobian of motion model
        G = np.array([
            [1, 0, -v * dt * np.sin(theta)],
            [0, 1, v * dt * np.cos(theta)],
            [0, 0, 1]
        ])

        self.P = G @ self.P @ G.T + self.Q

    def update(self, measurement):
        """
        Correction step with vision measurement.

        Args:
            measurement: np.array [x, y, theta] from vision, or None if not detected

        Updates:
            self.state: Corrected state estimate
            self.P: Corrected covariance matrix
        """
        if measurement is None:
            # Increase uncertainty when no measurement available
            self.P += self.Q * 10
            return

        z = measurement.astype(float)

        # Innovation (measurement residual)
        innovation = z - self.state
        innovation[2] = (innovation[2] + np.pi) % (2 * np.pi) - np.pi

        # Kalman gain
        S = self.H @ self.P @ self.H.T + self.R
        K = self.P @ self.H.T @ np.linalg.inv(S)

        # State and covariance update
        self.state = self.state + K @ innovation
        self.state[2] = (self.state[2] + np.pi) % (2 * np.pi) - np.pi
        self.P = (np.eye(3) - K @ self.H) @ self.P

    def get_state(self):
        """
        Get current filtered pose estimate.

        Returns:
            np.array: [x, y, theta] filtered pose
        """
        return self.state