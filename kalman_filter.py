import numpy as np


class KalmanFilter:
    """
    Bayesian filtering for robot pose estimation.
    """

    def __init__(self, initial_pose, process_noise=0.1, measurement_noise=1.0):
        """
        Args:
            initial_pose: np.array [x, y, theta]
            process_noise: Process noise covariance (motion uncertainty)
            measurement_noise: Measurement noise covariance (vision uncertainty)
        """
        self.state = initial_pose  # [x, y, theta]
        self.P = np.eye(3) * 10  # Covariance matrix

        # Process noise (model)
        self.Q = np.eye(3) * process_noise

        # Measurement noise (vision)
        self.R = np.eye(3) * measurement_noise

    def predict(self, control_input, dt):
        """
        Predict next state.

        Args:
            control_input: np.array [v_left, v_right] wheel speeds
            dt: float, time step in seconds

        Updates:
            self.state, self.P (predicted state and covariance)
        """
        # TODO: Implement motion model
        # TODO: Predict next state using differential drive kinematics
        # TODO: Update covariance: P = F*P*F^T + Q
        pass

    def update(self, measurement):
        """
        Correct prediction with vision measurement.

        Args:
            measurement: np.array [x, y, theta] from vision, or None if no detection

        Updates:
            self.state, self.P (corrected state and covariance)
        """
        if measurement is None:
            return  # No measurement, keep predicted state

        # TODO: Kalman gain: K = P*H^T*(H*P*H^T + R)^-1
        # TODO: Innovation: y = measurement - H*state
        # TODO: Update state: state = state + K*y
        # TODO: Update covariance: P = (I - K*H)*P
        pass

    def get_state(self):
        """
        Current filtered pose estimate.

        Returns:
            np.array: [x, y, theta] filtered pose
        """
        return self.state.copy()
