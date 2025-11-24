import numpy as np
import utils
from tdmclient import aw


class MotionController:
    """
    Motion control and local obstacle avoidance.
    Combines global path following with reactive sensor-based avoidance.
    """

    def __init__(self, mm2px):
        """
        Args:
            mm2px: float, millimeters to pixels conversion factor
        """
        self.wheel_radius = utils.WHEEL_RADIUS_MM * mm2px
        self.robot_width = utils.THYMIO_WIDTH_MM * mm2px

        # Local avoidance parameters
        self.prox_and_memory = np.zeros(7)
        self.k_ann = 1500
        self.offset_ann = 150
        self.w = np.array([
            [80, 20, -25, -20, -80, 12, 0],
            [-80, -20, -20, 20, 80, 0, 12]
        ])

    def compute_speed(self, actual_pos, target_pos, r, l, max_speed=200,
                    k_rho=100, k_alpha=3):
        """
        Compute wheel speeds for differential drive to reach target.

        Args:
            actual_pos: np.array [x, y, theta]
            target_pos: tuple (x, y)
            max_speed: int
            k_rho: float
            k_alpha: float
            r: float
            l: float
            is_near_checkpoint: bool

        Returns:
            np.array: [left_speed, right_speed] motor commands
        """

        target_array = np.array(target_pos)
        delta_pos = target_array - actual_pos[0:2]
        alpha = -actual_pos[2] + np.arctan2(delta_pos[1], delta_pos[0])
        alpha = (alpha + np.pi) % (2 * np.pi) - np.pi #wraps angle to (-pi, pi]
        rho = np.linalg.norm(delta_pos)

        v = k_rho * rho * np.cos(alpha) #linear velocity
        omega = k_alpha * alpha #angular velocity

        phi1_dot = (v + l * omega) / r
        phi2_dot = (v - l * omega) / r

        # If computed angular velocity surpasses maximum angular velocity of Thymio, reduce both wheel speeds equally.
        phi_dot_max = max(abs(phi1_dot), abs(phi2_dot))
        if phi_dot_max > max_speed:
            decrease_ratio = max_speed / phi_dot_max
            phi1_dot *= decrease_ratio
            phi2_dot *= decrease_ratio

        return np.array([phi1_dot, phi2_dot])

    def apply_local_avoidance(self, speed_robot, node):
        """
        Apply reactive obstacle avoidance based on proximity sensors.

        Args:
            speed_robot: np.array [left_speed, right_speed] desired speeds
            node: Thymio node for reading sensors

        Returns:
            np.array: [left_speed, right_speed] corrected speeds
        """
        # TODO: implement local obstacle avoidance

        return speed_robot

    def set_speed(self, speed, node):
        """
        Send speed commands to Thymio motors.

        Args:
            speed: np.array [left_speed, right_speed]
            node: Thymio node
        """
        v = {
            "motor.left.target": [int(speed[0])],
            "motor.right.target": [int(speed[1])],
        }
        aw(node.set_variables(v))

    def get_sensor_data(self, node):
        """
        Get proximity sensor readings for visualization.

        Args:
            node: Thymio node

        Returns:
            np.array: Proximity sensor values [front 5 sensors]
        """
        return np.array([x for x in node['prox.horizontal'][0:5]])


