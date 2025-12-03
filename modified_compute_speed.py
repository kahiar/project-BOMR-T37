## In utils.py add RADS2THYMIO = WHEEL_RADIUS_MM / THYMIO2MMS

def compute_speed(self, actual_pos, target_pos, max_speed=500,
                k_rho=0.35, k_alpha=0.7):
    """
    Compute wheel speeds for differential drive to reach target.

    Args:
        actual_pos: np.array [x, y, theta] in pixels and radians
        target_pos: np.array [x, y] in pixels
        max_speed: int, maximum motor command (0-500 for Thymio)
        k_rho: float, linear gain in Hz (1/second)
        k_alpha: float, angular gain in Hz (1/second)

    Returns:
        np.array: [left_speed, right_speed] motor commands
    """
    # Compute position error
    target_array = np.array(target_pos)
    delta_pos = target_array - actual_pos[0:2]
    
    # Compute angle to target
    alpha = -actual_pos[2] + np.arctan2(delta_pos[1], delta_pos[0])
    alpha = (alpha + np.pi) % (2 * np.pi) - np.pi  # wraps angle to (-pi, pi]
    
    # Distance to target (in pixels)
    rho = np.linalg.norm(delta_pos)

    # Control laws (v in pixels/sec, omega in rad/sec)
    v = k_rho * rho * np.cos(alpha)  # linear velocity [pixels/sec]
    omega = k_alpha * alpha  # angular velocity [rad/sec]

    # Convert to wheel angular velocities [rad/sec]
    l = self.robot_width / 2  # half width in pixels
    r = self.wheel_radius  # wheel radius in pixels
    
    phi1_dot = (v + l * omega) / r  # left wheel [rad/sec]
    phi2_dot = (v - l * omega) / r  # right wheel [rad/sec]

    # Convert from rad/sec to motor commands 
    left_cmd = phi1_dot * RADS2THYMIO
    right_cmd = phi2_dot * RADS2THYMIO

    # Clip to max_speed
    max_cmd = max(abs(left_cmd), abs(right_cmd))
    if max_cmd > max_speed:
        scale = max_speed / max_cmd
        left_cmd *= scale
        right_cmd *= scale

    return np.array([left_cmd, right_cmd])