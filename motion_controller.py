import numpy as np
from tdmclient import ClientAsync, aw
import utils


class MotionController:
    """Motion control with local obstacle avoidance using proximity sensors."""

    def __init__(self, mm2px):
        """
        Args:
            mm2px: Conversion factor from millimeters to pixels
        """
        self.wheel_radius = utils.WHEEL_RADIUS_MM * mm2px
        self.robot_width = utils.THYMIO_WIDTH_MM * mm2px

    def compute_speed(self, actual_pos, target_pos, r, l, max_speed=500,
                      k_rho=20, k_alpha=40):
        """
        Compute wheel speeds to reach target using proportional control.

        Args:
            actual_pos: Current pose [x, y, theta]
            target_pos: Target position [x, y]
            r: Wheel radius
            l: Half of wheel separation
            max_speed: Maximum wheel speed
            k_rho: Distance gain
            k_alpha: Angle gain

        Returns:
            np.array [left_speed, right_speed]
        """
        delta_pos = np.array(target_pos) - actual_pos[0:2]
        alpha = -actual_pos[2] + np.arctan2(delta_pos[1], delta_pos[0])
        alpha = (alpha + np.pi) % (2 * np.pi) - np.pi
        rho = np.linalg.norm(delta_pos)

        v = k_rho * rho * np.cos(alpha)
        omega = k_alpha * alpha

        phi_left = (v + l * omega) / r
        phi_right = (v - l * omega) / r

        # Limit to max speed while preserving ratio
        phi_max = max(abs(phi_left), abs(phi_right))
        if phi_max > max_speed:
            scale = max_speed / phi_max
            phi_left *= scale
            phi_right *= scale

        return np.array([phi_left, phi_right])

    def upload_local_avoidance(self, node, threshold=2000):
        """
        Upload ANN-based obstacle avoidance program to Thymio.
        Runs at sensor update rate (~10Hz) and modifies motor targets
        when obstacles are detected.

        Args:
            node: Thymio node
            threshold: Proximity value to trigger avoidance (0-4500)
        """
        program = f"""
var w_l[7]
var w_r[7]
var sensor_scale
var y[2]
var x[7]
var i
var max_prox

onevent prox
    w_l = [40, 20, -20, -20, -40, 30, -10]
    w_r = [-40, -20, -20, 20, 40, -10, 30]
    sensor_scale = 200
    y = [motor.left.speed, motor.right.speed]
    x = [0, 0, 0, 0, 0, 0, 0]

    max_prox = 0
    i = 0
    while i < 7 do
        if prox.horizontal[i] > max_prox then
            max_prox = prox.horizontal[i]
        end
        i++
    end

    if max_prox > {threshold} then
        i = 0
        while i < 7 do
            x[i] = prox.horizontal[i] / sensor_scale
            y[0] = y[0] + x[i] * w_l[i]
            y[1] = y[1] + x[i] * w_r[i]
            i++
        end
    end

    motor.left.target = y[0]
    motor.right.target = y[1]
"""
        error = aw(node.compile(program))
        if error is not None:
            print(f"Compilation error: {error}")
            return False

        aw(node.run())
        return True

    def set_speed(self, speed, node):
        """Send speed commands to motors."""
        aw(node.set_variables({
            "motor.left.target": [int(speed[0])],
            "motor.right.target": [int(speed[1])],
        }))

    def get_sensor_data(self, node):
        """Get front proximity sensor readings."""
        aw(node.wait_for_variables({"prox.horizontal"}))
        return np.array(list(node['prox.horizontal'][0:5]))


class ThymioConnection:
    """Context manager for safe Thymio connection handling."""

    def __init__(self):
        self.client = None
        self.node = None

    def __enter__(self):
        self.client = ClientAsync()
        self.node = aw(self.client.wait_for_node())
        aw(self.node.lock())
        return self.client, self.node

    def __exit__(self, exc_type, exc_val, exc_tb):
        if self.node is not None:
            try:
                aw(self.node.set_variables({
                    "motor.left.target": [0],
                    "motor.right.target": [0],
                }))
                aw(self.node.stop())
                aw(self.node.unlock())
            except Exception:
                pass

        if self.client is not None:
            try:
                self.client.close()
            except Exception:
                pass

        return False


def force_unlock_thymio():
    """Force unlock Thymio if stuck in locked state."""
    try:
        client = ClientAsync()
        node = aw(client.wait_for_node(timeout=5))
        aw(node.set_variables({
            "motor.left.target": [0],
            "motor.right.target": [0],
        }))
        aw(node.stop())
        aw(node.unlock())
        client.close()
        print("Force unlock successful")
        return True
    except Exception as e:
        print(f"Force unlock failed: {e}")
        return False