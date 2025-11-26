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
        # Read proximity sensors
        aw(node.wait_for_variables({"prox.horizontal"}))
        prox_horizontal = list(node["prox.horizontal"])

        # Scale factor for sensors
        sensor_scale = 200

        # Build input vector: 5 front sensors + 2 memory values
        x = np.zeros(7)

        # Get and scale the 5 front proximity sensors
        for i in range(5):
            x[i] = prox_horizontal[i] / sensor_scale

        # Memory component (previous outputs scaled down)
        x[5] = self.prox_and_memory[5]
        x[6] = self.prox_and_memory[6]

        # Compute ANN outputs using weight matrix
        delta_left = np.dot(self.w[0], x)
        delta_right = np.dot(self.w[1], x)

        # Update memory for next iteration
        self.prox_and_memory[0:5] = x[0:5]
        self.prox_and_memory[5] = delta_left / 10
        self.prox_and_memory[6] = delta_right / 10

        # Add avoidance correction to desired speed
        corrected_speed = np.array([
            speed_robot[0] + delta_left,
            speed_robot[1] + delta_right
        ])

        return corrected_speed

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
        aw(node.wait_for_variables({"prox.horizontal"}))
        return np.array([x for x in node['prox.horizontal'][0:5]])


# TEST AREA

if __name__ == "__main__":
    import asyncio
    from tdmclient import ClientAsync
    import time

    async def test_motion_controller():
        """Test set_speed and apply_local_avoidance functions"""

        print("=" * 50)
        print("MOTION CONTROLLER TEST")
        print("=" * 50)

        # Connect to Thymio
        print("\n[1] Connecting to Thymio...")

        with ClientAsync() as client:
            async with client.lock() as node:
                print(f"    Connected to: {node}")

                # Create controller (using dummy mm2px since we don't need vision)
                # We won't use wheel_radius/robot_width in this test
                class DummyUtils:
                    WHEEL_RADIUS_MM = 21
                    THYMIO_WIDTH_MM = 95

                # Temporarily replace utils
                import sys
                sys.modules['utils'] = DummyUtils()

                controller = MotionController(mm2px=1.0)

                # -------------------------------------------------------------
                # TEST 1: set_speed function
                # -------------------------------------------------------------
                print("\n[2] TEST: set_speed function")
                print("    Robot will move FORWARD for 2 seconds...")
                print("    Press Ctrl+C to abort at any time")

                input("    Press ENTER to start test 1...")

                # Move forward
                controller.set_speed(np.array([100, 100]), node)
                time.sleep(2)

                # Stop
                controller.set_speed(np.array([0, 0]), node)
                print("    ✓ Forward motion test complete")

                time.sleep(1)

                print("\n    Robot will TURN LEFT for 1 second...")
                input("    Press ENTER to continue...")

                # Turn left (right wheel faster)
                controller.set_speed(np.array([50, 150]), node)
                time.sleep(1)

                # Stop
                controller.set_speed(np.array([0, 0]), node)
                print("    ✓ Turn test complete")

                # -------------------------------------------------------------
                # TEST 2: apply_local_avoidance function
                # -------------------------------------------------------------
                print("\n[3] TEST: apply_local_avoidance function")
                print("    Robot will move forward and avoid obstacles")
                print("    Place your hand in front of the sensors to test")
                print("    Test runs for 15 seconds")

                input("    Press ENTER to start test 2...")

                base_speed = np.array([80, 80])  # Base forward speed
                start_time = time.time()

                while time.time() - start_time < 15:
                    # Get sensor data for display
                    sensors = controller.get_sensor_data(node)

                    # Apply local avoidance
                    corrected_speed = controller.apply_local_avoidance(base_speed.copy(), node)

                    # Clip speeds to valid range
                    corrected_speed = np.clip(corrected_speed, -500, 500)

                    # Set the corrected speed
                    controller.set_speed(corrected_speed, node)

                    # Print debug info
                    print(f"\r    Sensors: {sensors} | Speed: L={int(corrected_speed[0]):4d} R={int(corrected_speed[1]):4d}", end="")

                    time.sleep(0.1)

                # Stop robot
                controller.set_speed(np.array([0, 0]), node)
                print("\n    ✓ Local avoidance test complete")

                # -------------------------------------------------------------
                # TEST 3: Sensor reading test (for tuning)
                # -------------------------------------------------------------
                print("\n[4] TEST: Sensor reading (for weight tuning)")
                print("    Move your hand in front of sensors to see values")
                print("    Test runs for 10 seconds")

                input("    Press ENTER to start test 3...")

                start_time = time.time()
                while time.time() - start_time < 10:
                    sensors = controller.get_sensor_data(node)
                    print(f"\r    [FL:{sensors[0]:4d}] [L:{sensors[1]:4d}] [C:{sensors[2]:4d}] [R:{sensors[3]:4d}] [FR:{sensors[4]:4d}]", end="")
                    time.sleep(0.1)

                print("\n    ✓ Sensor test complete")

                print("\n" + "=" * 50)
                print("ALL TESTS COMPLETE")
                print("=" * 50)

    # Run the test
    asyncio.run(test_motion_controller())

