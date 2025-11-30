import numpy as np
import pip

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
            [40,  20, -20, -20, -40,  30, -10],
            [-40, -20, -20,  20,  40, -10,  30]
        ])

    def compute_speed(self, actual_pos, target_pos, r, l, max_speed=500,
                    k_rho=30, k_alpha=60):
        """
        Compute wheel speeds for differential drive to reach target.

        Args:
            actual_pos: np.array [x, y, theta]
            target_pos: np.array [x, y]
            r: right speed
            l: left speed
            max_speed: int
            k_rho: float
            k_alpha: float

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

    def upload_local_avoidance(self, node, base_speed=80, threshold=1000):
        """
        Upload ANN obstacle avoidance to run directly on Thymio at 10Hz.

        Args:
            node: Thymio node
            base_speed
            threshold: value to trigger avoidance (0-4500)
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
            print(f"[Thymio] Compilation error: {error}")
            return False

        aw(node.run())
        print(f"[Thymio] ANN uploaded (threshold={threshold}, base_speed={base_speed})")
        return True

    def stop_program(self, node):
        """Stop the running program and motors."""
        aw(node.stop())
        aw(node.set_variables({
            "motor.left.target": [0],
            "motor.right.target": [0],
        }))
        print("[Thymio] Program stopped")

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


class ThymioConnection:
    """
    Helper class to manage Thymio connection and avoid lock errors.

    Usage:
        with ThymioConnection() as (client, node):
            # Use node here
            pass
        # Automatically unlocks when done or on error
    """

    def __init__(self, timeout=10):
        self.timeout = timeout
        self.client = None
        self.node = None

    def __enter__(self):
        from tdmclient import ClientAsync, aw

        self.client = ClientAsync()

        try:
            self.node = aw(self.client.wait_for_node())
            aw(self.node.lock())
            print(f"[Thymio] Connected and locked: {self.node}")
        except Exception as e:
            print(f"[Thymio] Connection failed: {e}")
            raise

        return self.client, self.node

    def __exit__(self, exc_type, exc_val, exc_tb):
        # Always try to stop motors and unlock
        if self.node is not None:
            try:
                # Stop motors
                aw(self.node.set_variables({
                    "motor.left.target": [0],
                    "motor.right.target": [0],
                }))
                print("[Thymio] Motors stopped")
            except Exception:
                pass

            try:
                aw(self.node.stop())
                aw(self.node.unlock())
                print("[Thymio] Unlocked")
            except Exception:
                pass

        if self.client is not None:
            try:
                self.client.close()
                print("[Thymio] Disconnected")
            except Exception:
                pass

        return False  # Don't suppress exceptions


def force_unlock_thymio():
    """
    Force unlock the Thymio if it's stuck in a locked state.
    """
    from tdmclient import ClientAsync

    print("[Thymio] Attempting force unlock...")

    try:
        client = ClientAsync()
        node = aw(client.wait_for_node(timeout=5))

        try:
            aw(node.set_variables({
                "motor.left.target": [0],
                "motor.right.target": [0],
            }))
        except:
            pass

        # Unlock
        aw(node.stop())
        aw(node.unlock())
        print("[Thymio] Force unlock successful!")

        client.close()
        return True

    except Exception as e:
        print(f"[Thymio] Force unlock failed: {e}")
        print("[Thymio] Try turning the robot off and on again")
        return False


# TEST AREA

if __name__ == "__main__":
    import time
    import sys

    # Create dummy utils module if not available
    class DummyUtils:
        WHEEL_RADIUS_MM = 21
        THYMIO_WIDTH_MM = 95

    sys.modules['utils'] = DummyUtils()

    def run_tests():
        print("=" * 50)
        print("MOTION CONTROLLER TEST")
        print("=" * 50)
        print("\nOptions:")
        print("  1 - Run full test")
        print("  2 - Force unlock (use if you get lock errors)")
        print("  q - Quit")

        choice = input("\nEnter choice: ").strip().lower()

        if choice == '2':
            force_unlock_thymio()
            return
        elif choice == 'q':
            return
        elif choice != '1':
            print("Invalid choice")
            return

        # Use the context manager for safe connection handling
        with ThymioConnection() as (client, node):

            controller = MotionController(mm2px=1.0)

            # ---------------------------------------------------------
            # TEST 1: set_speed function
            # ---------------------------------------------------------
            print("\n[TEST 1] set_speed function")
            print("    Robot will move FORWARD for 2 seconds...")

            input("    Press ENTER to start...")

            # Move forward
            controller.set_speed(np.array([100, 100]), node)
            time.sleep(2)

            # Stop
            controller.set_speed(np.array([0, 0]), node)
            print("    ✓ Turn test complete")

            # ---------------------------------------------------------
            # TEST 2: apply_local_avoidance function
            # ---------------------------------------------------------
            print("\n[TEST 2] apply_local_avoidance function")
            print("    Robot will move forward and avoid obstacles")
            print("    Place your hand in front of sensors to test")
            print("    Test runs for 15 seconds (Ctrl+C to stop early)")

            input("    Press ENTER to start...")

            base_speed = np.array([100, 100])
            start_time = time.time()

            try:
                # Upload and run - avoidance now runs autonomously at 10Hz on Thymio
                controller.upload_local_avoidance(node, base_speed=100, threshold=1500)

                # Let it run for 15 seconds
                print("Running autonomous avoidance for 30 seconds...")
                time.sleep(30)

                # Stop
                controller.stop_program(node)
            except KeyboardInterrupt:
                print("\n    Stopped by user")

            # Stop robot
            controller.set_speed(np.array([0, 0]), node)
            print("\n    ✓ Local avoidance test complete")

            # ---------------------------------------------------------
            # TEST 3: Sensor reading (for tuning)
            # ---------------------------------------------------------
            print("\n[TEST 3] Sensor reading (for weight tuning)")
            print("    Move your hand in front of sensors to see values")
            print("    Test runs for 10 seconds (Ctrl+C to stop early)")

            input("    Press ENTER to start...")

            start_time = time.time()
            try:
                while time.time() - start_time < 10:
                    sensors = controller.get_sensor_data(node)
                    print(f"\r    [FL:{sensors[0]:4d}] [L:{sensors[1]:4d}] [C:{sensors[2]:4d}] [R:{sensors[3]:4d}] [FR:{sensors[4]:4d}]", end="")
                    time.sleep(0.1)
            except KeyboardInterrupt:
                print("\n    Stopped by user")

            print("\n    ✓ Sensor test complete")

        # Connection is automatically cleaned up here
        print("\n" + "=" * 50)
        print("ALL TESTS COMPLETE")
        print("=" * 50)

    # Run tests
    try:
        run_tests()
    except KeyboardInterrupt:
        print("\n\nTest interrupted by user")
        # Try to force unlock on interrupt
        force_unlock_thymio()
    except Exception as e:
        print(f"\n\nError: {e}")
        # Try to force unlock on error
        force_unlock_thymio()

