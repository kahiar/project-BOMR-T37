import numpy as np
import time
from tdmclient import ClientAsync, aw


class RobotController:
    """Main orchestrator integrating all components"""

    def __init__(self, vision, kalman_filter, planner, motion, visualizer):
        self.vision = vision
        self.filter = kalman_filter
        self.planner = planner
        self.motion = motion
        self.visualizer = visualizer

        self.node = None
        self.waypoint_threshold = 50  # pixels
        self.goal_threshold = 80  # pixels
        self.control_dt = 0.05  # 20Hz control loop

    async def connect(self, client):
        """
        Connect to Thymio.
        """
        with await client.lock() as node:
            self.node = node
            await node.watch(variables=True)

    def blink_success(self):
        """Signal goal reached with LED pattern"""
        # TODO: Implement LED blinking
        v = {
            "leds.top": [0, 32, 0],  # Green
        }
        aw(self.node.set_variables(v))

    async def navigate_to_goal(self, client):
        """
        Main navigation loop: plan path and follow it with filtering.

        Args:
            client: ClientAsync for sleep
        """
        print("=== Starting Navigation ===")

        # 1. Calibrate vision
        print("Calibrating vision system...")
        self.vision.calibrate()

        # 2. Detect obstacles and goal
        print("Detecting obstacles and goal...")
        obstacles = self.vision.detect_obstacles()
        goal = self.vision.get_goal_position()
        print(f"Found {len(obstacles)} obstacles")
        print(f"Goal at: {goal}")

        # 3. Wait for initial robot detection
        print("Waiting for robot detection...")
        raw_pose = None
        while raw_pose is None:
            raw_pose = self.vision.detect_robot_raw_pose()
            await client.sleep(0.1)

        # Initialize filter with first measurement
        self.filter.state = raw_pose
        print(f"Robot detected at: {raw_pose}")

        # 4. Plan path
        print("Planning path with A*...")
        start = tuple(raw_pose[0:2])
        path = self.planner.compute_path(start, goal, obstacles, margin=50)

        if path is None:
            print("ERROR: No path found!")
            return

        print(f"Path computed with {len(path)} waypoints")

        # 5. Navigation loop
        waypoint_idx = 0

        while waypoint_idx < len(path):
            loop_start = time.time()

            # Get raw measurement from vision
            raw_pose = self.vision.detect_robot_raw_pose()

            # Get current control (speeds being sent)
            current_speed = np.array([
                self.node['motor.left.speed'],
                self.node['motor.right.speed']
            ])

            # Kalman filter: predict and update
            self.filter.predict(current_speed, self.control_dt)
            self.filter.update(raw_pose)
            filtered_pose = self.filter.get_state()

            # Current target waypoint
            target_pos = path[waypoint_idx]
            distance_to_waypoint = np.linalg.norm(
                np.array(target_pos) - filtered_pose[0:2]
            )
            is_near_checkpoint = distance_to_waypoint < self.waypoint_threshold

            # Check if final goal
            is_final_goal = (waypoint_idx == len(path) - 1)
            is_near_goal = is_final_goal and (distance_to_waypoint < self.goal_threshold)

            # Compute motion control
            if is_near_goal:
                # Final precise approach
                speed = self.motion.compute_speed(
                    filtered_pose, target_pos,
                    min_angle=0.1, offset_speed=100,
                    k_forward=50, k_rot=2,
                    is_near_checkpoint=True
                )

                # Check if goal reached
                if distance_to_waypoint < 20:
                    self.motion.set_speed(np.array([0, 0]), self.node)
                    self.blink_success()
                    print("=== GOAL REACHED ===")
                    break
            else:
                # Normal navigation
                speed = self.motion.compute_speed(
                    filtered_pose, target_pos,
                    min_angle=np.pi / 2, offset_speed=200,
                    k_forward=100, k_rot=3,
                    is_near_checkpoint=is_near_checkpoint
                )

            # Apply local avoidance
            speed = self.motion.apply_local_avoidance(speed, self.node)

            # Send to robot
            self.motion.set_speed(speed, self.node)

            # Visualization
            sensor_data = self.motion.get_sensor_data(self.node)
            frame = self.vision.get_frame()
            self.visualizer.update(
                frame, obstacles, filtered_pose,
                path, waypoint_idx, sensor_data
            )

            # Advance waypoint if reached
            if is_near_checkpoint and not is_final_goal:
                waypoint_idx += 1
                print(f"Waypoint {waypoint_idx}/{len(path)} reached")

            # Maintain control rate
            elapsed = time.time() - loop_start
            if elapsed < self.control_dt:
                await client.sleep(self.control_dt - elapsed)