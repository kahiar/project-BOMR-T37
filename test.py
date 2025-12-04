import asyncio
import time
import numpy as np

from tdmclient import aw

from utils import THYMIO2MMS
from vision_system import VisionSystem
from kalman_filter import KalmanFilter
from path_planner import PathPlanner
from motion_controller import MotionController, ThymioConnection
from visualizer import Visualizer


async def main():
    """
    Main navigation loop.

    Executes the full navigation pipeline:
    1. Initialize vision system and calibrate camera
    2. Detect obstacles and compute initial path
    3. Navigate to goal using Kalman filter for state estimation
    4. Handle local obstacle avoidance via Thymio's proximity sensors
    5. Continue running - pick up robot to restart navigation
    """

    # Initialize components
    vision = VisionSystem(camera_id=0)
    planner = PathPlanner()
    visualizer = Visualizer(window_name="Thymio Navigation")

    # Calibration and static element detection
    vision.calibrate(corner_ids={0, 2, 3, 5}, goal_id=1)

    frame = vision.get_transform_frame()
    obstacles = vision.detect_obstacles(frame)

    # Wait for initial robot detection
    robot_pose = None
    while robot_pose is None:
        frame = vision.get_transform_frame()
        if frame is not None:
            robot_pose = vision.detect_robot_raw_pose(frame)

    # Initialize Kalman filter with first detection
    kalman = KalmanFilter(robot_pose, vision.mm2px)

    # Connect to Thymio and run navigation
    with ThymioConnection() as (client, node):
        motion = MotionController(mm2px=vision.mm2px)
        motion.upload_local_avoidance(node)

        last_time = time.time()
        WAYPOINT_THRESHOLD = 30
        GOAL_THRESHOLD = 40
        RESTART_THRESHOLD = 100  # Distance from goal to restart navigation

        path = None
        waypoint_idx = 0
        at_goal = False

        while True:
            frame = vision.get_transform_frame()
            if frame is None:
                continue

            # Get vision measurement (may be None if marker occluded)
            robot_pose = vision.detect_robot_raw_pose(frame)

            # Read wheel speeds from Thymio
            aw(node.wait_for_variables())
            current_speed = np.array([
                node["motor.left.speed"],
                node["motor.right.speed"]
            ])
            current_speed_px = current_speed * THYMIO2MMS * vision.mm2px

            # Compute time step
            current_time = time.time()
            dt = current_time - last_time
            last_time = current_time

            # Store previous state for replanning check
            last_state = kalman.state[0:2].copy()

            # Kalman filter: predict then update
            kalman.predict(current_speed_px, dt)
            if robot_pose is None:
                print(f"Predict: {kalman.state[0:2]}")
            kalman.update(robot_pose)
            if robot_pose is None:
                print(f"Camera: {kalman.state[0:2]}")


            # Distance to goal
            distance_to_goal = np.linalg.norm(kalman.state[0:2] - vision.goal_position)

            # Check if robot was moved away from goal (to restart)
            if at_goal and distance_to_goal > RESTART_THRESHOLD:
                print("Robot moved - restarting navigation")
                at_goal = False
                path = None

            # Compute path if needed
            if path is None and not at_goal:
                path = planner.compute_path(kalman.state[0:2], vision.goal_position, obstacles)
                waypoint_idx = 0
                if path is None:
                    print("No path found - waiting...")

            # Navigation logic
            if path is not None and not at_goal:
                current_waypoint = path[waypoint_idx]
                distance_to_waypoint = np.linalg.norm(kalman.state[0:2] - current_waypoint)

                # Check waypoint reached
                if distance_to_waypoint < WAYPOINT_THRESHOLD:
                    waypoint_idx += 1
                    print(f"Waypoint {waypoint_idx}/{len(path)} reached")

                    if waypoint_idx >= len(path):
                        # Goal reached
                        motion.set_speed(np.array([0, 0]), node)
                        print("Goal reached! Pick up robot to restart.")
                        at_goal = True
                        path = None
                        continue

                # Update current waypoint after potential increment
                current_waypoint = path[waypoint_idx]

                # Compute and send motor commands
                if distance_to_waypoint > 350:
                    # Change gains to turn less
                    target_speed = motion.compute_speed(
                        kalman.state,
                        current_waypoint,
                        k_rho=0.25,
                        k_alpha=0.35,
                    )
                else:
                    target_speed = motion.compute_speed(
                        kalman.state,
                        current_waypoint,
                    )
                motion.set_speed(target_speed, node)

                # Kidnapping check
                if np.linalg.norm(kalman.state[0:2] - last_state) > 100:
                    path = planner.compute_path(kalman.state[0:2], vision.goal_position, obstacles)
                    waypoint_idx = 0
            else:
                # At goal or no path - stop motors
                motion.set_speed(np.array([0, 0]), node)

            # Get sensor data for visualization
            sensor_data = np.array(list(node['prox.horizontal']))

            # Update visualization
            if path is not None:
                distance_to_waypoint = np.linalg.norm(kalman.state[0:2] - path[waypoint_idx])
                info = {
                    "Waypoint": f"{waypoint_idx + 1}/{len(path)}",
                    "To Waypoint": f"{distance_to_waypoint:.0f} px",
                    "To Goal": f"{distance_to_goal:.0f} px"
                }
            else:
                info = {
                    "Status": "AT GOAL" if at_goal else "NO PATH",
                    "To Goal": f"{distance_to_goal:.0f} px"
                }

            visualizer.update(
                frame=frame,
                obstacles=obstacles,
                robot_pos=robot_pose,
                path=path,
                current_waypoint_idx=waypoint_idx if path else 0,
                sensor_data=sensor_data,
                goal_pos=vision.goal_position,
                info_dict=info,
                kalman_state=kalman.state,
                kalman_covariance=kalman.P
            )

    visualizer.close()
    vision.release()


if __name__ == "__main__":
    asyncio.run(main())