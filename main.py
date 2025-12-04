import asyncio
import time
import numpy as np

from tdmclient import aw

from utils import WHEEL_RADIUS_MM, THYMIO_WIDTH_MM, THYMIO2MMS
from vision_system import VisionSystem
from kalman_filter import KalmanFilter
from path_planner import PathPlanner
from motion_controller import MotionController, ThymioConnection
from visualizer import Visualizer


async def main():
    """Main navigation loop: calibrate, plan path, and navigate to goal."""

    # Initialize components
    vision = VisionSystem(camera_id=0)
    planner = PathPlanner()
    visualizer = Visualizer(window_name="Thymio Navigation")

    # Calibration and static element detection
    vision.calibrate(corner_ids={0, 2, 3, 5}, goal_id=1)

    frame = vision.get_transform_frame()
    obstacles = vision.detect_obstacles(frame)
    robot_pose = vision.detect_robot_raw_pose(frame)

    # Compute initial path
    path = planner.compute_path(robot_pose[0:2], vision.goal_position, obstacles)
    if path is None:
        print("ERROR: No path found!")
        return

    # Initialize Kalman filter with first detection
    kalman = KalmanFilter(robot_pose, vision.mm2px)

    # Connect to Thymio and run navigation
    with ThymioConnection() as (client, node):
        motion = MotionController(mm2px=vision.mm2px)
        motion.upload_local_avoidance(node)

        last_time = time.time()
        waypoint_idx = 0
        WAYPOINT_THRESHOLD = 30

        while waypoint_idx < len(path):
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

            # Get sensor data for visualization
            sensor_data = np.array(list(node['prox.horizontal']))

            # Compute dt
            current_time = time.time()
            dt = current_time - last_time
            last_time = current_time

            last_state = kalman.state[0:2].copy()

            # Kalman filter
            kalman.predict(current_speed_px, dt)
            kalman.update(robot_pose)

            # Check waypoint reached
            current_waypoint = path[waypoint_idx]
            distance_to_waypoint = np.linalg.norm(kalman.state[0:2] - current_waypoint)

            if distance_to_waypoint < WAYPOINT_THRESHOLD:
                waypoint_idx += 1
                print(f"Waypoint {waypoint_idx}/{len(path)} reached")

                if waypoint_idx >= len(path):
                    motion.set_speed(np.array([0, 0]), node)
                    print("Goal reached!")
                    break

            # Compute and send motor commands
            current_waypoint = path[waypoint_idx]

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

            # Update visualization
            info = {
                "Waypoint": f"{waypoint_idx + 1}/{len(path)}",
                "Distance": f"{distance_to_waypoint:.0f} px"
            }

            visualizer.update(
                frame=frame,
                obstacles=obstacles,
                robot_pos=robot_pose,
                path=path,
                current_waypoint_idx=waypoint_idx,
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