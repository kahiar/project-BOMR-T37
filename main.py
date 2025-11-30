import asyncio
import time

from tdmclient import ClientAsync, aw

from utils import WHEEL_RADIUS_MM, THYMIO_WIDTH_MM, THYMIO2MMS
from vision_system import VisionSystem
from kalman_filter import KalmanFilter
from path_planner import PathPlanner
from motion_controller import MotionController, ThymioConnection
from robot_controller import RobotController
from visualizer import Visualizer
import numpy as np


async def main():
    print("Initializing system...")

    vision = VisionSystem(camera_id=0)
    print("pre-planner")
    planner = PathPlanner()

    # Static Elements (corners, obstacles, goal and path)
    vision.calibrate(
        corner_ids={0, 2, 3, 5},
        goal_id=1,
        map_width=800,
        map_height=600,
        real_height=1200
    )
    frame = vision.get_transform_frame()
    obstacles = vision.detect_obstacles(frame)
    robot_pose = vision.detect_robot_raw_pose(frame)
    print("mid")
    path = planner.compute_path(robot_pose[0:2], vision.goal_position, obstacles)
    print("path")
    # Initialize kalman filter
    kalman = KalmanFilter(robot_pose, vision.mm2px)
    print("Kalman started")

    # Initialize visualizer
    visualizer = Visualizer(window_name="Thymio Navigation")
    print("initializer started")

    # Initialize motion controller
    with ThymioConnection() as (client, node):

        motion = MotionController(mm2px=vision.mm2px)
        print("motion started")
        motion.upload_local_avoidance(node)
        frame_count = 0
        last_time = time.time()

        # Navigation loop
        waypoint_idx = 0

        WAYPOINT_THRESHOLD = 50

        while waypoint_idx < len(path):
            frame = vision.get_transform_frame()
            if frame is None:
                continue

            robot_pose = vision.detect_robot_raw_pose(frame)
            print(f"robot_pose: {robot_pose}")
            # get all thymio data we need at once
            aw(node.wait_for_variables())
            current_speed = np.array([
                node["motor.left.speed"],
                node["motor.right.speed"]
            ])
            # Convert to mm/s
            current_speed_mms = current_speed * THYMIO2MMS

            # Convert to px/s
            current_speed_pixs = current_speed_mms * vision.mm2px

            sensor_data = np.array([1000, 500, 200, 300, 800])

            # dt
            current_time = time.time()
            dt = current_time - last_time
            last_time = current_time

            # kalman filter to predict
            kalman.predict(current_speed_pixs[0:2], dt)
            kalman.update(robot_pose)
            #kalman.state = robot_pose
            print(f"predicted pos: {kalman.state[0], kalman.state[1]}")

            current_waypoint = path[waypoint_idx]

            distance_to_waypoint = np.linalg.norm(kalman.state[0:2] - current_waypoint)

            if distance_to_waypoint < WAYPOINT_THRESHOLD:
                waypoint_idx += 1
                print(f"Waypoint {waypoint_idx}/{len(path)} atteint!")

                if waypoint_idx >= len(path):
                    motion.set_speed(np.array([0, 0]), node)
                    print("The END!!!")
                    break

                # Set next waypoint
            current_waypoint = path[waypoint_idx]

            # Compute speed
            target_speed = motion.compute_speed(
                kalman.state,
                current_waypoint,
                WHEEL_RADIUS_MM ,
                THYMIO_WIDTH_MM/2
            )
            print(f"target speed: {target_speed}")

            # Set speed
            motion.set_speed(target_speed, node)

            # TODO: condition to recompute path

            # Visualizer

            info = {
                "Frame": frame_count,
                "Obstacles": len(obstacles),
                "Robot": "DETECTED" if robot_pose is not None else "NOT FOUND"
            }

            if robot_pose is not None:
                info["X"] = f"{int(robot_pose[0])}"
                info["Y"] = f"{int(robot_pose[1])}"
                info["Theta"] = f"{np.degrees(robot_pose[2]):.1f}°"

            visualizer.update(
                frame=frame,
                obstacles=obstacles,
                robot_pos=robot_pose,
                path=path,
                current_waypoint_idx=0,
                sensor_data=sensor_data,
                goal_pos=vision.goal_position,
                info_dict=info
            )

if __name__ == "__main__":
    asyncio.run(main())