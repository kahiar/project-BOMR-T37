import asyncio
import time

from tdmclient import ClientAsync, aw
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
    path = planner.compute_path(robot_pose[0:1], vision.goal_position, obstacles)

    # Initialize kalman filter
    kalman = KalmanFilter(robot_pose)

    # Initialize visualizer
    visualizer = Visualizer(window_name="Thymio Navigation")

    # Initialize motion controller
    with ThymioConnection() as (client, node):

        motion = MotionController(mm2px=vision.mm2px)
        motion.upload_local_avoidance(node)
        frame_count = 0
        last_time = time.time()

        while True: # TODO: change this condition
            frame = vision.get_transform_frame()
            if frame is None:
                continue

            robot_pose = vision.detect_robot_raw_pose(frame)

            # get all thymio data we need at once
            aw(node.wait_for_variables())
            current_speed = np.array([
                node["motor.left.speed"],
                node["motor.right.speed"]
            ])
            sensor_data = np.array(node["prox.horizontal"])

            # Compute speed
            target_speed = motion.compute_speed(robot_pose, vision.goal_position,
                                                current_speed[1], current_speed[0])

            # dt
            current_time = time.time()
            dt = current_time - last_time
            last_time = current_time

            # kalman filter to predict
            kalman.predict(target_speed, dt)
            kalman.update(robot_pose)

            # Set speed
            motion.set_speed(kalman.state[0], kalman.state[1])

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

            # TODO: condition to recompute path

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