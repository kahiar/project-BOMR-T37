import asyncio
from tdmclient import ClientAsync
from vision_system import VisionSystem
from kalman_filter import KalmanFilter
from path_planner import PathPlanner
from motion_controller import MotionController
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

    # Initialize motion controller ?
    motion = MotionController(mm2px=vision.mm2px) # Updated after vision.calibrate()

    # Initialize visualizer
    visualizer = Visualizer(window_name="Thymio Navigation")

    frame_count = 0

    # Navigation loop

    waipoint_idx = 0

    while waipoint_idx < len(path):
        frame = vision.get_transform_frame()
        if frame is None:
            continue

        robot_pose = vision.detect_robot_raw_pose(frame)

        # get thymio current motor speeds

        # kalman filter to predict

        # ???

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

        # TODO: controller implementation

        # TODO: Filter implementation

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