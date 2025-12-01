#Test
import cv2

import numpy as np
from path_planner import PathPlanner
from vision_system import VisionSystem
from visualizer import Visualizer

vision = VisionSystem()
visualizer = Visualizer(window_name="Robot Debug View")
planner = PathPlanner()

try:
    # === PHASE 1: CALIBRATION ===
    vision.calibrate(
        corner_ids={0, 2, 3, 5},
        goal_id=1
    )

    # === PHASE 2: NAVIGATION/DEBUG VIEW ===
    print("\n=== NAVIGATION DEBUG VIEW ===")
    print("Showing: obstacles, robot, goal, sensors")
    print("Press 'q' to quit\n")

    frame_count = 0

    while True:
        # Get transformed frame
        frame = vision.get_transform_frame()
        if frame is None:
            continue

        # Detect obstacles
        obstacles = vision.detect_obstacles(frame)

        # Detect robot
        robot_pose = vision.detect_robot_raw_pose(frame)
        print(f'Robot: {robot_pose}')

        # Create dummy sensor data for testing
        # TODO: Replace with real sensor data from Thymio
        sensor_data = np.array([1000, 500, 200, 300, 800])

        # Info panel
        info = {
            "Frame": frame_count,
            "Obstacles": len(obstacles),
            "Robot": "DETECTED" if robot_pose is not None else "NOT FOUND"
        }

        if robot_pose is not None:
            info["X"] = f"{int(robot_pose[0])}"
            info["Y"] = f"{int(robot_pose[1])}"
            info["Theta"] = f"{np.degrees(robot_pose[2]):.1f}°"

        # Path planning
        start = np.array([robot_pose[0], robot_pose[1]])
        path = planner.compute_path(start, vision.goal_position, obstacles)


        # Update visualization with everything
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

        frame_count += 1

        if cv2.waitKey(30) & 0xFF == ord('q'):
            break

finally:
    vision.release()
    visualizer.close()
    print("System shutdown complete")