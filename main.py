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

    # Initialize vision
    vision = VisionSystem(camera_id=0)

    # Initialize Kalman filter (will be set with first measurement)
    initial_pose = np.array([0.0, 0.0, 0.0])
    kalman = KalmanFilter(
        initial_pose,
        process_noise=0.1,
        measurement_noise=1.0
    )

    # Initialize path planner
    planner = PathPlanner()

    # Initialize motion controller
    motion = MotionController(mm2px=1.0)  # Updated after vision.calibrate()

    # Initialize visualizer
    visualizer = Visualizer(window_name="Thymio Navigation")

    # Main controller
    controller = RobotController(vision, kalman, planner, motion, visualizer)

    try:
        # Connect to robot and navigate
        client = ClientAsync()
        await controller.connect(client)
        print("System initialized")
            # Update motion controller with calibrated mm2px
            #motion.mm2px = vision.mm2px
            #motion.wheel_radius = utils.WHEEL_RADIUS_MM * vision.mm2px
            #motion.robot_width = utils.THYMIO_WIDTH_MM * vision.mm2px

            # Run navigation
            #await controller.navigate_to_goal(client)

    finally:
        # Cleanup
        vision.release()
        visualizer.close()
        print("System shutdown complete")


if __name__ == "__main__":
    asyncio.run(main())