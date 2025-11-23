import cv2
import numpy as np
from cv2 import aruco
from vision_system import VisionSystem
import time

vision = VisionSystem()
vision.calibrate()

for i in range(100):
    robot_pose = vision.detect_robot_raw_pose()
    if robot_pose is not None:
        x, y, theta_deg = robot_pose[0], robot_pose[1], np.degrees(robot_pose[2])
        print(f"Robot at ({x:.1f}, {y:.1f}) facing {theta_deg:.1f}°")
    else:
        print("Robot not detected")
    time.sleep(0.1)



