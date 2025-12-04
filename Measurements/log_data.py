import csv
import time
import numpy as np

from tdmclient import ClientAsync, aw
from motion_controller import ThymioConnection, MotionController
# ===========================================================
# IMPORT VISION SYSTEM
# ===========================================================

# Adapt this import to your project structure
from vision_system import VisionSystem  

vs = VisionSystem()  # Create your vision system object


# ===========================================================
# GET ARUCO MEASUREMENT (using detect_robot_raw_pose)
# ===========================================================

def get_aruco_measurement():
    """
    Uses the student's VisionSystem and detect_robot_raw_pose
    to return (x, y, theta) in MAP coordinates.
    Returns None if the marker is not detected.
    """

    frame = vs.get_transform_frame()  # warpPerspective + frame retrieval
    if frame is None:
        return None

    pose = vs.detect_robot_raw_pose(frame)
    if pose is None:
        return None

    # pose is already: np.array([x, y, theta])
    return pose

# ===========================================================
# MAIN LOGGER
# ===========================================================

def run_logger(duration=20.0, csv_path="log_data_Q.csv", freq=30):
    dt = 1.0 / freq
    print(f"Recording data at {freq} Hz for {duration} seconds...")
    print(f"Saving to: {csv_path}\n")

    with open(csv_path, mode='w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(["timestamp", "x_cam", "y_cam", "theta_cam", "vL", "vR"])

        start = time.time()
        last = start

        vs.calibrate()
        with ThymioConnection() as (client, node):
            motion = MotionController(mm2px = vs.mm2px)
            motion.set_speed(np.array([50, 40]), node)

            while True:
                now = time.time()
                if now - start >= duration:
                    break
                if now - last < dt:
                    continue
                last = now

                # 1) Get vision measurement
                meas = get_aruco_measurement()

                # 2) Get wheel speeds
                aw(node.wait_for_variables())
                vL, vR = np.array([
                    node["motor.left.speed"],
                    node["motor.right.speed"]
                ])

                # 3) Write row
                if meas is None:
                    writer.writerow([now, "", "", "", vL, vR])
                else:
                    x, y, theta = meas
                    writer.writerow([now, x, y, theta, vL, vR])

        print("\n✔ Logging complete.")
        print(f"File saved: {csv_path}")


# ===========================================================
# RUN DIRECTLY
# ===========================================================

if __name__ == "__main__":
    run_logger(duration=20.0, csv_path="log_data_Q.csv", freq=30)
