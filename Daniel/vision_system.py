import cv2
import numpy as np
from cv2 import aruco


class VisionSystem:
    """Camera detection of robot, obstacles, and goal"""

    def __init__(self, camera_id=0, aruco_dict_type=aruco.DICT_4X4_50):
        self.cap = cv2.VideoCapture(camera_id)
        self.aruco_dict = aruco.getPredefinedDictionary(aruco_dict_type)
        self.aruco_params = aruco.DetectorParameters()

        # Initialize variables
        self.corners = None  # [(x1, y1), ...]
        self.transform_matrix = None
        self.mm2px = None
        self.goal_position = None  # (x, y)

    def calibrate(self):
        """
        Detect map corners and goal using aruco markers.
        Computes perspective transform and scale.

        Sets:
            self.corners, self.transform_matrix, self.mm2px, self.goal_position
        """

        # TODO: Capture frame
        # TODO: Detect corner aruco markers (4 corners)
        # TODO: Detect goal aruco marker
        # TODO: Apply perspective transform
        pass

    def get_frame(self):
        """
        Returns:
            np.array: BGR image or None if capture fails
        """
        ret, frame = self.cap.read()
        return frame if ret else None

    def detect_robot_raw_pose(self):
        """
        This is RAW data - needs filtering.

        Returns:
            np.array: [x, y, theta] in map coordinates, or None if not detected
        """
        frame = self.get_frame()
        if frame is None:
            return None

        # TODO: Apply perspective transform
        # TODO: Detect arrow marker (color-based or aruco)
        # TODO: Extract position (x, y)
        # TODO: Extract orientation angle (theta)

        return np.array([x, y, theta])

    def detect_obstacles(self):
        """
        Returns:
            list: List of obstacle polygons, each as np.array([[x1,y1], [x2,y2], ...])
        """
        frame = self.get_frame()
        if frame is None:
            return []

        # TODO: Apply perspective transform
        # TODO: Detect shapes using contour detection
        # TODO: Filter for squares/rectangles
        # TODO: Return obstacle corner points

        return obstacles

    def get_goal_position(self):
        """
        Returns:
            tuple: (x, y) goal position
        """
        return self.goal_position

    def release(self):
        """Release camera resources"""
        self.cap.release()


#Test
'''
if __name__ == "__main__":
    vision = VisionSystem()
    print("VisionSystem initialized successfully")

    if not vision.cap.isOpened():
        print("Camera failed to open")
        exit(1)

    print("Camera is active")
    print("Press 'q' to quit...")

    while True:
        frame = vision.get_frame()
        if frame is None:
            print("Failed to capture frame")
            break

        cv2.imshow("Vision System - Press 'q' to quit", frame)

        # Arrêter si 'q' est pressé
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    # Libérer les ressources
    cv2.destroyAllWindows()
    vision.release()
    print("Camera released")
'''
