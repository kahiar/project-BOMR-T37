import cv2
import numpy as np
from cv2 import aruco
import matplotlib.pyplot as plt


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
        self.map_size = (800, 600)
        self.goal_position = None  # (x, y)

        # This works as 'cache' to make sure every function gets same frame
        self._current_transform_frame = None

    def calibrate(self, corner_ids={0,2,3,5}, robot_id=4, goal_id=1, map_width=800, map_height=600):
        """
        Detect map corners and goal using aruco markers.
        Computes perspective transform and scale.

        Sets:
            self.corners, self.transform_matrix, self.mm2px, self.goal_position
        """
        self.map_size = (map_width, map_height)

        frame = self.get_frame()
        if frame is None:
            raise Exception("Failed to capture frame")

        # Extract centers of detected markers
        marker_centers = self._detect_marker_centers(frame, target_ids=corner_ids)

        # Detect corners, put dictionary with id as key and position as value
        corner_markers = {id: pos for id, pos in marker_centers.items() if id in corner_ids}
        if len(corner_markers) != 4:
            raise Exception(f"Detected {len(corner_markers)} instead of 4.")

        corner_pts = [marker_centers[i] for i in sorted(corner_markers.keys())]
        self.corners = self._order_points(corner_pts)

        # Perspective transform
        dst_pts = np.array([
            [0, 0],
            [map_width - 1, 0],
            [map_width - 1, map_height - 1],
            [0, map_height - 1]
        ], dtype=np.float32)

        self.transform_matrix = cv2.getPerspectiveTransform(
            self.corners.astype(np.float32),
            dst_pts
        )

        # Detect goal marker
        if goal_id in marker_centers:
            goal_raw =np.array([[marker_centers[goal_id]]], dtype=np.float32)
            goal_transf = cv2.perspectiveTransform(goal_raw, self.transform_matrix)
            self.goal_position = tuple(goal_transf[0][0])
        else:
            raise Exception("Failed to detect goal marker")

    def _order_points(self, pts):
        """Order points as [TL, TR, BR, BL]"""
        pts = np.array(pts)

        s = pts.sum(axis=1)
        tl = pts[np.argmin(s)]
        br = pts[np.argmax(s)]

        diff = np.diff(pts, axis=1)
        tr = pts[np.argmin(diff)]
        bl = pts[np.argmax(diff)]

        return np.array([tl, tr, br, bl])

    def _detect_marker_centers(self, frame, target_ids=None):
        """ Detect Aruco markers and return centers."""
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        detector = aruco.ArucoDetector(self.aruco_dict, self.aruco_params)
        markers_detected, ids, _ = detector.detectMarkers(gray)

        if ids is None:
            return {}

        marker_centers = {}
        for corners, marker_id in zip(markers_detected, ids.flatten()):
            if target_ids is None or marker_id in target_ids:
                pts = corners.reshape((4, 2))
                cx = np.mean(pts[:, 0])
                cy = np.mean(pts[:, 1])
                marker_centers[marker_id] = (cx, cy)

        return marker_centers

    def get_frame(self):
        """
        Returns:
            np.array: BGR image or None if fail
        """
        ret, frame = self.cap.read()
        return frame if ret else None

    def get_transform_frame(self):
        """
        Transformed camera frame for detection functions

        Returns:
            np.array: BGR image or None if fail
        """
        frame = self.get_frame()
        if frame is None or self.transform_matrix is None:
            return None

        transformed = cv2.warpPerspective(frame, self.transform_matrix, self.map_size)

        self._current_transformed_frame = transformed

        return transformed


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

if __name__ == "__main__":
    vision = VisionSystem()
    print("VisionSystem initialized successfully")

    if not vision.cap.isOpened():
        print("Camera failed to open")
        exit(1)

    print("Camera is active")
    print("Press 'c' to calibrate")
    print("Press 'q' to quit...")

    calibrated = False

    # Remplacer la boucle while (à partir de la ligne 151)
    while True:
        frame = vision.get_frame()
        if frame is None:
            print("Failed to capture frame")
            break

        # Recalibration continue
        marker_centers = vision._detect_marker_centers(frame, target_ids={0, 2, 3, 5})
        if len(marker_centers) == 4:
            corner_pts = [marker_centers[i] for i in sorted(marker_centers.keys())]
            vision.corners = vision._order_points(corner_pts)

            # Mettre à jour la matrice de transformation
            dst_pts = np.array([
                [0, 0],
                [800 - 1, 0],
                [800 - 1, 600 - 1],
                [0, 600 - 1]
            ], dtype=np.float32)

            vision.transform_matrix = cv2.getPerspectiveTransform(
                vision.corners.astype(np.float32),
                dst_pts
            )
            calibrated = True

        # Afficher la vue transformée si calibré
        if calibrated and vision.transform_matrix is not None:
            # Appliquer la transformation perspective pour recadrer
            warped = cv2.warpPerspective(frame, vision.transform_matrix, (800, 600))

            cv2.putText(
                warped,
                "CALIBRATED (Live)",
                (10, 30),
                cv2.FONT_HERSHEY_SIMPLEX,
                1,
                (0, 255, 0),
                2
            )

            cv2.imshow("Vision System - Calibrated View (Press 'q' to quit)", warped)
        else:
            # Afficher la vue brute avec les coins détectés
            display_frame = frame.copy()
            if vision.corners is not None:
                for i, (x, y) in enumerate(vision.corners):
                    x, y = int(x), int(y)
                    cv2.circle(display_frame, (x, y), 10, (0, 255, 0), -1)

                pts = vision.corners.astype(np.int32).reshape((-1, 1, 2))
                cv2.polylines(display_frame, [pts], True, (0, 255, 0), 2)

            cv2.imshow("Vision System - Waiting for 4 markers...", display_frame)

        key = cv2.waitKey(1) & 0xFF
        if key == ord('q'):
            break


