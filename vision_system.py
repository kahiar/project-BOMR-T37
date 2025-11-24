import cv2
import numpy as np
from cv2 import aruco
import matplotlib.pyplot as plt

from path_planner import PathPlanner
from visualizer import Visualizer


class VisionSystem:
    """Camera detection of robot, obstacles, and goal"""

    def __init__(self, camera_id=0, aruco_dict_type=aruco.DICT_4X4_50):
        self.cap = cv2.VideoCapture(camera_id)
        self.aruco_dict = aruco.getPredefinedDictionary(aruco_dict_type)
        self.aruco_params = aruco.DetectorParameters()

        # Initialize variables
        self.detected_markers = {}
        self.corners = None  # [(x1, y1), ...]
        self.transform_matrix = None
        self.mm2px = None
        self.map_size = (800, 600)
        self.goal_position = None  # (x, y)

        # This works as 'cache' to make sure every function gets same frame
        # self._current_transform_frame = None

    def calibrate(self, corner_ids={0, 2, 3, 5}, goal_id=1, map_width=800, map_height=600):
        """
        Detect map corners and goal using aruco markers.

        Sets:
            self.corners, self.transform_matrix, self.mm2px, self.goal_position
        """
        self.map_size = (map_width, map_height)

        print("=== Calibration Mode ===")
        print(f"Looking for corner markers: {sorted(corner_ids)}")
        print(f"Looking for goal marker: {goal_id}")

        calibrated = False

        while not calibrated:
            frame = self.get_frame()
            if frame is None:
                print("Failed to capture frame")
                continue

            # Detect all markers
            marker_centers = self._detect_marker_centers(
                frame,
                target_ids=corner_ids.union({goal_id})
            )

            # Check corner markers
            corner_markers = {id: pos for id, pos in marker_centers.items()
                              if id in corner_ids}
            goal_marker = marker_centers.get(goal_id)

            # Order corners if all detected
            corners_ordered = None
            if len(corner_markers) == 4:
                corner_pts = [marker_centers[i] for i in sorted(corner_ids)]
                corners_ordered = self._order_points(corner_pts)


            # Display everything during calibration
            display_frame = frame.copy()

            # Draw markers
            for marker_id, (x, y) in marker_centers.items():
                x, y = int(x), int(y)
                color = (0, 255, 0) if marker_id in corner_ids else (255, 0, 255)
                cv2.circle(display_frame, (x, y), 10, color, -1)
                cv2.putText(display_frame, f"ID:{marker_id}", (x + 15, y),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)

            # Draw corners
            if corners_ordered is not None:
                pts = corners_ordered.astype(np.int32).reshape((-1, 1, 2))
                cv2.polylines(display_frame, [pts], True, (0, 255, 0), 3)

            # Status
            status_text = f"Corners: {len(corner_markers)}/4"
            cv2.putText(display_frame, status_text, (10, 30),
                        cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

            cv2.imshow("Calibration", display_frame)
            key = cv2.waitKey(1) & 0xFF

            # Complete calibration
            if key == ord('c'):
                if len(corner_markers) == 4 and goal_marker:
                    # Process corners
                    self.corners = corners_ordered

                    # Compute perspective transform
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

                    # Transform goal position
                    goal_raw = np.array([[goal_marker]], dtype=np.float32)
                    goal_transf = cv2.perspectiveTransform(goal_raw, self.transform_matrix)
                    self.goal_position = np.array(goal_transf[0][0])

                    print("✓ Calibration complete!")
                    print(f"  Corners: {self.corners}")
                    print(f"  Goal: {self.goal_position}")

                    calibrated = True
                else:
                    print("⚠ Cannot complete: Missing markers!")
                    if len(corner_markers) != 4:
                        print(f"  Need 4 corners, found {len(corner_markers)}")
                    if not goal_marker:
                        print(f"  Goal marker {goal_id} not detected")

            # Quit
            elif key == ord('q'):
                print("Calibration cancelled")
                if not visualizer:
                    cv2.destroyWindow("Calibration")
                raise Exception("Calibration cancelled by user")

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
        for marker, marker_id in zip(markers_detected, ids.flatten()):
            if target_ids is None or marker_id in target_ids:
                pts = marker.reshape((4, 2))
                cx = np.mean(pts[:, 0])
                cy = np.mean(pts[:, 1])
                marker_centers[marker_id] = (cx, cy)

                # put corner coordinate for orientation
                self.detected_markers[marker_id] = {
                    'center': (cx, cy),
                    'corners': pts  # [[x1,y1], [x2,y2], [x3,y3], [x4,y4]]
                }

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

        # self._current_transformed_frame = transformed

        return transformed


    def detect_robot_raw_pose(self, frame):
        """
        This is RAW data - needs filtering.

        Returns:
            np.array: [x, y, theta] in map coordinates, or None if not detected
        """

        marker_centers = self._detect_marker_centers(frame, target_ids={4})


        if 4 not in self.detected_markers:
            print("Robot marker not detected!")
            return None

        robot_data = self.detected_markers[4]

        # Orientation, corners are: TL, TR, BR, BL
        corners = robot_data['corners']

        tl = corners[0]
        tr = corners[1]

        top_center = (tl + tr) / 2
        center = robot_data['center']
        x, y = center

        dx = top_center[0] - center[0]
        dy = top_center[1] - center[1]

        # Compute angle in radians
        theta = np.arctan2(dy, dx)


        return np.array([x, y, theta])


    # 1. Filtrer la couleur (bleu)
    def filter_color(self, image):
        hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)

        LOWER_BLUE = np.array([90, 50, 50])
        UPPER_BLUE = np.array([130, 255, 255])

        mask = cv2.inRange(hsv, LOWER_BLUE, UPPER_BLUE)

        # nettoyage du masque avec des opérations morphologiques
        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, np.ones((5,5),np.uint8))
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, np.ones((5,5),np.uint8))

        return mask
    
    def process_image(self, mask):
        # Applique un flou et une détection de contours
        blurred = cv2.GaussianBlur(mask, (3, 3), 0)
        edges = cv2.Canny(blurred, 50, 150, 7, L2gradient=True)
        dilated_edges = cv2.dilate(edges, kernel=np.ones((5, 5), np.uint8), iterations=1)
        return dilated_edges
    
    def scale_contour(self, contour, scale):
        M = cv2.moments(contour)
        if M['m00'] == 0:
            return contour
        
        cx = int(M['m10'] / M['m00'])
        cy = int(M['m01'] / M['m00'])
        center = np.array([cx, cy])
        
        scaled = (contour - center) * scale + center
        return scaled.astype(np.int32)
    
    def detect_contours(self, dilated_edges):
        contours, _ = cv2.findContours(dilated_edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        scaled_contours = []
        all_vertices = []
  
        for cnt in contours:
            approx = cv2.approxPolyDP(cnt, 0.02 * cv2.arcLength(cnt, True), True)
            scaled = self.scale_contour(approx, scale=1.5)
            scaled_contours.append(scaled)

            vertices = scaled.reshape(-1, 2)   # (4,2)
            all_vertices.append(vertices)

        return scaled_contours, all_vertices
    
    def draw_contours(self, image, scaled_contours, all_vertices):
        output = image.copy()

        # Dessiner les vertices (points)
        for vertices in all_vertices:
            for (x, y) in vertices:
                cv2.circle(output, (x, y), 5, (0, 0, 255), -1)

        # Dessiner contours scalés
        cv2.drawContours(output, scaled_contours, -1, (0, 0, 255), 1)

        return output


    def detect_obstacles(self, frame):
        """
        Returns:
            list: List of obstacle polygons, each as np.array([[x1,y1], [x2,y2], ...])
        """
        if frame is None:
            return []

        mask = self.filter_color(frame)
        edges = self.process_image(mask)
        scaled_contours, all_vertices = self.detect_contours(edges)
        obstacles = all_vertices
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

vision = VisionSystem()
visualizer = Visualizer(window_name="Robot Debug View")
planner = PathPlanner()

try:
    # === PHASE 1: CALIBRATION (using the calibrate method!) ===
    vision.calibrate(
        corner_ids={0, 2, 3, 5},
        goal_id=1,
        map_width=800,
        map_height=600,
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
        start = np.array([vision.goal_position[0], vision.goal_position[1]])
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
