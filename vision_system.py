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
        self.detected_markers = {}
        self.corners = None  # [(x1, y1), ...]
        self.transform_matrix = None
        self.mm2px = None
        self.map_size = (800, 600)
        self.goal_position = None  # (x, y)

        # This works as 'cache' to make sure every function gets same frame
        # self._current_transform_frame = None

    def calibrate(self, corner_ids={0, 2, 3, 4}, goal_id=1, map_width=800, map_height=600):
        """
        Detect map corners and goal using aruco markers.

        Sets:
            self.corners, self.transform_matrix, self.mm2px, self.goal_position
        """
        self.map_size = (map_width, map_height)

        calibrated = False

        while not calibrated:
            frame = self.get_frame()
            if frame is None:
                print("Failed to capture frame")
                continue

            display_frame = frame.copy()

            # Detect all markers
            marker_centers = self._detect_marker_centers(
                frame,
                target_ids=corner_ids.union({goal_id})
            )

            # Check corner markers
            corner_markers = {id: pos for id, pos in marker_centers.items()
                              if id in corner_ids}

            # Display status
            status_text = f"Corners: {len(corner_markers)}/4"
            status_color = (0, 255, 0) if len(corner_markers) == 4 else (0, 0, 255)
            cv2.putText(display_frame, status_text, (10, 30),
                        cv2.FONT_HERSHEY_SIMPLEX, 1, status_color, 2)

            goal_detected = goal_id in marker_centers
            goal_text = f"Goal: {'DETECTED' if goal_detected else 'NOT FOUND'}"
            goal_color = (0, 255, 0) if goal_detected else (0, 0, 255)
            cv2.putText(display_frame, goal_text, (10, 70),
                        cv2.FONT_HERSHEY_SIMPLEX, 1, goal_color, 2)

            # Draw detected markers
            for marker_id, (x, y) in marker_centers.items():
                x, y = int(x), int(y)
                color = (0, 255, 0) if marker_id in corner_ids else (255, 0, 255)
                cv2.circle(display_frame, (x, y), 10, color, -1)
                cv2.putText(display_frame, f"ID:{marker_id}", (x + 15, y),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)

            # Draw corners polygon if all 4 detected
            if len(corner_markers) == 4:
                corner_pts = [marker_centers[i] for i in sorted(corner_ids)]
                ordered_corners = self._order_points(corner_pts)
                pts = ordered_corners.astype(np.int32).reshape((-1, 1, 2))
                cv2.polylines(display_frame, [pts], True, (0, 255, 0), 3)

                # Show completion hint
                cv2.putText(display_frame, "Press 'c' to complete calibration",
                            (10, map_height - 20 if map_height < 500 else 110),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)

            cv2.imshow("Calibration - Position Markers", display_frame)

            key = cv2.waitKey(1) & 0xFF

            # Complete calibration
            if key == ord('c'):
                if len(corner_markers) == 4 and goal_detected:
                    # Process corners
                    corner_pts = [marker_centers[i] for i in sorted(corner_ids)]
                    self.corners = self._order_points(corner_pts)

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
                    goal_raw = np.array([[marker_centers[goal_id]]], dtype=np.float32)
                    goal_transf = cv2.perspectiveTransform(goal_raw, self.transform_matrix)
                    self.goal_position = tuple(goal_transf[0][0])

                    print("✓ Calibration complete!")
                    print(f"  Corners: {self.corners}")
                    print(f"  Goal: {self.goal_position}")

                    calibrated = True
                    cv2.destroyWindow("Calibration - Position Markers")
                else:
                    print("Missing markers!")
                    if len(corner_markers) != 4:
                        print(f"  Need 4 corners, found {len(corner_markers)}")
                    if not goal_detected:
                        print(f"  Goal marker {goal_id} not detected")

            # Quit
            elif key == ord('q'):
                print("Calibration cancelled")
                cv2.destroyWindow("Calibration - Position Markers")
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


    def detect_robot_raw_pose(self):
        """
        This is RAW data - needs filtering.

        Returns:
            np.array: [x, y, theta] in map coordinates, or None if not detected
        """
        frame = self.get_transform_frame()

        if 4 not in self.detected_markers:
            print("Robot marker not detected!")
            return None

        if self.transform_matrix is None:
            return None

        marker_centers = self._detect_marker_centers(frame, target_ids={5})

        robot_data = self.detected_markers[5]

        center_raw = np.array([[robot_data['center']]], dtype=np.float32)
        center_transf = cv2.perspectiveTransform(center_raw, self.transform_matrix)
        x, y = center_transf[0][0]

        # Orientation, corners are: TL, TR, BR, BL
        corners = robot_data['corners']

        tl = corners[0]
        tr = corners[1]

        top_center = (tl + tr) / 2
        center = robot_data['center']

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


    def detect_obstacles(self):
        """
        Returns:
            list: List of obstacle polygons, each as np.array([[x1,y1], [x2,y2], ...])
        """
        frame = self.get_transform_frame()
        if frame is None:
            return []

        # TODO: Apply perspective transform
        # TODO: Detect shapes using contour detection
        # TODO: Filter for squares/rectangles
        # TODO: Return obstacle corner points
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

    # Remplacer la boucle while (à partir de la ligne 249)
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

            # Détecter le goal marker (ID 1)
            marker_centers = vision._detect_marker_centers(frame, target_ids={1})
            if 1 in marker_centers:
                goal_raw = np.array([[marker_centers[1]]], dtype=np.float32)
                goal_transf = cv2.perspectiveTransform(goal_raw, vision.transform_matrix)
                vision.goal_position = tuple(goal_transf[0][0])

            # Détecter les obstacles sur l'image transformée
            mask = vision.filter_color(warped)
            edges = vision.process_image(mask)
            scaled_contours, all_vertices = vision.detect_contours(edges)

            # Dessiner les contours sur l'image transformée
            display = vision.draw_contours(warped, scaled_contours, all_vertices)

            # Dessiner le goal si détecté
            if vision.goal_position is not None:
                gx, gy = int(vision.goal_position[0]), int(vision.goal_position[1])
                cv2.circle(display, (gx, gy), 5, (0, 255, 0), -1)
                cv2.circle(display, (gx, gy), 10, (0, 255, 0), 2)
                cv2.putText(
                    display,
                    "GOAL",
                    (gx - 30, gy - 25),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.6,
                    (0, 255, 0),
                    1
                )

                # Détecter et afficher le robot
                robot_pose = vision.detect_robot_raw_pose()
                if robot_pose is not None:
                    x, y, theta = robot_pose
                    rx, ry = int(x), int(y)

                    # Cercle bleu pour la position
                    cv2.circle(display, (rx, ry), 8, (255, 0, 0), -1)
                    cv2.circle(display, (rx, ry), 12, (255, 0, 0), 2)

                    # Flèche pour l'orientation
                    arrow_length = 30
                    end_x = int(rx + arrow_length * np.cos(theta))
                    end_y = int(ry + arrow_length * np.sin(theta))
                    cv2.arrowedLine(display, (rx, ry), (end_x, end_y), (255, 0, 0), 3, tipLength=0.3)

                    # Afficher les coordonnées
                    cv2.putText(
                        display,
                        f"Robot: ({rx}, {ry}, {np.degrees(theta):.1f}deg)",
                        (10, 60),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.6,
                        (255, 0, 0),
                        2
                    )

            cv2.putText(
                display,
                f"CALIBRATED - Obstacles: {len(all_vertices)}",
                (10, 30),
                cv2.FONT_HERSHEY_SIMPLEX,
                1,
                (0, 255, 0),
                2
            )

            cv2.imshow("Vision System - Calibrated View (Press 'q' to quit)", display)
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

    vision.release()
    cv2.destroyAllWindows()

