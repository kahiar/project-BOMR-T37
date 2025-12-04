import cv2
import numpy as np
from cv2 import aruco


class VisionSystem:
    """Camera-based detection of robot, obstacles, and goal using ArUco markers."""

    def __init__(self, camera_id=0, aruco_dict_type=aruco.DICT_4X4_50):
        """
        Initialize vision system with camera and ArUco detector.

        Args:
            camera_id: int, camera device index
            aruco_dict_type: ArUco dictionary type for marker detection
        """
        self.cap = cv2.VideoCapture(camera_id)
        self.aruco_dict = aruco.getPredefinedDictionary(aruco_dict_type)
        self.aruco_params = aruco.DetectorParameters()

        self.detected_markers = {}
        self.corners = None
        self.transform_matrix = None
        self.mm2px = None
        self.map_size = (900, 600)
        self.goal_position = None

    def calibrate(self, corner_ids={0, 2, 3, 5}, goal_id=1,
                  map_width=900, map_height=600, real_height=655,
                  stability_frames=15):
        """
        Auto-calibrate by detecting map corners and goal markers.
        Start navigation once all markers are stable for several frames.

        Args:
            corner_ids: set of int, ArUco IDs for the 4 corner markers
            goal_id: int, ArUco ID for the goal marker
            map_width: int, output map width in pixels
            map_height: int, output map height in pixels
            real_height: float, real-world height in mm (for mm2px conversion)
            stability_frames: int, consecutive frames with all markers before proceeding
        """
        self.map_size = (map_width, map_height)
        stable_count = 0

        print("=== Auto-Calibration ===")
        print(f"Waiting for corners {sorted(corner_ids)} and goal {goal_id}...")

        while True:
            frame = self.get_frame()
            if frame is None:
                continue

            marker_centers = self._detect_marker_centers(
                frame, target_ids=corner_ids.union({goal_id})
            )

            corner_markers = {id: pos for id, pos in marker_centers.items() if id in corner_ids}
            goal_marker = marker_centers.get(goal_id)
            all_found = len(corner_markers) == 4 and goal_marker is not None

            if all_found:
                stable_count += 1
            else:
                stable_count = 0

            # Display calibration progress
            display_frame = frame.copy()

            for marker_id, (x, y) in marker_centers.items():
                color = (0, 255, 0) if marker_id in corner_ids else (255, 0, 255)
                cv2.circle(display_frame, (int(x), int(y)), 10, color, -1)
                cv2.putText(display_frame, f"ID:{marker_id}", (int(x) + 15, int(y)),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)

            if len(corner_markers) == 4:
                corner_pts = [marker_centers[i] for i in sorted(corner_ids)]
                corners_ordered = self._order_points(corner_pts)
                pts = corners_ordered.astype(np.int32).reshape((-1, 1, 2))
                cv2.polylines(display_frame, [pts], True, (0, 255, 0), 3)

            status_color = (0, 255, 0) if all_found else (0, 0, 255)
            cv2.putText(display_frame, f"Corners: {len(corner_markers)}/4", (10, 30),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.8, status_color, 2)
            cv2.putText(display_frame, f"Goal: {'OK' if goal_marker else 'MISSING'}", (10, 60),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.8, status_color, 2)

            if all_found:
                progress = int((stable_count / stability_frames) * 100)
                cv2.putText(display_frame, f"Stabilizing: {progress}%", (10, 90),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 255), 2)

            cv2.imshow("Calibration", display_frame)
            key = cv2.waitKey(1) & 0xFF

            if key == ord('q'):
                cv2.destroyWindow("Calibration")
                raise Exception("Calibration cancelled by user")

            # Auto-complete when stable
            if stable_count >= stability_frames:
                corner_pts = [marker_centers[i] for i in sorted(corner_ids)]
                self.corners = self._order_points(corner_pts)

                dst_pts = np.array([
                    [0, 0],
                    [map_width - 1, 0],
                    [map_width - 1, map_height - 1],
                    [0, map_height - 1]
                ], dtype=np.float32)

                self.transform_matrix = cv2.getPerspectiveTransform(
                    self.corners.astype(np.float32), dst_pts
                )

                self.mm2px = map_height / real_height

                goal_raw = np.array([[goal_marker]], dtype=np.float32)
                goal_transf = cv2.perspectiveTransform(goal_raw, self.transform_matrix)
                self.goal_position = np.array(goal_transf[0][0])

                cv2.destroyWindow("Calibration")
                print("Calibration complete!")
                print(f"  Goal position: {self.goal_position}")
                return

    def _order_points(self, pts):
        """
        Order points as [TL, TR, BR, BL].

        Args:
            pts: list of (x, y) tuples

        Returns:
            np.array: Ordered points array of shape (4, 2)
        """
        pts = np.array(pts)
        s = pts.sum(axis=1)
        tl = pts[np.argmin(s)]
        br = pts[np.argmax(s)]
        diff = np.diff(pts, axis=1)
        tr = pts[np.argmin(diff)]
        bl = pts[np.argmax(diff)]
        return np.array([tl, tr, br, bl])

    def _detect_marker_centers(self, frame, target_ids=None):
        """
        Detect ArUco markers and return their centers.

        Args:
            frame: np.array, BGR image
            target_ids: set of int marker IDs to detect, or None for all

        Returns:
            dict: {marker_id: (center_x, center_y)}
        """
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        detector = aruco.ArucoDetector(self.aruco_dict, self.aruco_params)
        markers_detected, ids, _ = detector.detectMarkers(gray)

        if ids is None:
            return {}

        marker_centers = {}
        for marker, marker_id in zip(markers_detected, ids.flatten()):
            if target_ids is None or marker_id in target_ids:
                pts = marker.reshape((4, 2))
                cx, cy = np.mean(pts[:, 0]), np.mean(pts[:, 1])
                marker_centers[marker_id] = (cx, cy)
                self.detected_markers[marker_id] = {
                    'center': (cx, cy),
                    'corners': pts
                }
        return marker_centers

    def get_frame(self):
        """
        Capture a raw frame from the camera.

        Returns:
            np.array: BGR image, or None if capture failed
        """
        ret, frame = self.cap.read()
        return frame if ret else None

    def get_transform_frame(self):
        """
        Get perspective-transformed frame.

        Returns:
            np.array: Transformed BGR image, or None if not calibrated or capture failed
        """
        frame = self.get_frame()
        if frame is None or self.transform_matrix is None:
            return None
        return cv2.warpPerspective(frame, self.transform_matrix, self.map_size)

    def detect_robot_raw_pose(self, frame):
        """
        Detect robot pose from ArUco marker (ID 4).

        Args:
            frame: np.array, transformed BGR image

        Returns:
            np.array: [x, y, theta] pose in pixels for position, radians for angle, or None if not detected
        """
        marker_centers = self._detect_marker_centers(frame, target_ids={4})


        if 4 not in marker_centers:
            return None

        robot_data = self.detected_markers[4]
        corners = robot_data['corners']

        # Orientation from top edge of marker
        top_center = (corners[0] + corners[1]) / 2
        center = np.array(robot_data['center'])

        dx = top_center[0] - center[0]
        dy = top_center[1] - center[1]
        theta = np.arctan2(dy, dx)

        return np.array([center[0], center[1], theta])

    def detect_obstacles(self, frame):
        """
        Detect blue rectangular obstacles.

        Args:
            frame: np.array, transformed BGR image

        Returns:
            list: Obstacle polygons as list of np.array with shape (4, 2)
        """
        if frame is None:
            return []

        mask = self._filter_blue(frame)
        edges = self._process_edges(mask)
        _, vertices = self._detect_rectangles(edges)
        return vertices

    def _filter_blue(self, image):
        """
        Extract blue regions using HSV color filtering.

        Args:
            image: np.array, BGR image

        Returns:
            np.array: Binary mask of blue regions
        """
        hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
        mask = cv2.inRange(hsv, np.array([90, 50, 50]), np.array([130, 255, 255]))
        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, np.ones((5, 5), np.uint8))
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, np.ones((5, 5), np.uint8))
        return mask

    def _process_edges(self, mask):
        """
        Apply blur, edge detection, and dilation.

        Args:
            mask: np.array, binary mask

        Returns:
            np.array: Edge image
        """
        blurred = cv2.GaussianBlur(mask, (3, 3), 0)
        edges = cv2.Canny(blurred, 50, 150, 7, L2gradient=True)
        return cv2.dilate(edges, np.ones((5, 5), np.uint8), iterations=1)

    def _detect_rectangles(self, edges):
        """
        Find rectangular contours and scale them for safety margin.

        Args:
            edges: np.array, edge image

        Returns:
            tuple: (scaled_contours, vertices_list) where vertices_list contains
                   np.array of shape (4, 2) for each rectangle
        """
        contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        scaled_contours = []
        all_vertices = []

        for cnt in contours:
            approx = cv2.approxPolyDP(cnt, 0.02 * cv2.arcLength(cnt, True), True)
            if len(approx) != 4:
                continue

            scaled = self._scale_contour(approx, scale=2)
            scaled_contours.append(scaled)
            all_vertices.append(scaled.reshape(-1, 2))

        return scaled_contours, all_vertices

    def _scale_contour(self, contour, scale):
        """
        Scale contour outward from its centroid.

        Args:
            contour: np.array, contour points
            scale: float, scale factor (>1 expands, <1 shrinks)

        Returns:
            np.array: Scaled contour
        """
        M = cv2.moments(contour)
        if M['m00'] == 0:
            return contour

        cx = int(M['m10'] / M['m00'])
        cy = int(M['m01'] / M['m00'])
        center = np.array([cx, cy])

        scaled = (contour - center) * scale + center
        return scaled.astype(np.int32)

    def get_goal_position(self):
        """
        Get goal position.

        Returns:
            np.array: [x, y] goal position in pixels
        """
        return self.goal_position

    def release(self):
        """Release camera resources."""
        self.cap.release()