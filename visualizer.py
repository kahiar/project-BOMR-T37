import cv2
import numpy as np


class Visualizer:
    """Real-time navigation display with side panel for sensors and status."""

    def __init__(self, window_name="Robot Navigation", panel_width=320):
        """
        Initialize visualizer window.

        Args:
            window_name: str, title of the display window
            panel_width: int, width of side panel in pixels
        """
        self.window_name = window_name
        self.panel_width = panel_width
        cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)

    def draw_obstacles(self, frame, obstacles):
        """
        Draw obstacle polygons with semi-transparent fill.

        Args:
            frame: np.array, BGR image to draw on
            obstacles: list of np.array polygons

        Returns:
            np.array: Frame with obstacles drawn
        """
        for obstacle in obstacles:
            overlay = frame.copy()
            cv2.fillPoly(overlay, [obstacle.astype(np.int32)], (0, 0, 255))
            frame = cv2.addWeighted(frame, 0.7, overlay, 0.3, 0)
            cv2.polylines(frame, [obstacle.astype(np.int32)], True, (0, 0, 255), 2)
            for vertex in obstacle:
                cv2.circle(frame, tuple(vertex.astype(int)), 4, (0, 0, 255), -1)
        return frame

    def draw_uncertainty_ellipse(self, frame, position, covariance, robot_detected):
        """
        Draw uncertainty ellipse from Kalman filter covariance.
        Only visible when robot marker is not detected.

        Args:
            frame: np.array, BGR image to draw on
            position: np.array [x, y, theta] Kalman state
            covariance: np.array 3x3 covariance matrix
            robot_detected: bool, whether vision detected the robot

        Returns:
            np.array: Frame with ellipse drawn (if applicable)
        """
        if robot_detected or covariance is None or position is None:
            return frame

        x, y = int(position[0]), int(position[1])

        # Extract position covariance (2x2 submatrix)
        pos_cov = covariance[0:2, 0:2]

        # Compute eigenvalues and eigenvectors for ellipse orientation
        eigenvalues, eigenvectors = np.linalg.eigh(pos_cov)

        # Sort by eigenvalue (largest first)
        order = eigenvalues.argsort()[::-1]
        eigenvalues = eigenvalues[order]
        eigenvectors = eigenvectors[:, order]

        # Ellipse axes (2 standard deviations = ~95% confidence)
        axis_lengths = 2 * np.sqrt(eigenvalues)
        axes = (int(axis_lengths[0]), int(axis_lengths[1]))

        # Ellipse angle from eigenvector
        angle = np.degrees(np.arctan2(eigenvectors[1, 0], eigenvectors[0, 0]))

        cv2.ellipse(frame, (x, y), axes, angle, 0, 360, (0, 255, 255), 2)
        cv2.putText(frame, "PREDICTED", (x - 40, y - axes[1] - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 2)

        return frame

    def draw_robot(self, frame, robot_pos):
        """
        Draw robot position and orientation arrow.

        Args:
            frame: np.array, BGR image to draw on
            robot_pos: np.array [x, y, theta] robot pose, or None

        Returns:
            np.array: Frame with robot drawn
        """
        if robot_pos is None:
            return frame

        x, y, theta = robot_pos
        cv2.circle(frame, (int(x), int(y)), 12, (255, 0, 0), -1)
        cv2.circle(frame, (int(x), int(y)), 15, (255, 255, 255), 2)

        arrow_length = 35
        end_x = int(x + arrow_length * np.cos(theta))
        end_y = int(y + arrow_length * np.sin(theta))
        cv2.arrowedLine(frame, (int(x), int(y)), (end_x, end_y),
                        (255, 255, 0), 3, tipLength=0.4)
        return frame

    def draw_goal(self, frame, goal_pos):
        """
        Draw goal target symbol.

        Args:
            frame: np.array, BGR image to draw on
            goal_pos: tuple (x, y) goal position, or None

        Returns:
            np.array: Frame with goal drawn
        """
        if goal_pos is None:
            return frame

        gx, gy = int(goal_pos[0]), int(goal_pos[1])
        cv2.circle(frame, (gx, gy), 15, (0, 255, 0), 3)
        cv2.circle(frame, (gx, gy), 8, (0, 255, 0), 2)
        cv2.circle(frame, (gx, gy), 3, (0, 255, 0), -1)
        cv2.line(frame, (gx - 20, gy), (gx + 20, gy), (0, 255, 0), 2)
        cv2.line(frame, (gx, gy - 20), (gx, gy + 20), (0, 255, 0), 2)
        cv2.putText(frame, "GOAL", (gx - 30, gy - 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
        return frame

    def draw_path(self, frame, path, current_waypoint_idx):
        """
        Draw planned path with waypoint highlighting.

        Args:
            frame: np.array, BGR image to draw on
            path: list of np.array waypoints, or None
            current_waypoint_idx: int, index of current target waypoint

        Returns:
            np.array: Frame with path drawn
        """
        if path is None or len(path) == 0:
            return frame

        for i in range(len(path) - 1):
            pt1 = (int(path[i][0]), int(path[i][1]))
            pt2 = (int(path[i + 1][0]), int(path[i + 1][1]))
            cv2.line(frame, pt1, pt2, (255, 165, 0), 2)

        for i, waypoint in enumerate(path):
            if i == current_waypoint_idx:
                color, radius = (255, 255, 0), 8
            elif i < current_waypoint_idx:
                color, radius = (128, 128, 128), 4
            else:
                color, radius = (255, 200, 200), 5
            cv2.circle(frame, (int(waypoint[0]), int(waypoint[1])), radius, color, -1)

        return frame

    def _create_side_panel(self, height, sensor_data, info_dict, robot_detected, kalman_state):
        """
        Create side panel with sensors and status info.

        Args:
            height: int, panel height in pixels
            sensor_data: np.array of 5 proximity sensor values, or None
            info_dict: dict of status information to display
            robot_detected: bool, whether vision detected the robot
            kalman_state: np.array [x, y, theta] filtered pose

        Returns:
            np.array: BGR image of the side panel
        """
        panel = np.zeros((height, self.panel_width, 3), dtype=np.uint8)
        panel[:] = (30, 30, 30)

        y_offset = 30

        # Title
        cv2.putText(panel, "NAVIGATION STATUS", (10, y_offset),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
        y_offset += 40

        # Robot detection status
        status_color = (0, 255, 0) if robot_detected else (0, 165, 255)
        status_text = "TRACKING" if robot_detected else "PREDICTION"
        cv2.putText(panel, f"Mode: {status_text}", (10, y_offset),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, status_color, 1)
        y_offset += 30

        # Kalman state info
        if kalman_state is not None:
            cv2.putText(panel, f"X: {kalman_state[0]:.1f} px", (10, y_offset),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
            y_offset += 25
            cv2.putText(panel, f"Y: {kalman_state[1]:.1f} px", (10, y_offset),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
            y_offset += 25
            cv2.putText(panel, f"Theta: {np.degrees(kalman_state[2]):.1f} deg", (10, y_offset),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
            y_offset += 40

        # Separator
        cv2.line(panel, (10, y_offset), (self.panel_width - 10, y_offset), (80, 80, 80), 1)
        y_offset += 20

        # Proximity sensors
        cv2.putText(panel, "PROXIMITY SENSORS", (10, y_offset),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
        y_offset += 25

        if sensor_data is not None:
            bar_width = 45
            bar_max_height = 60
            max_sensor_value = 4500

            for i in range(5):
                value = min(sensor_data[i], max_sensor_value)
                bar_height = int((value / max_sensor_value) * bar_max_height)
                x = 15 + i * 58

                if value > 3000:
                    color = (0, 0, 255)
                elif value > 1500:
                    color = (0, 165, 255)
                else:
                    color = (0, 255, 0)

                bar_y = y_offset + bar_max_height - bar_height
                cv2.rectangle(panel, (x, bar_y), (x + bar_width, y_offset + bar_max_height),
                              color, -1)
                cv2.rectangle(panel, (x, y_offset), (x + bar_width, y_offset + bar_max_height),
                              (100, 100, 100), 1)

                labels = ["FL", "L", "C", "R", "FR"]
                cv2.putText(panel, labels[i], (x + 12, y_offset + bar_max_height + 15),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.4, (200, 200, 200), 1)

            y_offset += bar_max_height + 35

        # Separator
        cv2.line(panel, (10, y_offset), (self.panel_width - 10, y_offset), (80, 80, 80), 1)
        y_offset += 20

        # Additional info
        if info_dict:
            for key, value in info_dict.items():
                cv2.putText(panel, f"{key}: {value}", (10, y_offset),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.45, (180, 180, 180), 1)
                y_offset += 22

        return panel

    def update(self, frame, obstacles, robot_pos, path=None, current_waypoint_idx=0,
               sensor_data=None, goal_pos=None, info_dict=None,
               kalman_state=None, kalman_covariance=None):
        """
        Update display with camera frame and side panel.

        Args:
            frame: np.array, camera frame (transformed)
            obstacles: list of np.array obstacle polygons
            robot_pos: np.array [x, y, theta] raw vision pose, or None if not detected
            path: list of np.array waypoints, or None
            current_waypoint_idx: int, index of current target waypoint
            sensor_data: np.array of 5 proximity sensor values, or None
            goal_pos: tuple (x, y) goal position, or None
            info_dict: dict of additional status information
            kalman_state: np.array [x, y, theta] filtered pose from Kalman filter
            kalman_covariance: np.array 3x3 covariance matrix from Kalman filter
        """
        if frame is None:
            return

        display_frame = frame.copy()
        robot_detected = robot_pos is not None

        # Draw elements on camera frame
        display_frame = self.draw_obstacles(display_frame, obstacles)
        display_frame = self.draw_goal(display_frame, goal_pos)

        if path is not None:
            display_frame = self.draw_path(display_frame, path, current_waypoint_idx)

        # Draw uncertainty ellipse when robot not detected
        display_frame = self.draw_uncertainty_ellipse(
            display_frame, kalman_state, kalman_covariance, robot_detected
        )

        # Always display Kalman filter state (required for grading)
        display_frame = self.draw_robot(display_frame, kalman_state)

        # Create side panel
        side_panel = self._create_side_panel(
            frame.shape[0], sensor_data, info_dict, robot_detected, kalman_state
        )

        # Combine frame and panel horizontally
        combined = np.hstack([display_frame, side_panel])

        cv2.imshow(self.window_name, combined)
        cv2.waitKey(1)

    def close(self):
        """Release window resources."""
        cv2.destroyAllWindows()