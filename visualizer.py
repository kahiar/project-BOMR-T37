import cv2
import numpy as np


class Visualizer:
    """Camera view with overlays and sensor readings"""

    def __init__(self, window_name="Robot Navigation"):
        self.window_name = window_name
        cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)

    def draw_calibration_view(self, frame, corner_markers, goal_marker, corners_ordered=None):
        """
        Draw calibration phase visualization.

        Args:
            frame: np.array, BGR image (raw camera frame)
            corner_markers: dict {marker_id: (x, y), ...}
            goal_marker: tuple (x, y) or None
            corners_ordered: np.array of 4 ordered corner points or None
        """
        display = frame.copy()

        # Draw corner markers
        for marker_id, (x, y) in corner_markers.items():
            x, y = int(x), int(y)
            cv2.circle(display, (x, y), 10, (0, 255, 0), -1)
            cv2.putText(display, f"C{marker_id}", (x + 15, y),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)

        # Draw goal marker
        if goal_marker is not None:
            gx, gy = int(goal_marker[0]), int(goal_marker[1])
            cv2.circle(display, (gx, gy), 10, (255, 0, 255), -1)
            cv2.putText(display, "GOAL", (gx + 15, gy),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 0, 255), 2)

        # Draw polygon if all corners detected
        if corners_ordered is not None and len(corners_ordered) == 4:
            pts = corners_ordered.astype(np.int32).reshape((-1, 1, 2))
            cv2.polylines(display, [pts], True, (0, 255, 0), 3)

        # Status text
        status = f"Corners: {len(corner_markers)}/4"
        color = (0, 255, 0) if len(corner_markers) == 4 else (0, 0, 255)
        cv2.putText(display, status, (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, color, 2)

        goal_status = f"Goal: {'FOUND' if goal_marker else 'MISSING'}"
        goal_color = (0, 255, 0) if goal_marker else (0, 0, 255)
        cv2.putText(display, goal_status, (10, 70),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, goal_color, 2)

        if len(corner_markers) == 4 and goal_marker:
            cv2.putText(display, "Press 'c' to complete calibration",
                        (10, 110), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)

        cv2.imshow(self.window_name, display)
        return cv2.waitKey(1) & 0xFF

    def draw_obstacles(self, frame, obstacles):
        """
        Draw obstacles on frame.

        Args:
            frame: np.array, BGR image
            obstacles: list of obstacle polygons

        Returns:
            np.array: Frame with obstacles drawn
        """
        for obstacle in obstacles:
            # Draw filled polygon with transparency
            overlay = frame.copy()
            cv2.fillPoly(overlay, [obstacle.astype(np.int32)], (0, 0, 255))
            frame = cv2.addWeighted(frame, 0.7, overlay, 0.3, 0)

            # Draw outline
            cv2.polylines(frame, [obstacle.astype(np.int32)], True, (0, 0, 255), 2)

            # Draw vertices
            for vertex in obstacle:
                cv2.circle(frame, tuple(vertex.astype(int)), 4, (0, 0, 255), -1)

        return frame

    def draw_robot(self, frame, robot_pos):
        """
        Draw robot position and orientation.

        Args:
            frame: np.array, BGR image
            robot_pos: np.array [x, y, theta] or None

        Returns:
            np.array: Frame with robot drawn
        """
        if robot_pos is None:
            return frame

        x, y, theta = robot_pos

        # Draw robot body (circle)
        cv2.circle(frame, (int(x), int(y)), 12, (255, 0, 0), -1)
        cv2.circle(frame, (int(x), int(y)), 15, (255, 255, 255), 2)

        # Draw orientation arrow
        arrow_length = 35
        end_x = int(x + arrow_length * np.cos(theta))
        end_y = int(y + arrow_length * np.sin(theta))
        cv2.arrowedLine(frame, (int(x), int(y)), (end_x, end_y),
                        (255, 255, 0), 3, tipLength=0.4)

        # Draw position text
        pos_text = f"({int(x)}, {int(y)})"
        cv2.putText(frame, pos_text, (int(x) + 20, int(y) - 20),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2)

        # Draw angle text
        angle_deg = np.degrees(theta)
        angle_text = f"{angle_deg:.1f}deg"
        cv2.putText(frame, angle_text, (int(x) + 20, int(y) - 5),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2)

        return frame

    def draw_goal(self, frame, goal_pos):
        """
        Draw goal position.

        Args:
            frame: np.array, BGR image
            goal_pos: tuple (x, y) or None

        Returns:
            np.array: Frame with goal drawn
        """
        if goal_pos is None:
            return frame

        gx, gy = int(goal_pos[0]), int(goal_pos[1])

        # Draw target symbol
        cv2.circle(frame, (gx, gy), 15, (0, 255, 0), 3)
        cv2.circle(frame, (gx, gy), 8, (0, 255, 0), 2)
        cv2.circle(frame, (gx, gy), 3, (0, 255, 0), -1)

        # Draw crosshair
        cv2.line(frame, (gx - 20, gy), (gx + 20, gy), (0, 255, 0), 2)
        cv2.line(frame, (gx, gy - 20), (gx, gy + 20), (0, 255, 0), 2)

        # Label
        cv2.putText(frame, "GOAL", (gx - 30, gy - 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)

        return frame

    def draw_path(self, frame, path, current_waypoint_idx):
        """
        Draw planned path and highlight current target.

        Args:
            frame: np.array, BGR image
            path: list of np.array or tuples [(x,y), ...] or None
            current_waypoint_idx: int, index of current target

        Returns:
            np.array: Frame with path drawn
        """
        if path is None or len(path) == 0:
            return frame

        # Draw path lines
        for i in range(len(path) - 1):
            pt1 = (int(path[i][0]), int(path[i][1]))
            pt2 = (int(path[i + 1][0]), int(path[i + 1][1]))
            cv2.line(frame, pt1, pt2, (255, 165, 0), 2)

        # Draw waypoints
        for i, waypoint in enumerate(path):
            if i == current_waypoint_idx:
                color = (255, 255, 0)
                radius = 8
            elif i < current_waypoint_idx:
                color = (128, 128, 128)
                radius = 4
            else:
                color = (255, 200, 200)
                radius = 5

            cv2.circle(frame, (int(waypoint[0]), int(waypoint[1])), radius, color, -1)

        return frame

    def draw_sensor_panel(self, frame, sensor_data):
        """
        Draw proximity sensor visualization panel.

        Args:
            frame: np.array, BGR image
            sensor_data: np.array, proximity values [5 sensors]

        Returns:
            np.array: Frame with sensor panel
        """
        # Create sensor visualization in corner
        panel_height = 120
        panel_width = 320
        panel = np.zeros((panel_height, panel_width, 3), dtype=np.uint8)
        panel[:] = (40, 40, 40)  # Dark background

        # Title
        cv2.putText(panel, "Proximity Sensors", (10, 20),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)

        # Draw 5 bars representing sensors
        bar_width = 50
        max_sensor_value = 4500

        for i in range(5):
            value = min(sensor_data[i], max_sensor_value)
            bar_height = int((value / max_sensor_value) * (panel_height - 40))
            x = 15 + i * 60
            y = panel_height - 15 - bar_height

            # Color gradient based on value
            if value > 3000:
                color = (0, 0, 255)  # Red - danger
            elif value > 1500:
                color = (0, 165, 255)  # Orange - warning
            else:
                color = (0, 255, 255)  # Yellow - clear

            cv2.rectangle(panel, (x, y), (x + bar_width, panel_height - 15),
                          color, -1)
            cv2.rectangle(panel, (x, y), (x + bar_width, panel_height - 15),
                          (255, 255, 255), 1)

            # Sensor label
            cv2.putText(panel, f"S{i}", (x + 15, panel_height - 5),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 255), 1)

            # Value
            cv2.putText(panel, f"{int(value)}", (x + 5, y - 5),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.3, (255, 255, 255), 1)

        # Overlay on frame
        frame[10:10 + panel_height, 10:10 + panel_width] = panel

        return frame

    def draw_info_panel(self, frame, info_dict):
        """
        Draw information panel with status text.

        Args:
            frame: np.array, BGR image
            info_dict: dict with status information
        """
        panel_height = 100
        panel_width = 300

        # Position at bottom right
        y_start = frame.shape[0] - panel_height - 10
        x_start = frame.shape[1] - panel_width - 10

        # Create semi-transparent panel
        overlay = frame.copy()
        cv2.rectangle(overlay, (x_start, y_start),
                      (x_start + panel_width, y_start + panel_height),
                      (40, 40, 40), -1)
        frame = cv2.addWeighted(frame, 0.7, overlay, 0.3, 0)

        # Draw info text
        y_offset = y_start + 25
        for key, value in info_dict.items():
            text = f"{key}: {value}"
            cv2.putText(frame, text, (x_start + 10, y_offset),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
            y_offset += 25

        return frame

    def update(self, frame, obstacles, robot_pos, path=None,
               current_waypoint_idx=0, sensor_data=None, goal_pos=None,
               info_dict=None):
        """
        Update display with all visualization elements (navigation mode).

        Args:
            frame: np.array, camera frame (transformed)
            obstacles: list of obstacle polygons
            robot_pos: np.array [x, y, theta] or None
            path: list of waypoints or None
            current_waypoint_idx: int
            sensor_data: np.array, proximity values or None
            goal_pos: tuple (x, y) or None
            info_dict: dict with additional info or None
        """
        if frame is None:
            return

        display_frame = frame.copy()

        # Draw all elements in order (back to front)
        display_frame = self.draw_obstacles(display_frame, obstacles)
        display_frame = self.draw_goal(display_frame, goal_pos)

        if path is not None:
            display_frame = self.draw_path(display_frame, path, current_waypoint_idx)

        display_frame = self.draw_robot(display_frame, robot_pos)

        #if sensor_data is not None:
        #    display_frame = self.draw_sensor_panel(display_frame, sensor_data)

        if info_dict is not None:
            display_frame = self.draw_info_panel(display_frame, info_dict)

        cv2.imshow(self.window_name, display_frame)
        cv2.waitKey(1)

    def close(self):
        """Close visualization window"""
        cv2.destroyAllWindows()