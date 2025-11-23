import cv2
import numpy as np


class Visualizer:
    """Camera view with overlays and sensor readings"""

    def __init__(self, window_name="Robot Navigation"):
        self.window_name = window_name
        cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)

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
            cv2.polylines(frame, [obstacle.astype(np.int32)], True, (0, 0, 255), 2)
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
        # Draw robot position
        cv2.circle(frame, (int(x), int(y)), 10, (0, 255, 0), -1)

        # Draw orientation arrow
        arrow_length = 30
        end_x = int(x + arrow_length * np.cos(theta))
        end_y = int(y + arrow_length * np.sin(theta))
        cv2.arrowedLine(frame, (int(x), int(y)), (end_x, end_y), (0, 255, 0), 2)

        return frame

    def draw_path(self, frame, path, current_waypoint_idx):
        """
        Draw planned path and highlight current target.

        Args:
            frame: np.array, BGR image
            path: list of waypoints [(x,y), ...]
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
            cv2.line(frame, pt1, pt2, (255, 0, 0), 2)

        # Draw waypoints
        for i, waypoint in enumerate(path):
            color = (255, 255, 0) if i == current_waypoint_idx else (255, 200, 200)
            cv2.circle(frame, (int(waypoint[0]), int(waypoint[1])), 5, color, -1)

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
        panel_height = 100
        panel_width = 300
        panel = np.zeros((panel_height, panel_width, 3), dtype=np.uint8)

        # Draw 5 bars representing sensors
        bar_width = 50
        max_sensor_value = 4500

        for i in range(5):
            value = min(sensor_data[i], max_sensor_value)
            bar_height = int((value / max_sensor_value) * (panel_height - 20))
            x = 10 + i * 55
            y = panel_height - 10 - bar_height
            cv2.rectangle(panel, (x, y), (x + bar_width, panel_height - 10),
                          (0, 255, 255), -1)
            cv2.putText(panel, str(i), (x + 15, panel_height - 5),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 255), 1)

        # Overlay on frame
        frame[10:10 + panel_height, 10:10 + panel_width] = panel

        return frame

    def update(self, frame, obstacles, robot_pos, path, current_waypoint_idx, sensor_data):
        """
        Update display with all visualization elements.

        Args:
            frame: np.array, camera frame
            obstacles: list of obstacle polygons
            robot_pos: np.array [x, y, theta] or None
            path: list of waypoints
            current_waypoint_idx: int
            sensor_data: np.array, proximity values
        """
        if frame is None:
            return

        display_frame = frame.copy()

        # Draw all elements
        display_frame = self.draw_obstacles(display_frame, obstacles)
        display_frame = self.draw_path(display_frame, path, current_waypoint_idx)
        display_frame = self.draw_robot(display_frame, robot_pos)
        display_frame = self.draw_sensor_panel(display_frame, sensor_data)

        cv2.imshow(self.window_name, display_frame)
        cv2.waitKey(1)

    def close(self):
        """Close visualization window"""
        cv2.destroyAllWindows()
