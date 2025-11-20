"""
Robot Navigation Integration Module

This module integrates the path planning (shortest_path.py) with the motion control
(motion_control.py) to provide a complete navigation solution.

Example usage:
    from robot_navigation import RobotNavigator
    
    # Create navigator
    navigator = RobotNavigator()
    
    # Set up obstacles and goal
    obstacles = [quad1, quad2, quad3]  # List of obstacle polygons
    goal = np.array([750, 750])
    
    # Process frame from camera
    command = navigator.process_frame(frame, obstacles, goal)
    
    # Send command to robot
    if command:
        send_to_robot(command['left_speed'], command['right_speed'])
"""

import numpy as np
import cv2
from typing import List, Optional, Tuple
from shortest_path import a_star, build_visibility_graph
from motion_control import (
    motion_controller,
    detect_robot_position_and_orientation,
    NEAR_GOAL_RADIUS
)


class RobotNavigator:
    """
    Main navigation controller that integrates pathfinding and motion control.
    """
    
    def __init__(self, 
                 near_goal_radius: float = NEAR_GOAL_RADIUS,
                 path_replan_threshold: float = 100.0):
        """
        Initialize the robot navigator.
        
        Parameters
        ----------
        near_goal_radius : float
            Radius around goal to trigger final approach behavior
        path_replan_threshold : float
            Distance threshold for replanning (if robot drifts too far from path)
        """
        self.near_goal_radius = near_goal_radius
        self.path_replan_threshold = path_replan_threshold
        
        # Current navigation state
        self.current_path = None
        self.goal_position = None
        self.obstacles = None
        
        # ArUco setup for robot detection
        self.aruco_dict = cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_4X4_50)
        self.aruco_parameters = cv2.aruco.DetectorParameters()
        
    def compute_path(self, 
                    start: np.ndarray, 
                    goal: np.ndarray, 
                    obstacles: List[List[np.ndarray]]) -> Optional[List[np.ndarray]]:
        """
        Compute the shortest path from start to goal avoiding obstacles.
        
        Parameters
        ----------
        start : np.ndarray
            Start position [x, y]
        goal : np.ndarray
            Goal position [x, y]
        obstacles : list of lists of np.ndarray
            List of obstacle polygons, where each polygon is a list of vertices
            
        Returns
        -------
        list of np.ndarray or None
            Path as list of waypoint positions, or None if no path found
        """
        # Build nodes list: [start, goal, obstacle vertices...]
        nodes = [start, goal]
        for poly in obstacles:
            for v in poly:
                nodes.append(v)
        
        # Run A* pathfinding
        path_indices, path_length = a_star(nodes, obstacles, start_idx=0, goal_idx=1)
        
        if path_indices is None:
            return None
        
        # Convert indices to coordinates
        path = [nodes[i] for i in path_indices]
        
        return path
    
    def update_path(self, 
                   current_position: np.ndarray, 
                   goal: np.ndarray, 
                   obstacles: List[List[np.ndarray]]) -> bool:
        """
        Update the path from current position to goal.
        
        Parameters
        ----------
        current_position : np.ndarray
            Current robot position
        goal : np.ndarray
            Goal position
        obstacles : list of obstacle polygons
            
        Returns
        -------
        bool
            True if path was successfully computed, False otherwise
        """
        path = self.compute_path(current_position, goal, obstacles)
        
        if path is not None:
            self.current_path = path
            self.goal_position = goal
            self.obstacles = obstacles
            return True
        else:
            self.current_path = None
            return False
    
    def process_frame(self, 
                     frame: np.ndarray, 
                     obstacles: List[List[np.ndarray]], 
                     goal: np.ndarray,
                     force_replan: bool = False) -> Optional[dict]:
        """
        Process a camera frame and compute motion command.
        
        This is the main method to call in a control loop.
        
        Parameters
        ----------
        frame : np.ndarray
            Camera frame (BGR image)
        obstacles : list of obstacle polygons
            Current obstacles in the environment
        goal : np.ndarray
            Goal position [x, y]
        force_replan : bool
            If True, force replanning even if path exists
            
        Returns
        -------
        dict or None
            Motion command dictionary, or None if robot not detected or no path
            Command includes:
            - 'action': action type
            - 'left_speed': left wheel speed
            - 'right_speed': right wheel speed
            - 'angle_error': angle error
            - 'current_position': robot position
            - 'current_orientation': robot orientation
            - 'target_position': next waypoint
            - 'path_status': 'valid', 'replanned', or 'no_path'
        """
        # Detect robot position
        robot_state = detect_robot_position_and_orientation(
            frame, 
            self.aruco_dict, 
            self.aruco_parameters
        )
        
        if robot_state is None:
            return None
        
        current_position, current_orientation = robot_state
        
        # Check if we need to (re)plan the path
        need_replan = (
            force_replan or
            self.current_path is None or
            self.goal_position is None or
            not np.allclose(self.goal_position, goal)
        )
        
        if need_replan:
            success = self.update_path(current_position, goal, obstacles)
            if not success:
                return {
                    'action': 'no_path',
                    'left_speed': 0,
                    'right_speed': 0,
                    'angle_error': 0,
                    'current_position': current_position,
                    'current_orientation': current_orientation,
                    'target_position': None,
                    'path_status': 'no_path'
                }
            path_status = 'replanned'
        else:
            path_status = 'valid'
        
        # Get motion command
        command = motion_controller(
            frame,
            self.current_path,
            goal,
            self.aruco_dict,
            self.aruco_parameters
        )
        
        if command is None:
            return None
        
        # Add path status to command
        command['path_status'] = path_status
        
        return command
    
    def visualize_state(self, 
                       frame: np.ndarray, 
                       command: Optional[dict] = None) -> np.ndarray:
        """
        Visualize the current navigation state on the frame.
        
        Parameters
        ----------
        frame : np.ndarray
            Camera frame to draw on
        command : dict, optional
            Current motion command
            
        Returns
        -------
        np.ndarray
            Frame with visualization overlay
        """
        vis_frame = frame.copy()
        
        if command is None:
            cv2.putText(vis_frame, "Robot not detected", (10, 30),
                       cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
            return vis_frame
        
        # Draw current position
        if 'current_position' in command:
            pos = command['current_position'].astype(int)
            cv2.circle(vis_frame, tuple(pos), 10, (0, 255, 0), -1)
            
            # Draw orientation arrow
            if 'current_orientation' in command:
                angle_rad = np.radians(command['current_orientation'])
                arrow_len = 30
                end_x = int(pos[0] + arrow_len * np.cos(angle_rad))
                end_y = int(pos[1] + arrow_len * np.sin(angle_rad))
                cv2.arrowedLine(vis_frame, tuple(pos), (end_x, end_y),
                              (0, 255, 0), 2, tipLength=0.3)
        
        # Draw target position
        if 'target_position' in command and command['target_position'] is not None:
            target = command['target_position'].astype(int)
            cv2.circle(vis_frame, tuple(target), 8, (255, 0, 0), 2)
        
        # Draw path
        if self.current_path is not None and len(self.current_path) > 1:
            for i in range(len(self.current_path) - 1):
                pt1 = self.current_path[i].astype(int)
                pt2 = self.current_path[i + 1].astype(int)
                cv2.line(vis_frame, tuple(pt1), tuple(pt2), (255, 255, 0), 2)
        
        # Draw goal
        if self.goal_position is not None:
            goal_pt = self.goal_position.astype(int)
            cv2.circle(vis_frame, tuple(goal_pt), 15, (0, 0, 255), 3)
            cv2.circle(vis_frame, tuple(goal_pt), int(self.near_goal_radius), 
                      (0, 0, 255), 1)
        
        # Draw obstacles
        if self.obstacles is not None:
            for poly in self.obstacles:
                pts = np.array([p.astype(int) for p in poly])
                cv2.polylines(vis_frame, [pts], True, (128, 128, 128), 2)
        
        # Add text info
        action = command.get('action', 'unknown')
        angle_error = command.get('angle_error', 0)
        
        info_text = [
            f"Action: {action}",
            f"Angle error: {angle_error:.1f} deg",
            f"L/R speeds: {command.get('left_speed', 0):.0f}/{command.get('right_speed', 0):.0f}"
        ]
        
        y_offset = 30
        for i, text in enumerate(info_text):
            cv2.putText(vis_frame, text, (10, y_offset + i * 30),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
        
        return vis_frame


# Example usage and testing
if __name__ == "__main__":
    print("Robot Navigation Integration Module")
    print("=" * 50)
    
    # Create a simple test scenario
    navigator = RobotNavigator()
    
    # Define test obstacles
    quad1 = [
        np.array([200.0, 300.0]),
        np.array([200.0, 200.0]),
        np.array([300.0, 200.0]),
        np.array([300.0, 300.0]),
    ]
    
    quad2 = [
        np.array([400.0, 400.0]),
        np.array([550.0, 380.0]),
        np.array([580.0, 500.0]),
        np.array([420.0, 520.0]),
    ]
    
    obstacles = [quad1, quad2]
    
    # Test path computation
    start = np.array([50.0, 50.0])
    goal = np.array([750.0, 750.0])
    
    print(f"\nComputing path from {start} to {goal}...")
    path = navigator.compute_path(start, goal, obstacles)
    
    if path:
        print(f"Path found with {len(path)} waypoints:")
        for i, waypoint in enumerate(path):
            print(f"  {i}: {waypoint}")
    else:
        print("No path found!")
    
    print("\nRobot Navigator module loaded successfully!")
