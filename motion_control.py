import numpy as np
import cv2
from typing import Tuple, Optional, List

# Constants for motion control
TURN_IN_PLACE_ANGLE_THRESHOLD = 90.0  # degrees
NEAR_GOAL_RADIUS = 50.0  # pixels - radius within which robot uses simplified control
ROBOT_SPEED_MAX = 100  # arbitrary units for maximum speed
ROBOT_TURN_SPEED = 50  # arbitrary units for turning speed


def detect_robot_position_and_orientation(frame: np.ndarray, 
                                         aruco_dict=None, 
                                         parameters=None) -> Optional[Tuple[np.ndarray, float]]:
    """
    Detect the robot's position and orientation using an arrow marker on top.
    
    The arrow marker should be implemented using ArUco markers or color detection.
    For this implementation, we'll use ArUco markers where the marker's orientation
    gives us the robot's heading.
    
    Parameters
    ----------
    frame : np.ndarray
        The camera frame (BGR image)
    aruco_dict : cv2.aruco.Dictionary, optional
        ArUco dictionary to use for detection
    parameters : cv2.aruco.DetectorParameters, optional
        ArUco detector parameters
        
    Returns
    -------
    tuple or None
        (position, orientation) where:
        - position is np.array([x, y]) in pixel coordinates
        - orientation is angle in degrees (0-360, where 0 is pointing right/east)
        Returns None if robot cannot be detected
    """
    if aruco_dict is None:
        aruco_dict = cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_4X4_50)
    
    if parameters is None:
        parameters = cv2.aruco.DetectorParameters()
    
    # Convert to grayscale for ArUco detection
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    
    # Detect ArUco markers
    detector = cv2.aruco.ArucoDetector(aruco_dict, parameters)
    corners, ids, rejected = detector.detectMarkers(gray)
    
    if ids is None or len(ids) == 0:
        return None
    
    # Find the robot marker (assuming robot uses marker ID 0)
    # You may need to adjust this based on your marker setup
    robot_marker_id = 0
    robot_idx = None
    
    for i, marker_id in enumerate(ids):
        if marker_id[0] == robot_marker_id:
            robot_idx = i
            break
    
    if robot_idx is None:
        return None
    
    # Get the corners of the robot marker
    marker_corners = corners[robot_idx][0]
    
    # Calculate center position (average of 4 corners)
    position = np.mean(marker_corners, axis=0)
    
    # Calculate orientation from marker corners
    # ArUco markers are detected with corners in a specific order
    # We can use the direction from bottom-left to bottom-right to get orientation
    # corners are ordered: top-left, top-right, bottom-right, bottom-left
    
    # Vector from bottom-left (corner 3) to bottom-right (corner 2)
    direction_vector = marker_corners[2] - marker_corners[3]
    
    # Calculate angle in degrees (0 = pointing right/east)
    orientation = np.degrees(np.arctan2(direction_vector[1], direction_vector[0]))
    
    # Normalize to 0-360 range
    if orientation < 0:
        orientation += 360
    
    return position, orientation


def calculate_angle_difference(current_orientation: float, 
                               target_orientation: float) -> float:
    """
    Calculate the shortest angle difference between current and target orientations.
    
    Parameters
    ----------
    current_orientation : float
        Current robot orientation in degrees (0-360)
    target_orientation : float
        Target orientation in degrees (0-360)
        
    Returns
    -------
    float
        Angle difference in degrees (-180 to 180)
        Positive means turn counter-clockwise, negative means turn clockwise
    """
    # Calculate raw difference
    diff = target_orientation - current_orientation
    
    # Normalize to -180 to 180 range (shortest path)
    while diff > 180:
        diff -= 360
    while diff < -180:
        diff += 360
    
    return diff


def calculate_target_orientation(current_position: np.ndarray, 
                                 target_position: np.ndarray) -> float:
    """
    Calculate the orientation needed to face the target position.
    
    Parameters
    ----------
    current_position : np.ndarray
        Current position [x, y]
    target_position : np.ndarray
        Target position [x, y]
        
    Returns
    -------
    float
        Target orientation in degrees (0-360)
    """
    # Calculate direction vector
    direction = target_position - current_position
    
    # Calculate angle in degrees
    angle = np.degrees(np.arctan2(direction[1], direction[0]))
    
    # Normalize to 0-360 range
    if angle < 0:
        angle += 360
    
    return angle


def is_near_goal(current_position: np.ndarray, 
                goal_position: np.ndarray, 
                radius: float = NEAR_GOAL_RADIUS) -> bool:
    """
    Check if the robot is within the goal radius.
    
    Parameters
    ----------
    current_position : np.ndarray
        Current position [x, y]
    goal_position : np.ndarray
        Goal position [x, y]
    radius : float
        Radius in pixels to consider "near" the goal
        
    Returns
    -------
    bool
        True if robot is within radius of goal
    """
    distance = np.linalg.norm(goal_position - current_position)
    return distance <= radius


def compute_motion_command(current_position: np.ndarray,
                          current_orientation: float,
                          target_position: np.ndarray,
                          goal_position: np.ndarray) -> dict:
    """
    Compute the motion command for the robot based on current state and target.
    
    This function implements the motion control logic:
    - If angle > 90°: turn in place
    - If angle ≤ 90°: move forward with differential speed
    - If near goal: simplified control for final approach
    
    Parameters
    ----------
    current_position : np.ndarray
        Current robot position [x, y]
    current_orientation : float
        Current robot orientation in degrees (0-360)
    target_position : np.ndarray
        Next waypoint position [x, y]
    goal_position : np.ndarray
        Final goal position [x, y]
        
    Returns
    -------
    dict
        Motion command with keys:
        - 'action': 'turn_in_place', 'move_forward', 'turn_and_move', or 'reached_goal'
        - 'left_speed': speed for left wheel (-100 to 100)
        - 'right_speed': speed for right wheel (-100 to 100)
        - 'angle_error': angle difference in degrees
    """
    # Check if we've reached the goal
    if is_near_goal(current_position, goal_position):
        # At goal - stop and signal completion
        return {
            'action': 'reached_goal',
            'left_speed': 0,
            'right_speed': 0,
            'angle_error': 0
        }
    
    # Calculate target orientation
    target_orientation = calculate_target_orientation(current_position, target_position)
    
    # Calculate angle error
    angle_error = calculate_angle_difference(current_orientation, target_orientation)
    
    # Determine action based on angle error
    abs_angle_error = abs(angle_error)
    
    if abs_angle_error > TURN_IN_PLACE_ANGLE_THRESHOLD:
        # Large angle error - turn in place first
        turn_direction = 1 if angle_error > 0 else -1
        
        return {
            'action': 'turn_in_place',
            'left_speed': turn_direction * ROBOT_TURN_SPEED,
            'right_speed': -turn_direction * ROBOT_TURN_SPEED,
            'angle_error': angle_error
        }
    
    elif abs_angle_error > 5:  # Small threshold to avoid jitter
        # Moderate angle error - use differential drive to turn while moving
        # Calculate speed reduction factor based on angle error
        # Larger errors cause more differential
        turn_factor = abs_angle_error / TURN_IN_PLACE_ANGLE_THRESHOLD
        
        # Base forward speed
        base_speed = ROBOT_SPEED_MAX
        
        # Calculate differential speeds
        if angle_error > 0:  # Turn left (counter-clockwise)
            left_speed = base_speed * (1 - turn_factor)
            right_speed = base_speed
        else:  # Turn right (clockwise)
            left_speed = base_speed
            right_speed = base_speed * (1 - turn_factor)
        
        return {
            'action': 'turn_and_move',
            'left_speed': left_speed,
            'right_speed': right_speed,
            'angle_error': angle_error
        }
    
    else:
        # Very small angle error - move straight forward
        return {
            'action': 'move_forward',
            'left_speed': ROBOT_SPEED_MAX,
            'right_speed': ROBOT_SPEED_MAX,
            'angle_error': angle_error
        }


def get_next_waypoint(current_position: np.ndarray, 
                     path: List[np.ndarray],
                     waypoint_reached_threshold: float = 30.0) -> Optional[np.ndarray]:
    """
    Get the next waypoint from the path based on current position.
    
    Parameters
    ----------
    current_position : np.ndarray
        Current robot position [x, y]
    path : list of np.ndarray
        List of waypoints forming the path
    waypoint_reached_threshold : float
        Distance threshold to consider a waypoint reached
        
    Returns
    -------
    np.ndarray or None
        Next waypoint to move towards, or None if path is empty
    """
    if not path or len(path) == 0:
        return None
    
    # Find the first waypoint that hasn't been reached yet
    for waypoint in path:
        distance = np.linalg.norm(waypoint - current_position)
        if distance > waypoint_reached_threshold:
            return waypoint
    
    # If all waypoints have been reached, return the last one (goal)
    return path[-1]


def motion_controller(frame: np.ndarray,
                     path: List[np.ndarray],
                     goal_position: np.ndarray,
                     aruco_dict=None,
                     parameters=None) -> Optional[dict]:
    """
    Main motion controller that integrates all components.
    
    This function:
    1. Detects robot position and orientation
    2. Determines next waypoint
    3. Computes motion command
    
    Parameters
    ----------
    frame : np.ndarray
        Current camera frame
    path : list of np.ndarray
        Path waypoints from current position to goal
    goal_position : np.ndarray
        Final goal position [x, y]
    aruco_dict : cv2.aruco.Dictionary, optional
        ArUco dictionary for robot detection
    parameters : cv2.aruco.DetectorParameters, optional
        ArUco detector parameters
        
    Returns
    -------
    dict or None
        Motion command dict, or None if robot cannot be detected
        Command dict contains:
        - 'action': type of action
        - 'left_speed': left wheel speed
        - 'right_speed': right wheel speed
        - 'angle_error': angle error in degrees
        - 'current_position': robot position
        - 'current_orientation': robot orientation
        - 'target_position': next waypoint
    """
    # Detect robot
    robot_state = detect_robot_position_and_orientation(frame, aruco_dict, parameters)
    
    if robot_state is None:
        return None
    
    current_position, current_orientation = robot_state
    
    # Get next waypoint
    next_waypoint = get_next_waypoint(current_position, path)
    
    if next_waypoint is None:
        # No path - stop
        return {
            'action': 'no_path',
            'left_speed': 0,
            'right_speed': 0,
            'angle_error': 0,
            'current_position': current_position,
            'current_orientation': current_orientation,
            'target_position': None
        }
    
    # Compute motion command
    command = compute_motion_command(
        current_position,
        current_orientation,
        next_waypoint,
        goal_position
    )
    
    # Add state information to command
    command['current_position'] = current_position
    command['current_orientation'] = current_orientation
    command['target_position'] = next_waypoint
    
    return command


# Example usage
if __name__ == "__main__":
    # Test angle calculation functions
    print("Testing angle calculations:")
    
    # Test 1: Simple angle difference
    current = 45.0
    target = 90.0
    diff = calculate_angle_difference(current, target)
    print(f"Current: {current}°, Target: {target}° => Difference: {diff}°")
    
    # Test 2: Angle wrapping
    current = 10.0
    target = 350.0
    diff = calculate_angle_difference(current, target)
    print(f"Current: {current}°, Target: {target}° => Difference: {diff}°")
    
    # Test 3: Target orientation calculation
    current_pos = np.array([100.0, 100.0])
    target_pos = np.array([200.0, 200.0])
    target_orient = calculate_target_orientation(current_pos, target_pos)
    print(f"From {current_pos} to {target_pos} => Orientation: {target_orient}°")
    
    # Test 4: Motion command computation
    current_pos = np.array([100.0, 100.0])
    current_orient = 0.0  # facing right
    target_pos = np.array([100.0, 200.0])  # directly above
    goal_pos = np.array([100.0, 300.0])
    
    command = compute_motion_command(current_pos, current_orient, target_pos, goal_pos)
    print(f"\nMotion command test:")
    print(f"  Current position: {current_pos}, orientation: {current_orient}°")
    print(f"  Target position: {target_pos}")
    print(f"  Command: {command}")
    
    # Test 5: Near goal detection
    is_near = is_near_goal(np.array([100.0, 100.0]), np.array([120.0, 120.0]))
    print(f"\nNear goal test: {is_near}")
    
    print("\nMotion control module loaded successfully!")
