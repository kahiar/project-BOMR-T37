# Robot Motion Control System

This module implements a complete motion control system for a differential drive robot navigating through obstacles to reach a goal position.

## Overview

The system consists of three main modules:

1. **shortest_path.py** - A* pathfinding with obstacle avoidance
2. **motion_control.py** - Robot detection, orientation tracking, and motion command generation
3. **robot_navigation.py** - Integration module that combines pathfinding and motion control

## Key Features

### Robot Detection and Orientation
- Uses ArUco markers for robot position and orientation detection
- Robot marker (ID 0) serves as both position indicator and orientation reference
- Supports real-time tracking from overhead camera feed

### Motion Control Strategy
The system implements intelligent turning behavior:

- **Large angle errors (> 90°)**: Robot turns in place before moving forward
- **Moderate angle errors (≤ 90°)**: Robot uses differential drive to turn while moving
- **Near goal**: Simplified control for final approach

### Path Following
- A* algorithm computes optimal path avoiding obstacles
- Dynamic waypoint tracking with automatic path updates
- Collision avoidance through enlarged obstacle boundaries

## Usage

### Basic Example

```python
from robot_navigation import RobotNavigator
import cv2
import numpy as np

# Initialize navigator
navigator = RobotNavigator()

# Define obstacles (as lists of vertex coordinates)
obstacles = [
    [np.array([200.0, 300.0]), np.array([200.0, 200.0]), 
     np.array([300.0, 200.0]), np.array([300.0, 300.0])],
    # ... more obstacles
]

# Set goal position
goal = np.array([750.0, 750.0])

# Main control loop
cap = cv2.VideoCapture(0)  # Your overhead camera

while True:
    ret, frame = cap.read()
    if not ret:
        break
    
    # Get motion command
    command = navigator.process_frame(frame, obstacles, goal)
    
    if command is None:
        print("Robot not detected!")
        continue
    
    # Check if goal reached
    if command['action'] == 'reached_goal':
        print("Goal reached!")
        # Stop robot and signal completion
        break
    
    # Send speeds to robot motors
    left_speed = command['left_speed']
    right_speed = command['right_speed']
    # send_to_robot(left_speed, right_speed)  # Your robot API
    
    # Optional: Visualize current state
    vis_frame = navigator.visualize_state(frame, command)
    cv2.imshow('Navigation', vis_frame)
    
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
```

### Running the Demo

```bash
python3 demo_motion_control.py
```

Note: The demo uses a simplified simulation. Real robot control requires:
- Actual camera feed with ArUco markers
- Robot hardware interface (e.g., Thymio API)
- Properly calibrated camera for accurate position detection

## Motion Control Parameters

You can adjust these constants in `motion_control.py`:

- `TURN_IN_PLACE_ANGLE_THRESHOLD`: Angle above which robot turns in place (default: 90°)
- `NEAR_GOAL_RADIUS`: Distance to goal for simplified final approach (default: 50 pixels)
- `ROBOT_SPEED_MAX`: Maximum forward speed (default: 100)
- `ROBOT_TURN_SPEED`: Speed for turning in place (default: 50)

## Robot Setup

### ArUco Marker Configuration

1. Place an ArUco marker (ID 0 from DICT_4X4_50) on top of the robot
2. Orient the marker so that its "forward" edge aligns with the robot's front
3. The marker should be clearly visible from the overhead camera

### Camera Setup

- Mount camera with bird's eye view of the workspace
- Ensure good lighting for reliable marker detection
- Calibrate camera if needed for accurate distance measurements

## Module Reference

### motion_control.py

Key functions:
- `detect_robot_position_and_orientation(frame)` - Detects robot using ArUco markers
- `compute_motion_command(position, orientation, target, goal)` - Generates motor commands
- `calculate_angle_difference(current, target)` - Computes shortest turning angle
- `is_near_goal(position, goal, radius)` - Checks goal proximity

### robot_navigation.py

Key class: `RobotNavigator`

Methods:
- `process_frame(frame, obstacles, goal)` - Main control loop function
- `compute_path(start, goal, obstacles)` - Calculates shortest path
- `visualize_state(frame, command)` - Draws navigation state on frame

### shortest_path.py

Functions:
- `a_star(nodes, obstacles, start_idx, goal_idx)` - A* pathfinding algorithm
- `build_visibility_graph(nodes, obstacles)` - Constructs visibility graph
- `visible(i, j, nodes, obstacles)` - Checks if two points have line of sight

## Motion Command Format

The `compute_motion_command()` function returns a dictionary with:

```python
{
    'action': str,           # 'turn_in_place', 'turn_and_move', 'move_forward', or 'reached_goal'
    'left_speed': float,     # Speed for left wheel (-100 to 100)
    'right_speed': float,    # Speed for right wheel (-100 to 100)
    'angle_error': float,    # Angle difference in degrees
    'current_position': np.ndarray,    # Robot position [x, y]
    'current_orientation': float,       # Robot orientation (0-360°)
    'target_position': np.ndarray       # Next waypoint [x, y]
}
```

## Coordinate System

- Origin (0, 0) is at top-left of camera frame
- X-axis points right
- Y-axis points down
- Orientation 0° points right (positive X direction)
- Counter-clockwise rotation increases orientation angle

## Integration with Existing Code

This motion control system is designed to work with your existing vision and obstacle detection:

1. Use your camera calibration and perspective transform
2. Detect obstacles and compute their enlarged boundaries
3. Identify goal position from ArUco marker
4. Pass obstacles and goal to `RobotNavigator.process_frame()`
5. Send resulting motor commands to your robot (e.g., via Thymio API)

## Troubleshooting

**Robot not detected:**
- Check ArUco marker is visible and well-lit
- Ensure marker ID is 0 from DICT_4X4_50 dictionary
- Verify marker isn't occluded or tilted excessively

**Robot turning wrong direction:**
- Verify differential drive motor mapping
- Check that left_speed/right_speed match your robot's convention
- Ensure coordinate system matches camera orientation

**Path not found:**
- Verify obstacles don't completely block the goal
- Check obstacle coordinates are in correct format
- Ensure start and goal positions are not inside obstacles

## Future Enhancements

Possible improvements:
- Adaptive speed control based on proximity to obstacles
- Smooth trajectory generation (e.g., Bezier curves)
- Predictive control for better path following
- Multi-robot coordination
- Dynamic obstacle avoidance

## Dependencies

```
numpy
opencv-python (cv2)
matplotlib (for visualization)
```

Install with:
```bash
pip install numpy opencv-python matplotlib
```
