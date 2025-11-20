# Quick Start Guide: Integrating Motion Control with Your Robot

This guide shows how to integrate the motion control system with your actual robot hardware.

## Prerequisites

1. **Hardware Setup**:
   - Overhead camera with bird's eye view of workspace
   - Robot (e.g., Thymio) with motors
   - ArUco marker (ID 0, DICT_4X4_50) attached to robot top
   - Good lighting for marker detection

2. **Software**:
   ```bash
   pip install numpy opencv-python matplotlib
   ```

3. **Existing Components** (you should already have):
   - Camera calibration and perspective transform
   - Obstacle detection from camera
   - Goal detection (ArUco marker)
   - Robot communication interface (e.g., Thymio API)

## Step-by-Step Integration

### Step 1: Print and Attach ArUco Marker

```python
import cv2
import numpy as np

# Generate and save ArUco marker
aruco_dict = cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_4X4_50)
marker_image = cv2.aruco.generateImageMarker(aruco_dict, 0, 200)
cv2.imwrite('robot_marker.png', marker_image)
```

- Print `robot_marker.png` on white paper
- Cut out and attach to top of robot
- Ensure marker's "up" direction aligns with robot's forward direction

### Step 2: Test Robot Detection

```python
import cv2
from motion_control import detect_robot_position_and_orientation

# Capture test frame
cap = cv2.VideoCapture(0)  # Your camera ID
ret, frame = cap.read()

# Test detection
result = detect_robot_position_and_orientation(frame)

if result is not None:
    position, orientation = result
    print(f"Robot detected at {position}, facing {orientation}°")
else:
    print("Robot not detected - check marker visibility")

cap.release()
```

### Step 3: Create Robot Interface

```python
# Example for Thymio robot
from thymiodirect import Connection, Thymio

class ThymioInterface:
    def __init__(self):
        self.connection = Connection.serial()  # or Connection.tcp()
        self.robot = Thymio(self.connection)
        
    def set_motor_speeds(self, left_speed, right_speed):
        """
        Convert motion control speeds to Thymio motor values.
        Motion control: -100 to 100
        Thymio: -500 to 500 (typical)
        """
        # Scale speeds
        scale_factor = 5.0
        left_motor = int(left_speed * scale_factor)
        right_motor = int(right_speed * scale_factor)
        
        # Clamp to Thymio limits
        left_motor = max(-500, min(500, left_motor))
        right_motor = max(-500, min(500, right_motor))
        
        # Send to robot
        self.robot['motor.left.target'] = left_motor
        self.robot['motor.right.target'] = right_motor
    
    def stop(self):
        self.robot['motor.left.target'] = 0
        self.robot['motor.right.target'] = 0
    
    def blink_lights(self):
        # Celebrate reaching goal
        for i in range(5):
            self.robot['leds.top'] = [32, 0, 0]  # Red
            time.sleep(0.3)
            self.robot['leds.top'] = [0, 32, 0]  # Green
            time.sleep(0.3)
```

### Step 4: Main Control Loop

```python
import cv2
import numpy as np
from robot_navigation import RobotNavigator

def main():
    # Initialize
    navigator = RobotNavigator()
    robot = ThymioInterface()  # Your robot interface
    cap = cv2.VideoCapture(0)  # Your camera
    
    # Get obstacles and goal from your vision system
    obstacles = detect_obstacles(cap)  # Your existing function
    goal = detect_goal_marker(cap)     # Your existing function
    
    print("Starting navigation...")
    
    try:
        while True:
            # Capture frame
            ret, frame = cap.read()
            if not ret:
                print("Failed to capture frame")
                break
            
            # Apply your calibration/perspective transform
            frame_transformed = apply_perspective_transform(frame)
            
            # Get motion command
            command = navigator.process_frame(
                frame_transformed, 
                obstacles, 
                goal
            )
            
            if command is None:
                print("Robot not detected!")
                robot.stop()
                continue
            
            # Check if goal reached
            if command['action'] == 'reached_goal':
                print("Goal reached!")
                robot.stop()
                robot.blink_lights()
                break
            
            # Send speeds to robot
            robot.set_motor_speeds(
                command['left_speed'],
                command['right_speed']
            )
            
            # Optional: Visualize
            vis_frame = navigator.visualize_state(frame, command)
            cv2.imshow('Navigation', vis_frame)
            
            # Exit on 'q'
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
    
    finally:
        robot.stop()
        cap.release()
        cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
```

### Step 5: Tuning Parameters

Adjust these in `motion_control.py` based on testing:

```python
# Turn threshold - when to turn in place vs. while moving
TURN_IN_PLACE_ANGLE_THRESHOLD = 90.0  # Try 60-120

# Goal proximity - when to use simplified control
NEAR_GOAL_RADIUS = 50.0  # Adjust based on robot size

# Speeds - adjust based on robot capabilities
ROBOT_SPEED_MAX = 100  # Scale factor for forward speed
ROBOT_TURN_SPEED = 50  # Scale factor for turning
```

In your robot interface, adjust `scale_factor`:
```python
# Increase for faster movement, decrease for more precision
scale_factor = 5.0  # Start with this, then tune
```

## Troubleshooting

### Robot Not Detected
- Check marker is visible and well-lit
- Verify marker ID is 0 from DICT_4X4_50
- Test with `detect_robot_position_and_orientation()` directly

### Robot Turning Wrong Direction
- Check left/right motor mapping in your interface
- Verify motor speed signs (positive = forward)
- Test with simple turn commands first

### Robot Not Following Path
- Verify obstacle coordinates are in camera pixel space
- Check perspective transform is applied correctly
- Ensure goal position is correct
- Test pathfinding independently with `shortest_path.py`

### Robot Oscillating
- Decrease `NEAR_GOAL_RADIUS` if oscillating around waypoints
- Increase `TURN_IN_PLACE_ANGLE_THRESHOLD` for smoother turns
- Reduce motor speed scale_factor for more precise control

## Testing Checklist

- [ ] ArUco marker detection works reliably
- [ ] Robot motors respond to speed commands
- [ ] Obstacle detection provides correct coordinates
- [ ] Goal marker is detected accurately
- [ ] Robot turns in correct direction
- [ ] Robot moves forward when aligned
- [ ] Robot stops at goal
- [ ] Visualization shows expected behavior

## Next Steps

Once basic navigation works:

1. **Add Local Avoidance**: Integrate with your local avoidance system
2. **Optimize Speeds**: Tune parameters for your robot's dynamics
3. **Handle Edge Cases**: Add recovery behaviors (stuck, lost, etc.)
4. **Improve Robustness**: Add error handling and timeouts
5. **Multi-Robot**: Extend to coordinate multiple robots

## Example Complete System

```python
# complete_navigation_system.py

import cv2
import numpy as np
from robot_navigation import RobotNavigator
# Your existing imports
from your_vision_module import (
    calibrate_camera,
    detect_obstacles,
    detect_goal,
    apply_perspective_transform
)
from your_robot_module import ThymioInterface

def main():
    # Setup
    cap = cv2.VideoCapture(0)
    robot = ThymioInterface()
    navigator = RobotNavigator()
    
    # Initial calibration
    print("Calibrating camera...")
    calibration_params = calibrate_camera(cap)
    
    # Detect environment
    print("Detecting obstacles and goal...")
    ret, frame = cap.read()
    frame_calib = apply_perspective_transform(frame, calibration_params)
    obstacles = detect_obstacles(frame_calib)
    goal = detect_goal(frame_calib)
    
    print(f"Found {len(obstacles)} obstacles")
    print(f"Goal at {goal}")
    
    # Navigation loop
    print("Starting navigation...")
    while True:
        ret, frame = cap.read()
        frame_calib = apply_perspective_transform(frame, calibration_params)
        
        command = navigator.process_frame(frame_calib, obstacles, goal)
        
        if command and command['action'] == 'reached_goal':
            robot.stop()
            robot.celebrate()
            break
        elif command:
            robot.set_motor_speeds(
                command['left_speed'],
                command['right_speed']
            )
        else:
            robot.stop()
    
    # Cleanup
    robot.stop()
    cap.release()

if __name__ == "__main__":
    main()
```

## Support

For detailed API documentation, see `MOTION_CONTROL_README.md`

For implementation details, see `IMPLEMENTATION_SUMMARY.md`

For code examples, see `demo_motion_control.py`
