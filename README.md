# Mobile Robotics Project

## Functions of the robot
- Move from point A to point B and avoid obstacles during its movement

## Workflow:
- Camera detects corners of the environment, flattens if needed. (Cropping and perspective transform)
- Camera detects obstacles, finds vertices and puts them at a distance where the thymio doesn't touch the obstacle.
- Map the dots and use an algorithm to find the shortest paths connecting the dots.
- **✅ Motion Control System** - We send instructions to the robot in order for it to move (differential drive with intelligent turning)
- Tracking to correct robot's position (using ArUco markers)
- Local avoidance (unexpected obstacles and return to initial trajectory)

## Project Structure

### Path Planning
- **shortest_path.py** - A* pathfinding algorithm with obstacle avoidance

### Motion Control (NEW ✅)
- **motion_control.py** - Robot detection, orientation tracking, and motion command generation
- **robot_navigation.py** - Integration of pathfinding and motion control
- **demo_motion_control.py** - Working demonstration and examples

### Documentation
- **QUICK_START_GUIDE.md** - Step-by-step guide for real robot integration
- **MOTION_CONTROL_README.md** - Complete API reference and usage
- **IMPLEMENTATION_SUMMARY.md** - Design decisions and technical details

## Quick Start

```python
from robot_navigation import RobotNavigator

# Initialize navigator
navigator = RobotNavigator()

# Process camera frame
command = navigator.process_frame(camera_frame, obstacles, goal)

# Send to robot
if command['action'] == 'reached_goal':
    robot.stop()
    robot.celebrate()
else:
    robot.set_motors(command['left_speed'], command['right_speed'])
```

See **QUICK_START_GUIDE.md** for complete integration instructions.

## Key Features

### Motion Control
- **ArUco marker detection** for robot position and orientation
- **Intelligent turning**: Turn in place for large angles, differential drive for moderate angles
- **Goal proximity detection** with simplified final approach
- **Configurable parameters** for tuning to your robot

### Path Planning
- A* algorithm for optimal path finding
- Visibility graph construction
- Obstacle avoidance with safety margins

## Requirements
- Python 3.x
- NumPy
- OpenCV (cv2)
- Matplotlib

```bash
pip install numpy opencv-python matplotlib
```

## Remarks:
- ✅ **Solved**: We track the robot's position and direction using ArUco markers
- The motion control system intelligently handles turning and movement
- Integration with existing path planning is complete
