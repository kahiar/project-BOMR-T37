"""
Demo script for robot motion control system.

This script demonstrates how to use the motion control system with:
1. Path planning using A* algorithm
2. Robot detection and orientation tracking via ArUco markers
3. Motion control with differential drive

Usage:
    python3 demo_motion_control.py

Note: This is a simulation/demo. For real robot control, you would:
    - Replace the simulated frames with actual camera frames
    - Send the computed speeds to the actual robot motors via Thymio API
"""

import numpy as np
import cv2
import matplotlib.pyplot as plt
from robot_navigation import RobotNavigator
from motion_control import TURN_IN_PLACE_ANGLE_THRESHOLD


def create_test_frame_with_robot(position, orientation, obstacles, goal, frame_size=(800, 800)):
    """
    Create a simulated camera frame with ArUco marker at robot position.
    
    Parameters
    ----------
    position : np.ndarray
        Robot position [x, y]
    orientation : float
        Robot orientation in degrees
    obstacles : list of obstacle polygons
    goal : np.ndarray
        Goal position [x, y]
    frame_size : tuple
        Frame dimensions (width, height)
        
    Returns
    -------
    np.ndarray
        Simulated camera frame with ArUco marker
    """
    # Create white background
    frame = np.ones((frame_size[1], frame_size[0], 3), dtype=np.uint8) * 255
    
    # Draw obstacles
    for poly in obstacles:
        pts = np.array([p.astype(int) for p in poly])
        cv2.fillPoly(frame, [pts], (200, 200, 200))
        cv2.polylines(frame, [pts], True, (100, 100, 100), 2)
    
    # Draw goal
    goal_pt = goal.astype(int)
    cv2.circle(frame, tuple(goal_pt), 20, (0, 0, 255), -1)
    cv2.circle(frame, tuple(goal_pt), 25, (0, 0, 200), 2)
    
    # Generate ArUco marker for robot
    aruco_dict = cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_4X4_50)
    marker_size = 100  # Size of the ArUco marker image (smaller for better detection)
    marker_id = 0  # Robot marker ID
    marker_img = cv2.aruco.generateImageMarker(aruco_dict, marker_id, marker_size)
    
    # Convert marker to BGR
    marker_img_bgr = cv2.cvtColor(marker_img, cv2.COLOR_GRAY2BGR)
    
    # Rotate marker according to orientation
    # Note: negative orientation because image coordinates are flipped
    M = cv2.getRotationMatrix2D((marker_size/2, marker_size/2), -orientation, 1.0)
    marker_img_rotated = cv2.warpAffine(marker_img_bgr, M, (marker_size, marker_size), 
                                       borderMode=cv2.BORDER_CONSTANT, 
                                       borderValue=(255, 255, 255))
    
    # Calculate position to place marker (centered on robot position)
    x, y = position.astype(int)
    x1 = max(0, x - marker_size // 2)
    y1 = max(0, y - marker_size // 2)
    x2 = min(frame_size[0], x + marker_size // 2)
    y2 = min(frame_size[1], y + marker_size // 2)
    
    # Calculate corresponding region in marker image
    mx1 = marker_size // 2 - (x - x1)
    my1 = marker_size // 2 - (y - y1)
    mx2 = mx1 + (x2 - x1)
    my2 = my1 + (y2 - y1)
    
    # Place marker on frame
    if mx1 >= 0 and my1 >= 0 and mx2 <= marker_size and my2 <= marker_size:
        frame[y1:y2, x1:x2] = marker_img_rotated[my1:my2, mx1:mx2]
    
    return frame


def simulate_robot_movement(position, orientation, command, time_step=0.1):
    """
    Simulate robot movement based on command.
    
    Parameters
    ----------
    position : np.ndarray
        Current position [x, y]
    orientation : float
        Current orientation in degrees
    command : dict
        Motion command with left_speed and right_speed
    time_step : float
        Time step for simulation
        
    Returns
    -------
    tuple
        (new_position, new_orientation)
    """
    left_speed = command['left_speed']
    right_speed = command['right_speed']
    
    # Simple differential drive model
    # Average speed determines forward motion
    # Speed difference determines rotation
    
    avg_speed = (left_speed + right_speed) / 2.0
    
    # For differential drive: positive speed_diff (right > left) turns LEFT (counter-clockwise)
    # negative speed_diff (left > right) turns RIGHT (clockwise)
    speed_diff = left_speed - right_speed  # Fixed: was right - left
    
    # Convert orientation to radians
    orientation_rad = np.radians(orientation)
    
    # Update position based on average speed
    # Scale down the speed for visualization (speeds are in arbitrary units)
    speed_scale = 0.2  # Adjust this to control robot movement speed in simulation
    forward_dist = avg_speed * time_step * speed_scale
    new_position = position + forward_dist * np.array([
        np.cos(orientation_rad),
        np.sin(orientation_rad)
    ])
    
    # Update orientation based on speed difference
    # For differential drive: when right wheel is faster (left < right), robot turns LEFT (counter-clockwise)
    # This means orientation should increase
    # speed_diff = left - right is negative when turning left, so we negate it
    wheel_base = 100.0  # Simplified wheel base
    angular_velocity = -speed_diff / wheel_base  # Negated to get correct turn direction
    new_orientation = orientation + angular_velocity * time_step * 57.3  # Convert to degrees
    
    # Normalize orientation to 0-360
    new_orientation = new_orientation % 360
    
    return new_position, new_orientation


def run_simulation_demo():
    """
    Run a complete simulation demo of the robot navigation system.
    
    Note: This is a simplified simulation for demonstration purposes.
    In a real system, robot detection would use actual camera frames with real ArUco markers.
    """
    print("=" * 70)
    print("ROBOT MOTION CONTROL SIMULATION DEMO")
    print("=" * 70)
    print("\nNote: This is a simplified simulation using synthetic robot states.")
    print("Real robot control requires actual camera frames with ArUco markers.")
    
    # Set up environment
    print("\n1. Setting up environment...")
    
    # Define obstacles (enlarged for safety margin)
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
    
    quad3 = [
        np.array([200.0, 350.0]),
        np.array([300.0, 350.0]),
        np.array([420.0, 550.0]),
        np.array([250.0, 550.0]),
    ]
    
    obstacles = [quad1, quad2, quad3]
    
    # Initial robot state
    robot_position = np.array([100.0, 100.0])
    robot_orientation = 0.0  # degrees, 0 = pointing right
    
    # Goal position
    goal = np.array([750.0, 750.0])
    
    print(f"   Robot start position: {robot_position}")
    print(f"   Robot start orientation: {robot_orientation}°")
    print(f"   Goal position: {goal}")
    print(f"   Number of obstacles: {len(obstacles)}")
    
    # Create navigator
    print("\n2. Initializing robot navigator...")
    navigator = RobotNavigator()
    
    # Compute initial path
    print("\n3. Computing initial path...")
    path = navigator.compute_path(robot_position, goal, obstacles)
    
    if path is None:
        print("   ERROR: No path found!")
        return
    
    navigator.current_path = path
    navigator.goal_position = goal
    navigator.obstacles = obstacles
    
    print(f"   Path computed with {len(path)} waypoints:")
    for i, wp in enumerate(path):
        print(f"      Waypoint {i}: {wp}")
    
    # Simulation loop (simplified - without ArUco detection)
    print("\n4. Running simulation...")
    print(f"   Turn-in-place threshold: {TURN_IN_PLACE_ANGLE_THRESHOLD}°")
    print()
    
    max_iterations = 500
    iteration = 0
    reached_goal = False
    
    # Store trajectory for visualization
    trajectory = [robot_position.copy()]
    
    while iteration < max_iterations and not reached_goal:
        # Get next waypoint
        from motion_control import get_next_waypoint, compute_motion_command, is_near_goal
        
        next_waypoint = get_next_waypoint(robot_position, path)
        
        if next_waypoint is None:
            print(f"   Iteration {iteration}: No more waypoints!")
            break
        
        # Check if reached goal
        if is_near_goal(robot_position, goal):
            print(f"\n   SUCCESS! Goal reached at iteration {iteration}")
            print(f"   Final position: {robot_position}")
            print(f"   Final orientation: {robot_orientation:.1f}°")
            reached_goal = True
            break
        
        # Compute motion command
        command = compute_motion_command(
            robot_position,
            robot_orientation,
            next_waypoint,
            goal
        )
        
        action = command['action']
        angle_error = command['angle_error']
        
        # Print status every 25 iterations
        if iteration % 25 == 0:
            print(f"   Iteration {iteration:3d}: Action={action:15s}, "
                  f"Angle error={angle_error:6.1f}°, "
                  f"Position={robot_position.astype(int)}")
        
        # Simulate robot movement
        robot_position, robot_orientation = simulate_robot_movement(
            robot_position, 
            robot_orientation, 
            command,
            time_step=0.5  # Larger time step since we're not using visual detection
        )
        
        # Store trajectory
        trajectory.append(robot_position.copy())
        
        iteration += 1
    
    if not reached_goal:
        print(f"\n   Simulation stopped at iteration {iteration}")
        distance_to_goal = np.linalg.norm(goal - robot_position)
        print(f"   Distance to goal: {distance_to_goal:.1f} pixels")
        if distance_to_goal < 100:
            print(f"   (Very close to goal!)")
    
    # Visualize final result
    print("\n5. Generating visualization...")
    visualize_simulation_result(obstacles, goal, path, trajectory, robot_position, robot_orientation)
    
    print("\n" + "=" * 70)
    print("SIMULATION COMPLETE")
    print("=" * 70)


def visualize_simulation_result(obstacles, goal, planned_path, actual_trajectory, 
                                final_position, final_orientation):
    """
    Visualize the simulation results.
    """
    fig, ax = plt.subplots(figsize=(10, 10))
    
    # Draw obstacles
    for poly in obstacles:
        xs = [p[0] for p in poly] + [poly[0][0]]
        ys = [p[1] for p in poly] + [poly[0][1]]
        ax.fill(xs, ys, alpha=0.3, color='gray', edgecolor='black', linewidth=2)
    
    # Draw planned path
    if planned_path:
        path_xs = [p[0] for p in planned_path]
        path_ys = [p[1] for p in planned_path]
        ax.plot(path_xs, path_ys, 'b--', linewidth=2, label='Planned path', alpha=0.7)
        ax.scatter(path_xs, path_ys, c='blue', s=50, zorder=5, alpha=0.7)
    
    # Draw actual trajectory
    traj_xs = [p[0] for p in actual_trajectory]
    traj_ys = [p[1] for p in actual_trajectory]
    ax.plot(traj_xs, traj_ys, 'g-', linewidth=2, label='Actual trajectory')
    
    # Draw start and goal
    start = actual_trajectory[0]
    ax.scatter(start[0], start[1], c='green', s=200, marker='s', 
              label='Start', zorder=10, edgecolors='black', linewidths=2)
    ax.scatter(goal[0], goal[1], c='red', s=200, marker='X', 
              label='Goal', zorder=10, edgecolors='black', linewidths=2)
    
    # Draw final robot position and orientation
    ax.scatter(final_position[0], final_position[1], c='orange', s=150, 
              marker='o', label='Final position', zorder=10)
    
    # Draw orientation arrow
    arrow_length = 50
    angle_rad = np.radians(final_orientation)
    dx = arrow_length * np.cos(angle_rad)
    dy = arrow_length * np.sin(angle_rad)
    ax.arrow(final_position[0], final_position[1], dx, dy,
            head_width=20, head_length=15, fc='orange', ec='orange', linewidth=2)
    
    ax.set_xlim(0, 800)
    ax.set_ylim(0, 800)
    ax.set_aspect('equal')
    ax.grid(True, alpha=0.3)
    ax.legend(loc='upper left', fontsize=10)
    ax.set_title('Robot Navigation Simulation Result', fontsize=14, fontweight='bold')
    ax.set_xlabel('X (pixels)', fontsize=12)
    ax.set_ylabel('Y (pixels)', fontsize=12)
    
    plt.tight_layout()
    plt.savefig('/home/runner/work/project-BOMR-T37/project-BOMR-T37/simulation_result.png', dpi=150)
    print(f"   Visualization saved to: simulation_result.png")
    
    # Don't show in headless environment
    # plt.show()


def print_usage_example():
    """
    Print example code for using the motion control system.
    """
    print("\n" + "=" * 70)
    print("USAGE EXAMPLE - Integrating with Real Robot")
    print("=" * 70)
    
    example_code = """
# 1. Import required modules
from robot_navigation import RobotNavigator
import cv2
import numpy as np

# 2. Initialize navigator
navigator = RobotNavigator()

# 3. Set up camera
cap = cv2.VideoCapture(0)  # Use your camera

# 4. Define obstacles (detected from camera or predefined)
obstacles = [quad1, quad2, quad3]  # Your obstacle polygons
goal = np.array([750.0, 750.0])   # Your goal position

# 5. Main control loop
while True:
    # Capture frame
    ret, frame = cap.read()
    if not ret:
        break
    
    # Get motion command
    command = navigator.process_frame(frame, obstacles, goal)
    
    if command is None:
        print("Robot not detected!")
        continue
    
    # Send command to robot (Thymio API)
    if command['action'] == 'reached_goal':
        # Stop and blink lights
        set_motor_speeds(0, 0)
        blink_lights()
        break
    else:
        # Send speeds to motors
        left_speed = command['left_speed']
        right_speed = command['right_speed']
        set_motor_speeds(left_speed, right_speed)
    
    # Visualize (optional)
    vis_frame = navigator.visualize_state(frame, command)
    cv2.imshow('Robot Navigation', vis_frame)
    
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
"""
    
    print(example_code)
    print("=" * 70)


if __name__ == "__main__":
    # Run simulation demo
    run_simulation_demo()
    
    # Print usage example
    print_usage_example()
    
    print("\nDemo complete! Check simulation_result.png for visualization.")
