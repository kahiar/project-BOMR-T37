# Mobile Robotics Project
## Functions of the robot
- Move from point A to point B and avoid obstacles during its movement

Workflow:
- Camera detects corners of the environment, flattens if needed. (Croping and perspective transform)
- Camera detects obstacles, finds vertices and puts them at a distance where the thymio doesn't touch the obstacle.
- Map the dots and use an algorithm to find the shortest paths connecting the dots.
- We send instructions to the robot in order for it to move (move with control or move turn type of movement)
- Tracking to correct robot's position
- Local avoidance (unexpected obstacles and return to initial trajectory)


Remarks:
- We need to know the direction of the robot at all times.
