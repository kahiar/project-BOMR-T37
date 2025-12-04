import cv2
import numpy as np
import matplotlib.pyplot as plt
from modified_astar import PathPlanner
from visualizer import Visualizer

def filter_color(image):
    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)

    LOWER_BLUE = np.array([90, 50, 50])
    UPPER_BLUE = np.array([130, 255, 255])

    mask = cv2.inRange(hsv, LOWER_BLUE, UPPER_BLUE)

    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, np.ones((5,5),np.uint8))
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, np.ones((5,5),np.uint8))

    return mask

def process_image(mask):
    blurred = cv2.GaussianBlur(mask, (3, 3), 0)
    edges = cv2.Canny(blurred, 50, 150, 7, L2gradient=True)
    dilated_edges = cv2.dilate(edges, kernel=np.ones((5, 5), np.uint8), iterations=1)
    return dilated_edges

def scale_contour(contour, scale=2):
    M = cv2.moments(contour)
    if M['m00'] == 0:
        return contour
    
    cx = int(M['m10'] / M['m00'])
    cy = int(M['m01'] / M['m00'])
    center = np.array([cx, cy])
    
    scaled = (contour - center) * scale + center
    return scaled.astype(np.int32)
    
def detect_contours(dilated_edges):
    contours, _ = cv2.findContours(dilated_edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    scaled_contours = []
    all_vertices = []

    for cnt in contours:
        approx = cv2.approxPolyDP(cnt, 0.02 * cv2.arcLength(cnt, True), True)
        if len(approx)  != 4:
            continue
        scaled = scale_contour(approx, scale=2)
        scaled_contours.append(scaled)
        vertices = scaled.reshape(-1, 2)
        all_vertices.append(vertices)

    return scaled_contours, all_vertices

def draw_contours(image, scaled_contours, all_vertices):
    output = image.copy()
    for vertices in all_vertices:
        for (x, y) in vertices:
            cv2.circle(output, (x, y), 6, (0, 0, 255), -1)

    cv2.drawContours(output, scaled_contours, -1, (0, 0, 255), 2)

    return output


def visualize_visibility_graph(start, goal, obstacles, image_path, show_path=False):
    """
    Visualize the visibility graph on a calibrated frame image.
    
    Args:
        start: np.array [x, y] start position
        goal: np.array [x, y] goal position
        obstacles: list of np.array polygons
        image_path: str, path to the background image
        show_path: bool, if True, compute and draw the shortest path
    
    Returns:
        matplotlib figure object
    """
    
    # Build nodes list
    nodes = [start, goal]
    for poly in obstacles:
        for v in poly:
            nodes.append(np.array(v))
    
    # Create planner and build visibility graph
    planner = PathPlanner()
    planner.graph = planner._build_visibility_graph(nodes, obstacles)
    
    # Load the calibrated frame image
    frame = cv2.imread(image_path)
    
    # Draw obstacles on the frame
    vis = Visualizer()
    frame = vis.draw_obstacles(frame, obstacles)
    
    # Draw all visibility graph edges
    for i, connections in planner.graph.items():
        for j, dist in connections:
            if i < j:  # Draw each edge only once
                pt1 = (int(nodes[i][0]), int(nodes[i][1]))
                pt2 = (int(nodes[j][0]), int(nodes[j][1]))
                cv2.line(frame, pt1, pt2, (0, 0, 0), 2)  # Black lines, thickness 2
    
    # Draw all nodes
    for i, node in enumerate(nodes):
        if i == 0:  # Start
            cv2.circle(frame, (int(node[0]), int(node[1])), 8, (255, 0, 0), -1)  # Blue
        elif i == 1:  # Goal
            cv2.circle(frame, (int(node[0]), int(node[1])), 8, (0, 255, 0), -1)  # Green
        else:  # Obstacle vertices
            cv2.circle(frame, (int(node[0]), int(node[1])), 5, (255, 255, 255), -1)  # White
    
    # Optionally draw the shortest path
    if show_path:
        path = planner.compute_path(start, goal, obstacles)
        if path is not None:
            for i in range(len(path) - 1):
                pt1 = (int(path[i][0]), int(path[i][1]))
                pt2 = (int(path[i+1][0]), int(path[i+1][1]))
                cv2.line(frame, pt1, pt2, (255, 200, 100), 3)  # Light blue line
    
    # Display the result
    fig = plt.figure(figsize=(9, 6))
    plt.imshow(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
    title = 'Shortest Path over Visibility Graph' if show_path else 'Visibility Graph'
    plt.title(title)
    plt.axis('off')
    plt.tight_layout()
    
    return fig

