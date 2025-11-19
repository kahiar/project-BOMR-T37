import numpy as np
import heapq
import matplotlib.pyplot as plt

#CALCULATES EDGES OF A GIVEN POLYGON
def polygon_edges(polygon): 
    """
    polygon: list of points [p0, p1, ..., p_{n-1}]
    returns list of (p_i, p_{i+1}) including last->first
    """
    edges = []
    n = len(polygon)
    for i in range(n):
        p1 = polygon[i]
        p2 = polygon[(i + 1) % n]  # wrap around
        edges.append((p1, p2))
    return edges

#TELLS US WHETHER r IS TO THE LEFT OR TO THE RIGHT OF SEGMENT pq
def orientation(p, q, r):
    """
    Returns:
        >0 if p->q->r is counter-clockwise (left turn)
        <0 if clockwise (right turn)
        0 if collinear
    """
    # Cross product of (q - p) and (r - p)
    val = (q[0] - p[0]) * (r[1] - p[1]) - (q[1] - p[1]) * (r[0] - p[0])
    if val > 0:
        return 1
    elif val < 0:
        return -1
    else:
        return 0

#Check if a point lies on a segment
def on_segment(p, q, r):
    """
    Given collinear points p, q, r:
    returns True if q lies on segment pr.
    """
    return (min(p[0], r[0]) <= q[0] <= max(p[0], r[0]) and
            min(p[1], r[1]) <= q[1] <= max(p[1], r[1]))


#Checks whether segments {p1,p2} and {q1,q2} intersect
def segments_intersect(p1, p2, q1, q2):
    """
    Returns True if segments p1-p2 and q1-q2 intersect (including touching).
    """
    o1 = orientation(p1, p2, q1)
    o2 = orientation(p1, p2, q2)
    o3 = orientation(q1, q2, p1)
    o4 = orientation(q1, q2, p2)

    # General case: segments straddle each other
    if o1 != o2 and o3 != o4:
        return True

    # Special cases: collinear and overlapping
    if o1 == 0 and on_segment(p1, q1, p2):
        return True
    if o2 == 0 and on_segment(p1, q2, p2):
        return True
    if o3 == 0 and on_segment(q1, p1, q2):
        return True
    if o4 == 0 and on_segment(q1, p2, q2):
        return True

    return False

def segment_intersects_any_obstacle(p1, p2, obstacles):
    """
    Returns True if segment p1-p2 intersects any obstacle edge.
    """
    for polygon in obstacles: #takes polygons 1 at a time 
        for q1, q2 in polygon_edges(polygon): #q1 and q2 represent the endpoints of one obstacle edge
            # Optional: skip if they share an endpoint (for visibility later)
            if np.allclose(p1, q1) or np.allclose(p1, q2) \
                or np.allclose(p2, q1) or np.allclose(p2, q2): #test to see if p1 or p2 is exactly on a vertex of the obstacle edge
                continue

            if segments_intersect(p1, p2, q1, q2):
                return True
    return False

def find_accessible_neighbours(current_position, end, obstacles):
    """
    Returns a vector with the indices of all accessible neighbours.

    The indices are with respect to the internal node list:
    nodes = [end] + [all obstacle vertices in order]
    """
    nodes = []

    # 1) Add goal point first
    nodes.append(end)

    # 2) Add all obstacle vertices
    for poly in obstacles:
        for v in poly:
            nodes.append(v)

    accessible_indices = []

    # 3) Check visibility from current_position to each node
    for idx, p in enumerate(nodes):

        # Skip if it's (numerically) the same point
        if np.allclose(p, current_position):
            continue

        # If the segment doesn't intersect any obstacle, it's a neighbour
        if not segment_intersects_any_obstacle(current_position, p, obstacles):
            accessible_indices.append(idx)

    return accessible_indices

def visible(i, j, nodes, obstacles):
    """
    Returns True if the segment between nodes[i] and nodes[j]
    is visible (i.e. does NOT intersect any obstacle edge).

    i, j: integer indices into the nodes list
    nodes: list of np.array([x, y])
    obstacles: list of polygons (each polygon = list of np.array([x,y]))
    """
    p1 = nodes[i]
    p2 = nodes[j]

    # --- 1) Forbid diagonals across the *same* polygon (non-adjacent vertices) ---
    for polygon in obstacles:
        idx1 = None
        idx2 = None
        n_poly = len(polygon)

        # Find p1 and p2 indices in this polygon (if they belong to it)
        for k, v in enumerate(polygon):
            if idx1 is None and np.allclose(v, p1):
                idx1 = k
            if idx2 is None and np.allclose(v, p2):
                idx2 = k

        if idx1 is not None and idx2 is not None:
            # Both endpoints are vertices of this polygon
            # Check adjacency (modulo wrap-around)
            diff = (idx1 - idx2) % n_poly
            if diff not in (1, n_poly - 1):
                # Non-adjacent vertices -> segment goes through the interior
                return False
            # If they are adjacent, that's the polygon edge itself; allowed.
            # We do NOT early-return True because it might still intersect other polygons.

    # --- 2) Usual visibility test against all obstacle edges ---
    for polygon in obstacles:
        n = len(polygon)

        for k in range(n):
            q1 = polygon[k]
            q2 = polygon[(k + 1) % n]  # wrap around to close loop

            # Allow touching at vertices (needed for visibility graphs):
            # we skip edges that share a vertex with the segment.
            if (np.allclose(p1, q1) or np.allclose(p1, q2) or
                np.allclose(p2, q1) or np.allclose(p2, q2)):
                continue

            # If the segment intersects an obstacle edge → not visible
            if segments_intersect(p1, p2, q1, q2):
                return False

    return True

def build_visibility_graph(nodes, obstacles):
    """
    Constructs a visibility graph.

    nodes: list of np.array([x,y])
        nodes[0] = start
        nodes[1] = goal
        nodes[2...] = obstacle vertices

    Returns:
        graph: dictionary where
                graph[i] = list of (j, distance_ij)
    """
    n = len(nodes)
    graph = {i: [] for i in range(n)}

    for i in range(n):
        for j in range(i + 1, n):

            if visible(i, j, nodes, obstacles):
                dist = np.linalg.norm(nodes[i] - nodes[j])

                graph[i].append((j, dist))
                graph[j].append((i, dist))

    return graph


def a_star(nodes, obstacles, start_idx=0, goal_idx=1):
    """
    A* search over the visibility graph implicitly defined by `visible()`.

    Parameters
    ----------
    nodes : list of np.array([x, y])
        nodes[start_idx] = start
        nodes[goal_idx]  = goal
        others are obstacle vertices.
    obstacles : list of polygons (each polygon = list of np.array([x, y]))
    start_idx : int
    goal_idx : int

    Returns
    -------
    path_indices : list[int] or None
        Sequence of node indices from start to goal (inclusive). None if no path.
    path_length : float
        Total length of the path (np.inf if no path).
    """

    def heuristic(i, j):
        # Euclidean distance heuristic (admissible)
        return np.linalg.norm(nodes[i] - nodes[j])

    n = len(nodes)

    # g_score: cost from start to node
    g_score = {i: np.inf for i in range(n)}
    g_score[start_idx] = 0.0

    # f_score: g + h
    f_score = {i: np.inf for i in range(n)}
    f_score[start_idx] = heuristic(start_idx, goal_idx)

    # To reconstruct path
    came_from = {}

    # Min-heap of (f_score, node_index)
    open_heap = []
    heapq.heappush(open_heap, (f_score[start_idx], start_idx))

    while open_heap:
        current_f, current = heapq.heappop(open_heap)

        # Skip outdated entry
        if current_f > f_score[current]:
            continue

        # Goal reached
        if current == goal_idx:
            path = [current]
            while current in came_from:
                current = came_from[current]
                path.append(current)
            path.reverse()
            return path, g_score[path[-1]]

        # ---- Generate neighbors using your `visible` function ----
        for neighbor in range(n):
            if neighbor == current:
                continue

            # Only consider edges that are visible (no obstacle intersection)
            if not visible(current, neighbor, nodes, obstacles):
                continue

            dist = np.linalg.norm(nodes[current] - nodes[neighbor])
            tentative_g = g_score[current] + dist

            if tentative_g < g_score[neighbor]:
                came_from[neighbor] = current
                g_score[neighbor] = tentative_g
                f_score[neighbor] = tentative_g + heuristic(neighbor, goal_idx)
                heapq.heappush(open_heap, (f_score[neighbor], neighbor))

    # No path found
    return None, np.inf



def main():

    # 1) PARAMETERS
    GRID_SIZE = 800
    NB_OBSTACLES = 2

    start = np.array([50, 50])       # Thymio current position
    goal  = np.array([750, 750])     # Target position from camera

    #IMPORTANT: THE OBSTACLES MUST BE THE ENLARGED OBSTACLES
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


    

    # Stack everything into a single tensor to simulate argument of the function
    obstacles = [quad1, quad2, quad3]

    # Build nodes
    nodes = [start, goal]
    for poly in obstacles:
        for v in poly:
            nodes.append(v)
    
    graph = build_visibility_graph(nodes, obstacles)
    
    # --- Run A* using visibility as neighbor criterion ---
    path_indices, path_len = a_star(nodes, obstacles, start_idx=0, goal_idx=1)
    print("Path indices:", path_indices)
    print("Path length:", path_len)

    if path_indices is not None:
        path_coords = [nodes[i] for i in path_indices]
        print("Path coordinates:", path_coords)

    # Test: is direct start->goal line blocked?
    # blocked = segment_intersects_any_obstacle(start, goal, obstacles)
    # print("Direct path blocked by an obstacle?", blocked)

    # current_position = start for the first step
    current_position = ([800,100])

    neigh_indices = find_accessible_neighbours(current_position, goal, obstacles)
    #print("Accessible neighbours indices:", neigh_indices)  

    # Call the plotting function
    # plot_environment(start, goal, obstacles, GRID_SIZE)
    
    if path_indices is not None:
        path_coords = [nodes[i] for i in path_indices]

        fig, ax = plt.subplots(figsize=(6, 6))
        # Plot obstacles
        for poly in obstacles:
            xs = [p[0] for p in poly] + [poly[0][0]]
            ys = [p[1] for p in poly] + [poly[0][1]]
            ax.fill(xs, ys, alpha=0.3, edgecolor='k')
            ax.scatter([p[0] for p in poly], [p[1] for p in poly], color='k')

        # Start & goal
        ax.scatter(start[0], start[1], marker='s', s=120, label='Start')
        ax.scatter(goal[0],  goal[1],  marker='X', s=120, label='Goal')

        # A* path
        xs = [p[0] for p in path_coords]
        ys = [p[1] for p in path_coords]
        ax.plot(xs, ys, linewidth=2, label='A* path')

        ax.set_xlim(0, GRID_SIZE)
        ax.set_ylim(0, GRID_SIZE)
        ax.set_aspect('equal', adjustable='box')
        ax.grid(True, linestyle='--', linewidth=0.5)
        ax.legend()
        ax.set_title("Shortest A* path with obstacles")
        plt.show()


if __name__ == "__main__":
    main()
