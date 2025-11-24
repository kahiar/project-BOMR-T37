import numpy as np
import networkx as nx
from scipy.spatial import distance
import heapq


class PathPlanner:
    """A* path planning using obstacle vertices"""

    def __init__(self):
        self.graph = None

    def compute_path(self, start, goal, obstacles):
        """
        Args:
            start: np.array(x, y)
            goal: np.array(x, y)
            obstacles: list of obstacle polygons [np.array([[x1,y1], [x2,y2], ...]), ...]

        Returns:
            list: Waypoints [(x1,y1), (x2,y2), ...] including start and goal,
                  or None if no path exists
        """
        nodes = [start, goal] + [v for poly in obstacles for v in poly]
        self.graph = self.build_visibility_graph(nodes, obstacles)

        path_idx, path_len = self.a_star(nodes, obstacles)
        print("Path indices:", path_idx)
        print("Path length:", path_len)

        if path_idx is not None:
            path_cords = [nodes[i] for i in path_idx]
            print("Path coordinates:", path_cords)
        else:
            path_cords = None

        return path_cords


    def _polygon_edges(self, polygon):
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

    def _orientation(self, p, q, r):
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

    def _on_segment(self, p, q, r):
        """
            Given collinear points p, q, r:
            returns True if q lies on segment pr.
            """
        return (min(p[0], r[0]) <= q[0] <= max(p[0], r[0]) and
                min(p[1], r[1]) <= q[1] <= max(p[1], r[1]))

    def _segments_intersect(self, p1, p2, q1, q2):
        """
        Returns True if segments p1-p2 and q1-q2 intersect (including touching).
        """
        o1 = self._orientation(p1, p2, q1)
        o2 = self._orientation(p1, p2, q2)
        o3 = self._orientation(q1, q2, p1)
        o4 = self._orientation(q1, q2, p2)

        # General case: segments straddle each other
        if o1 != o2 and o3 != o4:
            return True

        # Special cases: collinear and overlapping
        if o1 == 0 and self._on_segment(p1, q1, p2):
            return True
        if o2 == 0 and self._on_segment(p1, q2, p2):
            return True
        if o3 == 0 and self._on_segment(q1, p1, q2):
            return True
        if o4 == 0 and self._on_segment(q1, p2, q2):
            return True

        return False

    def _visible(self, i, j, nodes, obstacles):
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
                if self._segments_intersect(p1, p2, q1, q2):
                    return False

        return True  # The segment is obstacle-free

    def build_visibility_graph(self, nodes, obstacles):
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

                if self._visible(i, j, nodes, obstacles):
                    dist = np.linalg.norm(nodes[i] - nodes[j])

                    graph[i].append((j, dist))
                    graph[j].append((i, dist))

        return graph

    def a_star(self, nodes, obstacles, start_idx=0, goal_idx=1):
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
                if not self._visible(current, neighbor, nodes, obstacles):
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