import numpy as np
import heapq


class PathPlanner:
    """A* path planning using visibility graph on obstacle vertices."""

    def __init__(self):
        """Initialize path planner."""
        self.graph = None

    def compute_path(self, start, goal, obstacles):
        """
        Compute shortest path from start to goal avoiding obstacles.

        Args:
            start: np.array [x, y] start position in pixels
            goal: np.array [x, y] goal position in pixels
            obstacles: list of np.array polygons, each with shape (N, 2)

        Returns:
            list: Waypoints as list of np.array [x, y], or None if no path exists
        """
        nodes = [np.array(start), np.array(goal)]
        for poly in obstacles:
            for v in poly:
                nodes.append(np.array(v))

        self.graph = self._build_visibility_graph(nodes, obstacles)
        path_indices, _ = self._a_star(nodes, obstacles)

        if path_indices is None:
            return None
        return [nodes[i] for i in path_indices]

    def _build_visibility_graph(self, nodes, obstacles):
        """
        Build graph connecting mutually visible nodes.

        Args:
            nodes: list of np.array [x, y] positions
            obstacles: list of obstacle polygons

        Returns:
            dict: Adjacency list {node_idx: [(neighbor_idx, distance), ...]}
        """
        n = len(nodes)
        graph = {i: [] for i in range(n)}

        for i in range(n):
            for j in range(i + 1, n):
                if self._is_visible(i, j, nodes, obstacles):
                    dist = np.linalg.norm(nodes[i] - nodes[j])
                    graph[i].append((j, dist))
                    graph[j].append((i, dist))

        return graph

    def _is_visible(self, i, j, nodes, obstacles):
        """
        Check if segment between nodes[i] and nodes[j] is obstacle-free.

        Args:
            i: int, index of first node
            j: int, index of second node
            nodes: list of np.array positions
            obstacles: list of obstacle polygons

        Returns:
            bool: True if nodes can see each other
        """
        p1, p2 = nodes[i], nodes[j]

        # Check if segment crosses inside same polygon (non-adjacent vertices)
        for polygon in obstacles:
            idx1 = idx2 = None
            n_poly = len(polygon)

            for k, v in enumerate(polygon):
                if idx1 is None and np.allclose(v, p1):
                    idx1 = k
                if idx2 is None and np.allclose(v, p2):
                    idx2 = k

            if idx1 is not None and idx2 is not None:
                diff = (idx1 - idx2) % n_poly
                if diff not in (1, n_poly - 1):
                    return False

        # Check intersection with all obstacle edges
        for polygon in obstacles:
            n = len(polygon)
            for k in range(n):
                q1 = polygon[k]
                q2 = polygon[(k + 1) % n]

                # Skip edges sharing a vertex
                if (np.allclose(p1, q1) or np.allclose(p1, q2) or
                        np.allclose(p2, q1) or np.allclose(p2, q2)):
                    continue

                if self._segments_intersect(p1, p2, q1, q2):
                    return False

        return True

    def _segments_intersect(self, p1, p2, q1, q2):
        """
        Check if line segments p1-p2 and q1-q2 intersect.

        Args:
            p1, p2: np.array endpoints of first segment
            q1, q2: np.array endpoints of second segment

        Returns:
            bool: True if segments intersect
        """

        def orientation(a, b, c):
            val = (b[0] - a[0]) * (c[1] - a[1]) - (b[1] - a[1]) * (c[0] - a[0])
            return 1 if val > 0 else (-1 if val < 0 else 0)

        def on_segment(a, b, c):
            return (min(a[0], c[0]) <= b[0] <= max(a[0], c[0]) and
                    min(a[1], c[1]) <= b[1] <= max(a[1], c[1]))

        o1 = orientation(p1, p2, q1)
        o2 = orientation(p1, p2, q2)
        o3 = orientation(q1, q2, p1)
        o4 = orientation(q1, q2, p2)

        if o1 != o2 and o3 != o4:
            return True

        if o1 == 0 and on_segment(p1, q1, p2):
            return True
        if o2 == 0 and on_segment(p1, q2, p2):
            return True
        if o3 == 0 and on_segment(q1, p1, q2):
            return True
        if o4 == 0 and on_segment(q1, p2, q2):
            return True

        return False

    def _a_star(self, nodes, obstacles, start_idx=0, goal_idx=1):
        """
        A* search on visibility graph.

        Args:
            nodes: list of np.array positions
            obstacles: list of obstacle polygons
            start_idx: int, index of start node (default 0)
            goal_idx: int, index of goal node (default 1)

        Returns:
            tuple: (path_indices, path_length) where path_indices is list of
                   node indices from start to goal, or (None, inf) if no path
        """
        n = len(nodes)
        g_score = {i: np.inf for i in range(n)}
        g_score[start_idx] = 0.0

        f_score = {i: np.inf for i in range(n)}
        f_score[start_idx] = np.linalg.norm(nodes[start_idx] - nodes[goal_idx])

        came_from = {}
        open_heap = [(f_score[start_idx], start_idx)]
        closed_set = set()

        while open_heap:
            current_f, current = heapq.heappop(open_heap)

            # Skip outdated heap entries
            if current_f > f_score[current]:
                continue

            # Skip already expanded nodes
            if current in closed_set:
                continue
            closed_set.add(current)

            if current == goal_idx:
                path = [current]
                while current in came_from:
                    current = came_from[current]
                    path.append(current)
                return path[::-1], g_score[goal_idx]

            # Use pre-built visibility graph instead of rechecking visibility
            for neighbor, dist in self.graph[current]:
                # Skip already expanded nodes
                if neighbor in closed_set:
                    continue

                tentative_g = g_score[current] + dist

                if tentative_g < g_score[neighbor]:
                    came_from[neighbor] = current
                    g_score[neighbor] = tentative_g
                    h = np.linalg.norm(nodes[neighbor] - nodes[goal_idx])
                    f_score[neighbor] = tentative_g + h
                    heapq.heappush(open_heap, (f_score[neighbor], neighbor))

        return None, np.inf