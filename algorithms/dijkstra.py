"""
algorithms/dijkstra.py
======================
Dijkstra Full Rerun — baseline algorithm.

After every edge-weight change the entire SSSP is recomputed from
scratch using a binary min-heap (heapq).

Complexity (per update)
-----------------------
  Time : O((V + E) log V)
  Space: O(V)

Limitations
-----------
  - Does NOT support negative edge weights.
  - Does NOT support dynamic/incremental updates (always full rerun).

Breaking cases
--------------
  - Any negative edge weight produces wrong results silently.
  - Dynamic weight changes cannot be handled without a full rerun,
    making it O((V+E) log V) per change regardless of how few nodes
    are actually affected.
"""

import heapq
from algorithms.graph import Graph, INF


class DijkstraRerun:
    """
    Baseline: rerun Dijkstra from scratch after every weight change.
    Only correct on graphs with non-negative edge weights.
    """
    name = "Dijkstra Full Rerun"
    supports_negative = False

    def __init__(self, graph: Graph, source: int):
        self.graph = graph
        self.source = source
        self.dist: dict = {}
        self.nodes_visited: int = 0
        self._run()

    def _run(self) -> None:
        """Full Dijkstra from source."""
        self.nodes_visited = 0
        dist = {i: INF for i in range(self.graph.n)}
        dist[self.source] = 0
        pq = [(0, self.source)]
        visited: set = set()

        while pq:
            d, u = heapq.heappop(pq)
            if u in visited:
                continue
            visited.add(u)
            self.nodes_visited += 1
            for v, w in self.graph.neighbors(u).items():
                if dist[u] + w < dist[v]:
                    dist[v] = dist[u] + w
                    heapq.heappush(pq, (dist[v], v))

        self.dist = dist

    def update(self, u: int, v: int, w_new: float) -> int:
        """Apply weight change then rerun fully. Returns nodes visited."""
        self.graph.update_weight(u, v, w_new)
        self._run()
        return self.nodes_visited
