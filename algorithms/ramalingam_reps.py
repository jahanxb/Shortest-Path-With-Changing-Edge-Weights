"""
algorithms/ramalingam_reps.py
==============================
Ramalingam-Reps Dynamic SSSP (RR-SSSP).

Gold standard for non-negative, fully-dynamic SSSP.  Handles both
weight increases and decreases by maintaining an explicit shortest-path
tree (parent dict) and repairing only the affected subtree.

Complexity (per update)
-----------------------
  Time : O(k log V)  where k = nodes whose distance actually changes.
         Best case: O(log V) — a single node.
         Worst case: O(V log V) — the source edge changes.
  Space: O(V + E)

Limitations
-----------
  - Does NOT support negative edge weights (Dijkstra-style relaxation
    inside the repair step requires non-negative weights).
  - Requires extension (Johnson-style potential reweighting) to handle
    negative weights.

Breaking cases
--------------
  1. Negative edge weight          → silently wrong results.
  2. Source edge weight increase   → k = V, degrades to O(V log V).
  3. All edges change simultaneously → k = V, no benefit over rerun.
"""

import heapq
from algorithms.graph import Graph, INF


class RamalingamReps:
    """
    Incremental SSSP via explicit shortest-path tree repair.
    Handles weight increase (subtree invalidation + recompute) and
    decrease (Dijkstra-style propagation of improvement).
    """
    name = "Ramalingam-Reps (RR-SSSP)"
    supports_negative = False

    def __init__(self, graph: Graph, source: int):
        self.graph = graph
        self.source = source
        self.dist: dict = {}
        self.parent: dict = {}
        self.nodes_visited: int = 0
        self.visited_nodes_list: list = []
        self._full_dijkstra()

    def _full_dijkstra(self) -> None:
        """Build initial shortest-path tree via Dijkstra."""
        dist = {i: INF for i in range(self.graph.n)}
        parent = {i: None for i in range(self.graph.n)}
        dist[self.source] = 0
        pq = [(0, self.source)]
        visited: set = set()
        self.visited_nodes_list = []

        while pq:
            d, u = heapq.heappop(pq)
            if u in visited:
                continue
            visited.add(u)
            self.visited_nodes_list.append(u)
            for v, w in self.graph.neighbors(u).items():
                if dist[u] + w < dist[v]:
                    dist[v] = dist[u] + w
                    parent[v] = u
                    heapq.heappush(pq, (dist[v], v))

        self.dist = dist
        self.parent = parent

    def update(self, u: int, v: int, w_new: float) -> int:
        w_old = self.graph.update_weight(u, v, w_new)
        self.nodes_visited = 0
        self._v_set = set()

        if w_new <= w_old:
            self._handle_decrease(u, v, w_new)
        else:
            self._handle_increase(u, v)

        self.visited_nodes_list = list(self._v_set)
        return self.nodes_visited

    def _handle_decrease(self, u: int, v: int, w_new: float) -> None:
        """Edge weight decreased — check if path through it improves dist[v]."""
        if self.dist[u] == INF:
            return
        candidate = self.dist[u] + w_new
        if candidate >= self.dist[v]:
            return

        self.dist[v] = candidate
        self.parent[v] = u
        pq = [(candidate, v)]

        while pq:
            d, x = heapq.heappop(pq)
            self.nodes_visited += 1
            self._v_set.add(x)
            if d > self.dist[x]:
                continue   # stale heap entry
            for y, w in self.graph.neighbors(x).items():
                new_d = self.dist[x] + w
                if new_d < self.dist[y]:
                    self.dist[y] = new_d
                    self.parent[y] = x
                    heapq.heappush(pq, (new_d, y))

    def _handle_increase(self, u: int, v: int) -> None:
        """
        Edge weight increased.  If v used this edge in its shortest path,
        the subtree rooted at v must be invalidated and recomputed.
        """
        if self.parent.get(v) != u:
            return   # v did not rely on this edge — no effect

        # Collect the affected subtree (v and all descendants in the SPT)
        subtree = []
        queue = [v]
        visited: set = set()
        while queue:
            node = queue.pop()
            if node in visited:
                continue
            visited.add(node)
            subtree.append(node)
            for y in self.graph.neighbors(node):
                if self.parent.get(y) == node:
                    queue.append(y)

        # Invalidate distances for all subtree nodes
        subtree_set = set(subtree)
        for node in subtree:
            self.dist[node] = INF
            self.parent[node] = None

        # Seed the priority queue with best edges entering the subtree
        # from nodes that are NOT in the subtree (already have valid dists)
        pq = []
        for node in subtree:
            self.nodes_visited += 1
            self._v_set.add(node)
            for pred, w in self.graph.predecessors(node).items():
                if pred not in subtree_set and self.dist[pred] != INF:
                    candidate = self.dist[pred] + w
                    if candidate < self.dist[node]:
                        self.dist[node] = candidate
                        self.parent[node] = pred
                        heapq.heappush(pq, (candidate, node))

        # Propagate improvements through the subtree
        while pq:
            d, x = heapq.heappop(pq)
            self._v_set.add(x)
            if d > self.dist[x]:
                continue
            for y, w in self.graph.neighbors(x).items():
                new_d = self.dist[x] + w
                if new_d < self.dist[y]:
                    self.dist[y] = new_d
                    self.parent[y] = x
                    heapq.heappush(pq, (new_d, y))
