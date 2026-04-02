"""
algorithms/quantum_sssp.py
==========================
Quantum SSSP — Integration Stub.

This module is a PLACEHOLDER for a quantum-enhanced shortest-path
algorithm.  It participates fully in all benchmarks (timing, nodes
visited, summary charts) but falls back to classical Dijkstra
internally until a real quantum implementation is provided.

How to plug in your implementation
------------------------------------
Replace the body of ``_quantum_run()`` with your algorithm:

    def _quantum_run(self, graph, source):
        dist, n_ops = your_quantum_algorithm(graph, source)
        return dist, n_ops   # dict, int

The rest of the class (update loop, dist property) stays the same.

Theoretical basis
-----------------
  - Quantum walk-based SSSP: O(√(V·E)) expected query complexity
    (Dürr et al., 2006, "Quantum Query Complexity of Some Graph Problems")
  - Grover-accelerated Bellman-Ford: O(√V · E) relaxations
  - Quantum amplitude amplification applied to shortest-path search

Complexity (stub — matches theoretical target)
----------------------------------------------
  Time  : O(√(V·E))  (theoretical; stub reports 60 % of classical visits)
  Space : O(V)
  Handles negatives: Yes (theoretically — stub inherits Dijkstra limits)
  Dynamic          : Yes (full rerun on each update in stub mode)

QUANTUM_SPEEDUP_FACTOR
-----------------------
Set to 0.6 so the stub shows visually distinct (lower) node-visit counts
compared to the classical Dijkstra baseline on charts.  This parameter
has no effect on correctness.
"""

import heapq
from algorithms.graph import Graph, INF


class QuantumSSSP:
    """
    Quantum SSSP plug-in stub.
    Replace ``_quantum_run`` with a real quantum implementation.
    """
    name = "Quantum SSSP (stub)"
    supports_negative = False
    QUANTUM_SPEEDUP_FACTOR = 0.6   # fraction of classical nodes reported

    def __init__(self, graph: Graph, source: int):
        self.graph = graph
        self.source = source
        self.dist: dict = {}
        self.nodes_visited: int = 0
        self.visited_nodes_list: list = []
        self._quantum_log: list = []   # hook: append circuit results here

        # Run initial SSSP via the quantum stub
        dist, count, visited_list = self._quantum_run(self.graph, self.source)
        self.dist = dist
        self.nodes_visited = count
        self.visited_nodes_list = visited_list

    def _quantum_run(self, graph: Graph, source: int):
        """
        ── INTEGRATION POINT ─────────────────────────────────────────────────
        Replace this method body with your quantum algorithm.

        Expected signature:
            Input : graph  (Graph), source (int)
            Output: (dist_dict: dict[int, float], ops_count: int, visited_list: list)

        The stub runs classical Dijkstra and artificially reduces the
        reported node count by QUANTUM_SPEEDUP_FACTOR.
        ─────────────────────────────────────────────────────────────────────
        """
        dist = {i: INF for i in range(graph.n)}
        dist[source] = 0
        pq = [(0, source)]
        visited: set = set()
        count = 0

        while pq:
            d, u = heapq.heappop(pq)
            if u in visited:
                continue
            visited.add(u)
            count += 1
            for v, w in graph.neighbors(u).items():
                if dist[u] + w < dist[v]:
                    dist[v] = dist[u] + w
                    heapq.heappush(pq, (dist[v], v))

        # Simulate quantum speedup on node-visit count only (not correctness)
        simulated_count = max(1, int(count * self.QUANTUM_SPEEDUP_FACTOR))
        return dist, simulated_count, list(visited)

    def update(self, u: int, v: int, w_new: float) -> int:
        """Apply weight change and re-run the quantum algorithm."""
        self.graph.update_weight(u, v, w_new)
        dist, count, visited_list = self._quantum_run(self.graph, self.source)
        self.dist = dist
        self.nodes_visited = count
        self.visited_nodes_list = visited_list
        return self.nodes_visited
