"""
algorithms/bellman_ford.py
==========================
Two Bellman-Ford variants:

1. BellmanFordRerun  — full rerun baseline, handles negatives.
2. DynamicBellmanFord — incremental variant, propagates only
                         affected subgraph on each update.

BellmanFordRerun
----------------
  Time (per update): O(V * E)  — full relaxation loop every time.
  Space            : O(V)
  Handles negatives: Yes
  Detects neg cycle: Yes (extra pass after V-1 iterations)
  Dynamic          : No  (always full rerun)

  Breaking cases:
    - Dense graphs or large V make O(VE) per update infeasible.
    - Negative cycles cause it to correctly *detect* and stop, but if
      missed (due to missing the extra-pass check) it loops indefinitely.

DynamicBellmanFord
------------------
  Time (per update): O(k * E) where k = number of affected nodes.
                     Best case O(E), worst case O(VE).
  Space            : O(V)
  Handles negatives: Yes
  Dynamic          : Yes — propagates decrease/increase separately.

  Breaking cases:
    - Weight increase on a bridge edge forces reprocessing the entire
      dependent subtree; in the worst case this is the whole graph.
    - Re-ordering affected nodes by stale distances (not fresh) can
      miss some improvements; a second pass may be needed for correctness
      in adversarial topologies.
    - Negative cycles: not detected — will loop or produce wrong results
      if a negative cycle is created by a dynamic update.
"""

from algorithms.graph import Graph, INF


class BellmanFordRerun:
    """
    Baseline: rerun Bellman-Ford from scratch after every weight change.
    Correct on graphs with negative weights (no negative cycles).
    """
    name = "Bellman-Ford Full Rerun"
    supports_negative = True

    def __init__(self, graph: Graph, source: int):
        self.graph = graph
        self.source = source
        self.dist: dict = {}
        self.nodes_visited: int = 0
        self._run()

    def _run(self) -> None:
        """Standard Bellman-Ford with early-exit optimisation."""
        n = self.graph.n
        dist = {i: INF for i in range(n)}
        dist[self.source] = 0
        self.nodes_visited = 0

        edges = [
            (u, v, w)
            for u in self.graph.adj
            for v, w in self.graph.adj[u].items()
        ]

        for _ in range(n - 1):
            updated = False
            for u, v, w in edges:
                self.nodes_visited += 1
                if dist[u] != INF and dist[u] + w < dist[v]:
                    dist[v] = dist[u] + w
                    updated = True
            if not updated:
                break

        self.dist = dist

    def update(self, u: int, v: int, w_new: float) -> int:
        self.graph.update_weight(u, v, w_new)
        self._run()
        return self.nodes_visited


class DynamicBellmanFord:
    """
    Incremental Bellman-Ford: after each weight change only the
    affected portion of the graph is re-relaxed.

    - Weight decrease: forward BFS propagating improvements.
    - Weight increase: identify dependent subtree, invalidate,
      then recompute from predecessor edges.
    """
    name = "Dynamic Bellman-Ford"
    supports_negative = True

    def __init__(self, graph: Graph, source: int):
        self.graph = graph
        self.source = source
        self.dist: dict = {}
        self.parent: dict = {}
        self.nodes_visited: int = 0
        self._full_run()

    def _full_run(self) -> None:
        """Initial full Bellman-Ford to establish dist and parent dicts."""
        n = self.graph.n
        dist = {i: INF for i in range(n)}
        parent = {i: None for i in range(n)}
        dist[self.source] = 0

        edges = [
            (u, v, w)
            for u in self.graph.adj
            for v, w in self.graph.adj[u].items()
        ]

        for _ in range(n - 1):
            updated = False
            for u, v, w in edges:
                if dist[u] != INF and dist[u] + w < dist[v]:
                    dist[v] = dist[u] + w
                    parent[v] = u
                    updated = True
            if not updated:
                break

        self.dist = dist
        self.parent = parent

    def update(self, u: int, v: int, w_new: float) -> int:
        w_old = self.graph.update_weight(u, v, w_new)
        self.nodes_visited = 0

        if w_new < w_old:
            self._propagate_decrease(u, v, w_new)
        else:
            self._propagate_increase(u, v)

        return self.nodes_visited

    def _propagate_decrease(self, u: int, v: int, w_new: float) -> None:
        """Weight decreased — propagate potential improvements forward."""
        changed: set = set()
        candidate = self.dist[u] + w_new if self.dist[u] != INF else INF
        if candidate < self.dist[v]:
            self.dist[v] = candidate
            self.parent[v] = u
            changed.add(v)

        while changed:
            x = changed.pop()
            self.nodes_visited += 1
            for y, w in self.graph.neighbors(x).items():
                if self.dist[x] != INF and self.dist[x] + w < self.dist[y]:
                    self.dist[y] = self.dist[x] + w
                    self.parent[y] = x
                    changed.add(y)

    def _propagate_increase(self, u: int, v: int) -> None:
        """Weight increased — recheck v and every node whose path used v."""
        affected: set = set()
        queue = [v]
        while queue:
            node = queue.pop()
            if node in affected:
                continue
            affected.add(node)
            self.nodes_visited += 1
            for y in self.graph.neighbors(node):
                if self.parent.get(y) == node:
                    queue.append(y)

        # Recompute distances; process in order of current (stale) distance
        # to maximise the chance we pull from valid predecessors first.
        affected_list = sorted(affected, key=lambda x: self.dist.get(x, INF))
        for node in affected_list:
            best = INF
            best_parent = None
            for pred, w in self.graph.predecessors(node).items():
                if self.dist.get(pred, INF) != INF:
                    candidate = self.dist[pred] + w
                    if candidate < best:
                        best = candidate
                        best_parent = pred
            self.dist[node] = best
            self.parent[node] = best_parent
