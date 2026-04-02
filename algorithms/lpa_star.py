"""
algorithms/lpa_star.py
======================
LPA* — Lifelong Planning A* (Koenig & Likhachev, 2002).

LPA* maintains two values per node:
  g(v)   — current best known distance from source to v.
  rhs(v) — one-step lookahead: min over predecessors of g(pred) + w(pred->v).

A node is *consistent*   when g(v) == rhs(v).
A node is *inconsistent* when g(v) != rhs(v) — it lives in the open list.

On each update only the inconsistent nodes are re-expanded, so the
repair work is proportional to k (nodes whose distances change) not V.

Complexity (per update)
-----------------------
  Time : O(k log k)  where k = number of inconsistent nodes processed.
  Space: O(V)

Note: with h=0 (zero heuristic, as used here) LPA* reduces to
incremental Dijkstra.  This is a valid and correct special case.

Limitations
-----------
  - Does NOT support negative edge weights (priority queue ordering
    relies on non-negative cost monotonicity).
  - Each single-edge update triggers re-expansion of all transitively
    affected nodes; massive simultaneous updates can be expensive.

Breaking cases
--------------
  1. Negative edge weight     → over-consistency never terminates OR
                                 wrong g values accepted prematurely.
  2. Non-admissible heuristic → a custom h(v) > real_dist(v) causes
                                 LPA* to skip the optimal path.
  3. 10,000 simultaneous edge changes → O(k log k) per change summed
                                         is worse than one Dijkstra rerun.
"""

import heapq
from collections import defaultdict
from algorithms.graph import Graph, INF


class LPAStar:
    """
    LPA* with zero heuristic (= incremental Dijkstra).
    Maintains g/rhs tables and an open-list of inconsistent nodes.
    Repairs after each edge-weight change by processing only
    inconsistent nodes in priority order.
    """
    name = "LPA*"
    supports_negative = False

    def __init__(self, graph: Graph, source: int):
        self.graph = graph
        self.source = source
        self.g:   dict = defaultdict(lambda: INF)
        self.rhs: dict = defaultdict(lambda: INF)
        self.nodes_visited: int = 0
        self.visited_nodes_list: list = []
        self._v_set: set = set()

        self._pq: list = []            # min-heap of (key, node)
        self._in_pq: dict = {}         # node -> key currently in heap

        self.rhs[source] = 0
        self._insert(source, self._key(source))
        self._compute()
        self.visited_nodes_list = list(self._v_set)

    # ── Priority queue helpers ────────────────────────────────────────────────

    def _key(self, v: int) -> float:
        """Priority key = min(g, rhs) — smaller means more urgent."""
        return min(self.g[v], self.rhs[v])

    def _insert(self, v: int, key: float) -> None:
        self._in_pq[v] = key
        heapq.heappush(self._pq, (key, v))

    def _remove(self, v: int) -> None:
        self._in_pq.pop(v, None)

    # ── Core LPA* operations ──────────────────────────────────────────────────

    def _update_vertex(self, v: int) -> None:
        """
        Recompute rhs(v) from its predecessors and re-insert into
        the open list if v becomes inconsistent.
        """
        if v != self.source:
            best = INF
            for pred, w in self.graph.predecessors(v).items():
                if self.g[pred] != INF:
                    best = min(best, self.g[pred] + w)
            self.rhs[v] = best

        self._remove(v)
        if self.g[v] != self.rhs[v]:
            self._insert(v, self._key(v))

    def _compute(self) -> None:
        """
        Process all inconsistent nodes until the open list is empty.
        Underconsistent  (g > rhs): lower g to rhs, propagate to neighbors.
        Overconsistent   (g < rhs): raise g to INF, recheck self + neighbors.
        """
        while self._pq:
            key, u = heapq.heappop(self._pq)
            if self._in_pq.get(u) != key:
                continue   # stale heap entry
            self._in_pq.pop(u, None)
            self.nodes_visited += 1
            self._v_set.add(u)

            if self.g[u] > self.rhs[u]:
                # Underconsistent — accept the lower value
                self.g[u] = self.rhs[u]
                for v in self.graph.neighbors(u):
                    self._update_vertex(v)
            else:
                # Overconsistent — reset and force recheck
                self.g[u] = INF
                self._update_vertex(u)
                for v in self.graph.neighbors(u):
                    self._update_vertex(v)

    # ── Public interface ──────────────────────────────────────────────────────

    @property
    def dist(self) -> dict:
        """Return current best distances as a plain dict."""
        return dict(self.g)

    def update(self, u: int, v: int, w_new: float) -> int:
        """Apply weight change and repair inconsistencies. Returns nodes visited."""
        self.nodes_visited = 0
        self._v_set = set()
        self.graph.update_weight(u, v, w_new)
        # Only v's rhs is directly affected; LPA* propagates the rest.
        self._update_vertex(v)
        self._compute()
        self.visited_nodes_list = list(self._v_set)
        return self.nodes_visited
