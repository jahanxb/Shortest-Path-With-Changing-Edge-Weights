"""
algorithms/graph.py
===================
Shared Graph data structure used by all algorithm implementations.

The Graph uses two adjacency dicts (forward + reverse) so that both
successor and predecessor lookups are O(degree) — required by LPA*,
Dynamic Bellman-Ford, and Ramalingam-Reps.
"""

from collections import defaultdict
from typing import Dict

INF = float('inf')


class Graph:
    """
    Directed, weighted graph with O(1) edge-weight updates.

    Attributes
    ----------
    n    : int   — number of nodes (nodes are always integers 0..n-1)
    adj  : dict  — forward adjacency:  adj[u][v]  = weight of edge u->v
    radj : dict  — reverse adjacency:  radj[v][u] = weight of edge u->v
    """

    def __init__(self, n: int):
        self.n = n
        self.adj:  Dict[int, Dict[int, float]] = defaultdict(dict)
        self.radj: Dict[int, Dict[int, float]] = defaultdict(dict)

    def add_edge(self, u: int, v: int, w: float) -> None:
        """Add or overwrite directed edge u -> v with weight w."""
        self.adj[u][v] = w
        self.radj[v][u] = w

    def update_weight(self, u: int, v: int, w: float) -> float:
        """
        Change weight of existing edge u -> v to w.
        Returns the previous weight (INF if edge did not exist).
        """
        old = self.adj[u].get(v, INF)
        self.adj[u][v] = w
        self.radj[v][u] = w
        return old

    def get_weight(self, u: int, v: int) -> float:
        """Return current weight of edge u -> v (INF if not present)."""
        return self.adj[u].get(v, INF)

    def neighbors(self, u: int) -> Dict[int, float]:
        """Return {v: weight} for all outgoing edges from u."""
        return self.adj[u]

    def predecessors(self, v: int) -> Dict[int, float]:
        """Return {u: weight} for all incoming edges to v."""
        return self.radj[v]

    def edge_list(self):
        """Return list of (u, v) pairs for all edges."""
        return [(u, v) for u in self.adj for v in self.adj[u]]

    def clone(self) -> 'Graph':
        """Return a deep copy of this graph."""
        g = Graph(self.n)
        for u in self.adj:
            for v, w in self.adj[u].items():
                g.add_edge(u, v, w)
        return g
