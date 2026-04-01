"""
algorithms/ — Shortest Path Algorithm Package
=============================================
All algorithm classes share a common interface:

    __init__(graph: Graph, source: int)
        Initialises the algorithm and computes the initial SSSP.

    update(u: int, v: int, w_new: float) -> int
        Applies the edge-weight change (u, v) -> w_new and repairs
        the solution incrementally (or via full rerun, for baselines).
        Returns the number of nodes visited during this update.

    .dist  : dict[int, float]
        Current shortest distances from source to every node.

    .name  : str   (class attribute)
        Human-readable algorithm name used in reports and charts.

    .supports_negative : bool  (class attribute)
        True if the algorithm produces correct results on graphs
        that contain negative (but cycle-free) edge weights.
"""

from algorithms.graph import Graph, INF
from algorithms.dijkstra import DijkstraRerun
from algorithms.bellman_ford import BellmanFordRerun, DynamicBellmanFord
from algorithms.ramalingam_reps import RamalingamReps
from algorithms.lpa_star import LPAStar
from algorithms.quantum_sssp import QuantumSSSP

__all__ = [
    "Graph",
    "INF",
    "DijkstraRerun",
    "BellmanFordRerun",
    "DynamicBellmanFord",
    "RamalingamReps",
    "LPAStar",
    "QuantumSSSP",
]
