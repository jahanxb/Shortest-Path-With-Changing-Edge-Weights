# graph.py
# 9-node TDRN matching the experiment slide exactly.
# Nodes: v0 to v8, 11 bidirectional edges.
# Edge weights are piecewise linear functions (PLF) of departure time.
# Source: v0, Destination: v8

class TimeDependentGraph:

    def __init__(self):
        self.nodes = []
        self.edges = {}   # (u, v) -> [(time, weight), ...]

    def add_node(self, node_id):
        if node_id not in self.nodes:
            self.nodes.append(node_id)

    def add_edge(self, u, v, plf_points):
        self.edges[(u, v)] = plf_points
        self.edges[(v, u)] = plf_points

    def get_cost(self, u, v, t):
        if (u, v) not in self.edges:
            return float('inf')
        points = self.edges[(u, v)]
        if t <= points[0][0]:
            return points[0][1]
        if t >= points[-1][0]:
            return points[-1][1]
        for i in range(len(points) - 1):
            t0, w0 = points[i]
            t1, w1 = points[i + 1]
            if t0 <= t <= t1:
                slope = (w1 - w0) / (t1 - t0)
                return round(w0 + slope * (t - t0), 4)
        return points[-1][1]

    def neighbors(self, node):
        return [v for (u, v) in self.edges if u == node]


def build_wang_graph():
    """
    9-node time-dependent road network verified from experiment slide.
    Nodes v0-v8, 11 edges, source=v0, destination=v8.

    Edges exactly as shown in the table:
      e(v1,v2): {(0,8),(20,8),(35,20),(60,20)}  -- time-varying
      e(v0,v2): {(0,8),(60,8)}
      e(v0,v1): {(0,4),(60,4)}
      e(v1,v6): {(0,5),(20,5),(30,18),(60,5)}   -- time-varying (spikes at t=30)
      e(v5,v6): {(0,8),(25,8),(45,12),(60,12)}  -- time-varying
      e(v2,v3): {(0,15),(60,15)}
      e(v3,v4): {(0,6),(60,6)}
      e(v3,v5): {(0,22),(20,22),(35,6),(60,6)}  -- time-varying
      e(v2,v5): {(0,5),(60,5)}
      e(v6,v7): {(0,2),(60,2)}
      e(v7,v8): {(0,3),(60,3)}
    """
    g = TimeDependentGraph()
    for i in range(9):
        g.add_node(i)

    g.add_edge(0, 1, [(0, 4),  (60, 4)])
    g.add_edge(0, 2, [(0, 8),  (60, 8)])
    g.add_edge(1, 2, [(0, 8),  (20, 8), (35, 20), (60, 20)])
    g.add_edge(1, 6, [(0, 5),  (20, 5), (30, 18), (60, 5)])
    g.add_edge(2, 3, [(0, 15), (60, 15)])
    g.add_edge(2, 5, [(0, 5),  (60, 5)])
    g.add_edge(3, 4, [(0, 6),  (60, 6)])
    g.add_edge(3, 5, [(0, 22), (20, 22), (35, 6), (60, 6)])
    g.add_edge(5, 6, [(0, 8),  (25, 8), (45, 12), (60, 12)])
    g.add_edge(6, 7, [(0, 2),  (60, 2)])
    g.add_edge(7, 8, [(0, 3),  (60, 3)])

    return g


def compute_path_cost(graph, path, departure_time):
    if len(path) < 2:
        return 0.0
    t = departure_time
    for i in range(len(path) - 1):
        cost = graph.get_cost(path[i], path[i + 1], t)
        if cost == float('inf'):
            return float('inf')
        t += cost
    return round(t - departure_time, 4)
