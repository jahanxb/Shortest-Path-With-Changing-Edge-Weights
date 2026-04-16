# td_astar.py
# Time-Dependent A* (Zhao et al., 2008).
# Adds a static lower-bound heuristic to TD-Dijkstra to prune the search.
# h(v) = min-cost reverse shortest path from v to destination.
# Admissible: never overestimates, so still returns exact optimal answer.

import heapq

ALGORITHM_NAME = "TD-A*"
ALGORITHM_YEAR = "Zhao et al., 2008"
COLOR = "#388e3c"


def build_heuristic(graph, destination):
    min_cost = {}
    for (u, v), plf in graph.edges.items():
        min_cost[(u, v)] = min(w for _, w in plf)

    h = {n: float('inf') for n in graph.nodes}
    h[destination] = 0
    heap = [(0, destination)]
    settled = set()

    while heap:
        cost, u = heapq.heappop(heap)
        if u in settled:
            continue
        settled.add(u)
        for n in graph.nodes:
            if (n, u) in min_cost:
                nc = cost + min_cost[(n, u)]
                if nc < h[n]:
                    h[n] = nc
                    heapq.heappush(heap, (nc, n))
    return h


def run(graph, source, destination, departure_time):
    heuristic = build_heuristic(graph, destination)

    g = {n: float('inf') for n in graph.nodes}
    g[source] = departure_time
    prev = {n: None for n in graph.nodes}

    heap = [(departure_time + heuristic[source], departure_time, source)]
    settled = set()
    nodes_settled = 0
    nodes_relaxed = 0

    while heap:
        f, arrival, u = heapq.heappop(heap)
        if u in settled:
            continue
        settled.add(u)
        nodes_settled += 1
        if u == destination:
            break
        for v in graph.neighbors(u):
            if v in settled:
                continue
            cost = graph.get_cost(u, v, arrival)
            new_arr = arrival + cost
            nodes_relaxed += 1
            if new_arr < g[v]:
                g[v] = new_arr
                prev[v] = u
                heapq.heappush(heap, (new_arr + heuristic[v], new_arr, v))

    path = []
    cur = destination
    while cur is not None:
        path.append(cur)
        cur = prev[cur]
    path.reverse()

    if not path or path[0] != source:
        return {"travel_time": float('inf'), "path": [],
                "nodes_settled": nodes_settled, "nodes_relaxed": nodes_relaxed}

    return {
        "travel_time": round(g[destination] - departure_time, 4),
        "path": path,
        "nodes_settled": nodes_settled,
        "nodes_relaxed": nodes_relaxed
    }
