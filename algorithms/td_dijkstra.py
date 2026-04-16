# td_dijkstra.py
# Time-Dependent Dijkstra (Dreyfus, 1969).
# Exact algorithm. Settles nodes in order of earliest arrival time.
# Edge cost evaluated at actual arrival time at each node, not departure time.

import heapq

ALGORITHM_NAME = "TD-Dijkstra"
ALGORITHM_YEAR = "Dreyfus, 1969"
COLOR = "#1976d2"


def run(graph, source, destination, departure_time):
    dist = {n: float('inf') for n in graph.nodes}
    dist[source] = departure_time
    prev = {n: None for n in graph.nodes}
    heap = [(departure_time, source)]
    settled = set()
    nodes_settled = 0
    nodes_relaxed = 0

    while heap:
        arrival, u = heapq.heappop(heap)
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
            if new_arr < dist[v]:
                dist[v] = new_arr
                prev[v] = u
                heapq.heappush(heap, (new_arr, v))

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
        "travel_time": round(dist[destination] - departure_time, 4),
        "path": path,
        "nodes_settled": nodes_settled,
        "nodes_relaxed": nodes_relaxed
    }
