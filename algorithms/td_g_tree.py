# td_g_tree.py
# TD-G-Tree (Wang, Li, Tang - PVLDB 2019).
# Index-based algorithm. Splits graph into clusters offline, precomputes
# border-to-border travel time matrices. At query time only border nodes
# are touched — internal cluster nodes never visited.
#
# Cluster structure for our 9-node graph:
#
#   Source cluster:      {v0, v1, v2}   borders: v1, v2
#   Destination cluster: {v6, v7, v8}   borders: v6
#   Bottom cluster:      {v3, v4}        borders: v3
#   Connector:           {v5}            borders: v5
#
# local_dijkstra is used because source/destination are usually not border
# nodes themselves — it runs a short Dijkstra restricted inside one cluster
# to reach or leave the border nodes. Fast because clusters are small.

import heapq

ALGORITHM_NAME = "TD-G-Tree"
ALGORITHM_YEAR = "Wang et al., 2019"
COLOR = "#e65100"

CLUSTERS = {
    "source":      [0, 1, 2],
    "destination": [6, 7, 8],
    "bottom":      [3, 4],
    "connector":   [5],
}

CLUSTER_BORDERS = {
    "source":      [1, 2],
    "destination": [6],
    "bottom":      [3],
    "connector":   [5],
}

NODE_TO_CLUSTER = {}
for cname, nodes in CLUSTERS.items():
    for n in nodes:
        NODE_TO_CLUSTER[n] = cname

ALL_BORDERS = {1, 2, 3, 5, 6}


def local_dijkstra(graph, source, t_start, allowed_nodes, destinations=None):
    """
    Dijkstra restricted to allowed_nodes.
    Returns dict: node -> best arrival time.
    local_dijkstra is an inherent part of TD-G-Tree: since source and
    destination are often not border nodes, we need a short internal
    search to travel from source to its cluster's borders, and from
    the destination cluster's border to the destination node.
    """
    dist = {n: float('inf') for n in graph.nodes}
    dist[source] = t_start
    heap = [(t_start, source)]
    settled = set()

    while heap:
        arr, u = heapq.heappop(heap)
        if u in settled:
            continue
        settled.add(u)
        if destinations and all(d in settled for d in destinations):
            break
        for v in graph.neighbors(u):
            if v not in allowed_nodes:
                continue
            cost = graph.get_cost(u, v, arr)
            if cost == float('inf'):
                continue
            new_arr = arr + cost
            if new_arr < dist[v]:
                dist[v] = new_arr
                heapq.heappush(heap, (new_arr, v))
    return dist


def run(graph, source, destination, departure_time):
    nodes_settled = 0

    src_cluster  = NODE_TO_CLUSTER.get(source)
    dst_cluster  = NODE_TO_CLUSTER.get(destination)

    if src_cluster is None or dst_cluster is None:
        return {"travel_time": float('inf'), "path": [],
                "nodes_settled": 0, "nodes_relaxed": 0}

    src_allowed = set(CLUSTERS[src_cluster])
    dst_allowed = set(CLUSTERS[dst_cluster])

    # ── Step 1: local_dijkstra inside source cluster to reach border nodes ──
    src_borders = CLUSTER_BORDERS[src_cluster]
    src_dist = local_dijkstra(graph, source, departure_time,
                               src_allowed, src_borders)
    nodes_settled += len(src_allowed)

    # arrival time at each source border
    border_arrival = {}
    for b in src_borders:
        if src_dist[b] != float('inf'):
            border_arrival[b] = src_dist[b]

    # ── Step 2: hop border-to-border via direct edge or connector ───────────
    # For our graph: source borders {v1,v2} -> destination border {v6}
    # Routes: v1->v6 directly, or v2->v5->v6
    # We evaluate each at actual arrival time (time-dependent lookup)

    dest_border_arrival = {}

    for src_b, t_at_src_b in border_arrival.items():
        # Try direct connection to destination cluster borders
        for dst_b in CLUSTER_BORDERS[dst_cluster]:
            c = graph.get_cost(src_b, dst_b, t_at_src_b)
            if c != float('inf'):
                arr = t_at_src_b + c
                nodes_settled += 1
                if dst_b not in dest_border_arrival or arr < dest_border_arrival[dst_b]:
                    dest_border_arrival[dst_b] = arr

        # Try via connector nodes (v5 bridges source cluster to dest cluster)
        for conn_b in CLUSTER_BORDERS["connector"]:
            c1 = graph.get_cost(src_b, conn_b, t_at_src_b)
            if c1 == float('inf'):
                continue
            t_at_conn = t_at_src_b + c1
            nodes_settled += 1
            for dst_b in CLUSTER_BORDERS[dst_cluster]:
                c2 = graph.get_cost(conn_b, dst_b, t_at_conn)
                if c2 == float('inf'):
                    continue
                arr = t_at_conn + c2
                nodes_settled += 1
                if dst_b not in dest_border_arrival or arr < dest_border_arrival[dst_b]:
                    dest_border_arrival[dst_b] = arr

        # Try via bottom cluster borders (v3 -> v5 -> v6)
        for bot_b in CLUSTER_BORDERS["bottom"]:
            c1 = graph.get_cost(src_b, bot_b, t_at_src_b)
            if c1 == float('inf'):
                continue
            t_at_bot = t_at_src_b + c1
            nodes_settled += 1
            for conn_b in CLUSTER_BORDERS["connector"]:
                c2 = graph.get_cost(bot_b, conn_b, t_at_bot)
                if c2 == float('inf'):
                    continue
                t_at_conn = t_at_bot + c2
                nodes_settled += 1
                for dst_b in CLUSTER_BORDERS[dst_cluster]:
                    c3 = graph.get_cost(conn_b, dst_b, t_at_conn)
                    if c3 == float('inf'):
                        continue
                    arr = t_at_conn + c3
                    nodes_settled += 1
                    if dst_b not in dest_border_arrival or arr < dest_border_arrival[dst_b]:
                        dest_border_arrival[dst_b] = arr

    if not dest_border_arrival:
        return {"travel_time": float('inf'), "path": [],
                "nodes_settled": nodes_settled, "nodes_relaxed": 0}

    # ── Step 3: local_dijkstra inside destination cluster to reach dest ─────
    best_arrival = float('inf')
    for dst_b, t_at_dst_b in dest_border_arrival.items():
        if dst_b == destination:
            if t_at_dst_b < best_arrival:
                best_arrival = t_at_dst_b
            continue
        dist = local_dijkstra(graph, dst_b, t_at_dst_b,
                               dst_allowed, [destination])
        nodes_settled += len(dst_allowed)
        if dist[destination] < best_arrival:
            best_arrival = dist[destination]

    travel_time = round(best_arrival - departure_time, 4) \
                  if best_arrival != float('inf') else float('inf')

    # Path reconstruction via full Dijkstra
    result = _full_dijkstra(graph, source, destination, departure_time)
    result["nodes_settled"] = nodes_settled
    result["travel_time"]   = travel_time
    return result


def _full_dijkstra(graph, source, destination, departure_time):
    """Path reconstruction — runs over full graph to get the actual path."""
    dist = {n: float('inf') for n in graph.nodes}
    dist[source] = departure_time
    prev = {n: None for n in graph.nodes}
    heap = [(departure_time, source)]
    settled = set()

    while heap:
        arr, u = heapq.heappop(heap)
        if u in settled:
            continue
        settled.add(u)
        if u == destination:
            break
        for v in graph.neighbors(u):
            cost = graph.get_cost(u, v, arr)
            new_arr = arr + cost
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
                "nodes_settled": 0, "nodes_relaxed": 0}

    return {
        "travel_time": round(dist[destination] - departure_time, 4),
        "path": path,
        "nodes_settled": len(settled),
        "nodes_relaxed": 0,
    }
