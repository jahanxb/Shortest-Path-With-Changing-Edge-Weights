"""
dynamic_sp_benchmark.py
========================
CLI benchmarking suite: Shortest Paths with Changing Edge Weights.

Algorithms compared head-to-head:
  1. Dijkstra Full Rerun          (baseline, non-negative only)
  2. Bellman-Ford Full Rerun      (baseline, handles negatives)
  3. Dynamic Bellman-Ford         (incremental, handles negatives)
  4. Ramalingam-Reps (RR-SSSP)   (fully dynamic, non-negative)
  5. LPA* (Lifelong Planning A*) (fully dynamic, non-negative)

Scenarios:
  A. Random Graph  — Erdős-Rényi with mixed weight changes
  B. Road Network  — Grid graph with traffic-style updates
  C. Worst-Case    — Layered dense graph, maximal SPT disruption
  D. Negative Weights — Only BF-based algorithms, no negative cycles

Metrics:
  - Total time (ms)
  - Average time per update (ms)
  - Average / max nodes visited per update
  - Correctness % vs Dijkstra ground truth

Run:
  python dynamic_sp_benchmark.py

See BENCHMARK.md for full test case catalogue and scoring matrix.
"""

import heapq
import time
import random
import copy
from dataclasses import dataclass
from typing import Dict, List, Tuple

# ── Import all algorithms from the algorithms/ package ────────────────────────
from algorithms import (
    Graph, INF,
    DijkstraRerun, BellmanFordRerun, DynamicBellmanFord,
    RamalingamReps, LPAStar,
)

# ─────────────────────────────────────────────────────────────────────────────
# GRAPH GENERATORS
# ─────────────────────────────────────────────────────────────────────────────

def generate_random_graph(n: int, edge_prob: float = 0.15,
                           w_min: float = 1, w_max: float = 20,
                           seed: int = 42) -> Tuple[Graph, List[Tuple]]:
    """
    Scenario A: Random directed graph (Erdős-Rényi model).
    Returns (graph, list_of_edge_tuples).
    """
    random.seed(seed)
    g = Graph(n)
    edges = []
    for u in range(n):
        for v in range(n):
            if u != v and random.random() < edge_prob:
                w = random.uniform(w_min, w_max)
                g.add_edge(u, v, w)
                edges.append((u, v))
    return g, edges


def generate_road_network(rows: int, cols: int,
                           base_w: float = 5.0, seed: int = 42) -> Tuple[Graph, List[Tuple]]:
    """
    Scenario B: Grid graph simulating a road network.
    4-directional bidirectional edges with randomised weights.
    """
    random.seed(seed)
    n = rows * cols
    g = Graph(n)
    edges = []

    def node_id(r, c):
        return r * cols + c

    for r in range(rows):
        for c in range(cols):
            u = node_id(r, c)
            if c + 1 < cols:
                v = node_id(r, c + 1)
                w = base_w + random.uniform(-2, 8)
                g.add_edge(u, v, max(1, w))
                g.add_edge(v, u, max(1, w + random.uniform(-1, 3)))
                edges += [(u, v), (v, u)]
            if r + 1 < rows:
                v = node_id(r + 1, c)
                w = base_w + random.uniform(-2, 8)
                g.add_edge(u, v, max(1, w))
                g.add_edge(v, u, max(1, w + random.uniform(-1, 3)))
                edges += [(u, v), (v, u)]

    return g, edges


def generate_worst_case_graph(n: int, seed: int = 42) -> Tuple[Graph, List[Tuple]]:
    """
    Scenario C: Layered dense graph.
    Almost every edge change invalidates a large portion of the SPT.
    """
    random.seed(seed)
    g = Graph(n)
    edges = []
    layer_size = max(2, n // 5)

    for u in range(n):
        targets = random.sample(
            range(u + 1, min(u + layer_size + 1, n)),
            k=min(layer_size, n - u - 1)
        ) if u < n - 1 else []
        for v in targets:
            w = random.uniform(1, 10)
            g.add_edge(u, v, w)
            edges.append((u, v))

    return g, edges


# ─────────────────────────────────────────────────────────────────────────────
# UPDATE GENERATOR
# ─────────────────────────────────────────────────────────────────────────────

def generate_updates(edges: List[Tuple], n_updates: int,
                     graph: Graph, mode: str = "mixed",
                     seed: int = 99) -> List[Tuple]:
    """
    Generate edge weight-change events.
    mode: 'decrease' | 'increase' | 'mixed' | 'traffic'
    """
    random.seed(seed)
    updates = []
    for _ in range(n_updates):
        u, v = random.choice(edges)
        w_cur = graph.get_weight(u, v)
        if mode == "decrease":
            w_new = w_cur * random.uniform(0.5, 0.95)
        elif mode == "increase":
            w_new = w_cur * random.uniform(1.05, 2.5)
        elif mode == "traffic":
            if random.random() < 0.3:
                w_new = w_cur * random.uniform(3.0, 8.0)   # traffic jam
            else:
                w_new = w_cur * random.uniform(0.3, 0.7)   # road clear
        else:  # mixed
            w_new = w_cur * random.uniform(0.4, 2.5)
        updates.append((u, v, max(0.1, w_new)))
    return updates


# ─────────────────────────────────────────────────────────────────────────────
# BENCHMARK ENGINE
# ─────────────────────────────────────────────────────────────────────────────

@dataclass
class BenchmarkResult:
    algorithm: str
    scenario: str
    n_nodes: int
    n_edges: int
    n_updates: int
    total_time_ms: float
    avg_time_per_update_ms: float
    avg_nodes_visited: float
    max_nodes_visited: int
    correctness: float   # fraction of updates with correct answer (%)
    errors: int


def compute_ground_truth(graph_orig: Graph, updates: List[Tuple],
                          source: int) -> List[Dict]:
    """
    Dijkstra full-rerun after each update → reference distances.
    """
    graph = graph_orig.clone()
    ground_truth = []
    for u, v, w_new in updates:
        graph.update_weight(u, v, w_new)
        dist = {i: INF for i in range(graph.n)}
        dist[source] = 0
        pq = [(0, source)]
        visited: set = set()
        while pq:
            d, node = heapq.heappop(pq)
            if node in visited:
                continue
            visited.add(node)
            for nb, w in graph.neighbors(node).items():
                if dist[node] + w < dist[nb]:
                    dist[nb] = dist[node] + w
                    heapq.heappush(pq, (dist[nb], nb))
        ground_truth.append(dict(dist))
    return ground_truth


def run_benchmark(AlgoClass, graph_orig: Graph, updates: List[Tuple],
                  source: int, ground_truth_dists: List[Dict],
                  scenario: str) -> BenchmarkResult:
    """Run one algorithm against the update sequence; check correctness."""
    graph = graph_orig.clone()
    algo = AlgoClass(graph, source)

    total_time = 0.0
    total_nodes = 0
    max_nodes = 0
    correct = 0
    errors = 0

    for i, (u, v, w_new) in enumerate(updates):
        t0 = time.perf_counter()
        nodes = algo.update(u, v, w_new)
        t1 = time.perf_counter()

        elapsed_ms = (t1 - t0) * 1000
        total_time += elapsed_ms
        total_nodes += nodes
        max_nodes = max(max_nodes, nodes)

        gt = ground_truth_dists[i]
        algo_dist = algo.dist
        ok = True
        for node, gt_d in gt.items():
            algo_d = algo_dist.get(node, INF)
            if gt_d == INF:
                if algo_d != INF:
                    ok = False; break
            elif abs(algo_d - gt_d) > 1e-6:
                ok = False; break
        correct += int(ok)
        errors += int(not ok)

    n = len(updates)
    n_edges = sum(len(v) for v in graph_orig.adj.values())
    return BenchmarkResult(
        algorithm=AlgoClass.name,
        scenario=scenario,
        n_nodes=graph_orig.n,
        n_edges=n_edges,
        n_updates=n,
        total_time_ms=round(total_time, 3),
        avg_time_per_update_ms=round(total_time / n, 4),
        avg_nodes_visited=round(total_nodes / n, 1),
        max_nodes_visited=max_nodes,
        correctness=round(correct / n * 100, 1),
        errors=errors,
    )


# ─────────────────────────────────────────────────────────────────────────────
# REPORTING
# ─────────────────────────────────────────────────────────────────────────────

def print_header(title: str):
    width = 72
    print("\n" + "═" * width)
    print(f"  {title}")
    print("═" * width)


def print_results(results: List[BenchmarkResult]):
    col_w = [28, 10, 10, 14, 14, 12, 10]
    headers = ["Algorithm", "Total(ms)", "Avg(ms)", "Avg Nodes", "Max Nodes", "Correct%", "Errors"]
    sep = "─" * sum(col_w)
    print(sep)
    print("".join(h.ljust(col_w[i]) for i, h in enumerate(headers)))
    print(sep)
    for r in results:
        row = [r.algorithm[:27], f"{r.total_time_ms:.1f}", f"{r.avg_time_per_update_ms:.3f}",
               f"{r.avg_nodes_visited:.1f}", str(r.max_nodes_visited),
               f"{r.correctness}%", str(r.errors)]
        print("".join(str(row[i]).ljust(col_w[i]) for i in range(len(col_w))))
    print(sep)


def print_speedup(results: List[BenchmarkResult]):
    baseline = next((r for r in results if "Dijkstra" in r.algorithm), None)
    if not baseline:
        return
    print("\n  Speedup vs Dijkstra Full Rerun (avg time per update):")
    for r in results:
        speedup = baseline.avg_time_per_update_ms / max(r.avg_time_per_update_ms, 1e-9)
        bar = "█" * min(int(speedup * 4), 40)
        print(f"    {r.algorithm:<30} {speedup:>6.2f}x  {bar}")


# ─────────────────────────────────────────────────────────────────────────────
# SCENARIOS
# ─────────────────────────────────────────────────────────────────────────────

ALGORITHMS = [DijkstraRerun, BellmanFordRerun, DynamicBellmanFord, RamalingamReps, LPAStar]


def run_scenario_a(n_updates: int = 80):
    print_header("SCENARIO A: Random Graph with Random Weight Updates")
    for cfg in [
        {"n": 50,  "edge_prob": 0.2,  "label": "Small  (50 nodes)"},
        {"n": 150, "edge_prob": 0.1,  "label": "Medium (150 nodes)"},
        {"n": 400, "edge_prob": 0.05, "label": "Large  (400 nodes)"},
    ]:
        n = cfg["n"]
        print(f"\n  ▶ {cfg['label']}")
        graph, edges = generate_random_graph(n, edge_prob=cfg["edge_prob"])
        if not edges:
            print("    No edges generated — skipping.")
            continue
        updates = generate_updates(edges, n_updates, graph, mode="mixed")
        ground_truth = compute_ground_truth(graph, updates, source=0)
        results = []
        for Algo in ALGORITHMS:
            r = run_benchmark(Algo, graph, updates, source=0,
                              ground_truth_dists=ground_truth, scenario="Random Graph")
            results.append(r)
            print(f"    ✓ {Algo.name}")
        print(f"\n  Config: {n} nodes, {results[0].n_edges} edges, {n_updates} updates")
        print_results(results)
        print_speedup(results)


def run_scenario_b(n_updates: int = 80):
    print_header("SCENARIO B: Road Network Simulation (Grid Graph)")
    for cfg in [
        {"rows": 8,  "cols": 8,  "label": "Small  (8×8 grid, 64 nodes)"},
        {"rows": 14, "cols": 14, "label": "Medium (14×14 grid, 196 nodes)"},
        {"rows": 22, "cols": 22, "label": "Large  (22×22 grid, 484 nodes)"},
    ]:
        rows, cols = cfg["rows"], cfg["cols"]
        print(f"\n  ▶ {cfg['label']}")
        graph, edges = generate_road_network(rows, cols)
        updates = generate_updates(edges, n_updates, graph, mode="traffic")
        ground_truth = compute_ground_truth(graph, updates, source=0)
        results = []
        for Algo in ALGORITHMS:
            r = run_benchmark(Algo, graph, updates, source=0,
                              ground_truth_dists=ground_truth, scenario="Road Network")
            results.append(r)
            print(f"    ✓ {Algo.name}")
        print(f"\n  Config: {rows}×{cols} grid, {results[0].n_edges} edges, {n_updates} updates (traffic)")
        print_results(results)
        print_speedup(results)


def run_scenario_c(n_updates: int = 60):
    print_header("SCENARIO C: Worst-Case Stress Test (Layered Dense Graph)")
    print("""
  Structured to maximise the SPT subtree invalidated per update.
  Tests how well incremental algorithms degrade under adversarial input.
    """)
    for cfg in [
        {"n": 40,  "label": "Small  (40 nodes, layered)"},
        {"n": 120, "label": "Medium (120 nodes, layered)"},
        {"n": 300, "label": "Large  (300 nodes, layered)"},
    ]:
        n = cfg["n"]
        print(f"\n  ▶ {cfg['label']}")
        graph, edges = generate_worst_case_graph(n)
        if not edges:
            print("    No edges — skipping.")
            continue
        updates = generate_updates(edges, n_updates, graph, mode="mixed")
        ground_truth = compute_ground_truth(graph, updates, source=0)
        results = []
        for Algo in ALGORITHMS:
            r = run_benchmark(Algo, graph, updates, source=0,
                              ground_truth_dists=ground_truth, scenario="Worst Case")
            results.append(r)
            print(f"    ✓ {Algo.name}")
        print(f"\n  Config: {n} nodes, {results[0].n_edges} edges, {n_updates} updates")
        print_results(results)
        print_speedup(results)


def run_scenario_d_negative(n_updates: int = 60):
    print_header("BONUS SCENARIO D: Graphs with Negative Weights")
    print("""
  Only Bellman-Ford based algorithms are tested.
  Dijkstra-based algorithms (RR-SSSP, LPA*) are excluded —
  they produce incorrect results on negative-weight edges.
    """)
    n = 80
    random.seed(77)
    graph = Graph(n)
    edges = []
    for u in range(n):
        for _ in range(3):
            v = random.randint(0, n - 1)
            if v != u and v > u:   # forward edges only → no negative cycles
                w = random.uniform(-3, 15)
                graph.add_edge(u, v, w)
                edges.append((u, v))

    if not edges:
        print("  No edges generated.")
        return

    updates = generate_updates(edges, n_updates, graph, mode="mixed", seed=55)
    updates = [(u, v, max(w, -3.0)) for u, v, w in updates]  # clamp

    # Ground truth via Bellman-Ford
    graph_gt = graph.clone()
    ground_truth = []
    for u, v, w_new in updates:
        graph_gt.update_weight(u, v, w_new)
        dist = {i: INF for i in range(n)}
        dist[0] = 0
        edge_list = [(uu, vv, ww) for uu in graph_gt.adj for vv, ww in graph_gt.adj[uu].items()]
        for _ in range(n - 1):
            for uu, vv, ww in edge_list:
                if dist[uu] != INF and dist[uu] + ww < dist[vv]:
                    dist[vv] = dist[uu] + ww
        ground_truth.append(dict(dist))

    results = []
    for Algo in [BellmanFordRerun, DynamicBellmanFord]:
        r = run_benchmark(Algo, graph, updates, source=0,
                          ground_truth_dists=ground_truth, scenario="Negative Weights")
        results.append(r)
        print(f"    ✓ {Algo.name}")

    print(f"\n  Config: {n} nodes, {results[0].n_edges} edges, {n_updates} updates (negative weights)")
    print_results(results)
    baseline = results[0]
    print(f"\n  Dynamic BF speedup vs Full Rerun: "
          f"{baseline.avg_time_per_update_ms / max(results[1].avg_time_per_update_ms, 1e-9):.2f}x")


def print_summary():
    print_header("ALGORITHM REFERENCE SUMMARY")
    print("""
  ┌─────────────────────────────┬──────────┬──────────┬──────────┬──────────────────────┐
  │ Algorithm                   │ Decrease │ Increase │ Negative │ Per-Update Cost      │
  ├─────────────────────────────┼──────────┼──────────┼──────────┼──────────────────────┤
  │ Dijkstra Full Rerun         │   Yes    │   Yes    │    No    │ O((V+E) log V)       │
  │ Bellman-Ford Full Rerun     │   Yes    │   Yes    │   Yes    │ O(VE)                │
  │ Dynamic Bellman-Ford        │   Yes    │   Yes    │   Yes    │ O(k·E) — k affected  │
  │ Ramalingam-Reps (RR-SSSP)  │   Yes    │   Yes    │    No    │ O(k log V)           │
  │ LPA*                        │   Yes    │   Yes    │    No*   │ O(k log V)           │
  └─────────────────────────────┴──────────┴──────────┴──────────┴──────────────────────┘
  * LPA* can handle negatives with an admissible heuristic

  KEY INSIGHT: Dynamic algorithms outperform reruns when k << V
               (i.e., when only a small fraction of nodes are affected).
               In worst-case scenarios, all algorithms approach rerun cost.
    """)


# ─────────────────────────────────────────────────────────────────────────────
# MAIN
# ─────────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    print("\n" + "█" * 72)
    print("  DYNAMIC SHORTEST PATH BENCHMARK SUITE")
    print("  Comparing 5 algorithms across 3 scenarios + negative weight test")
    print("█" * 72)

    N_UPDATES = 80

    run_scenario_a(n_updates=N_UPDATES)
    run_scenario_b(n_updates=N_UPDATES)
    run_scenario_c(n_updates=N_UPDATES)
    run_scenario_d_negative(n_updates=N_UPDATES)
    print_summary()

    print("\n✅ Benchmark complete.\n")