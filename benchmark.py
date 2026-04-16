# benchmark.py
# Terminal benchmark for the three TDSPP algorithms.
# Graph: 9-node TDRN, Q(v0, v8) at t=0, t=30, t=50.

import time, sys, os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from algorithms.graph import build_wang_graph, compute_path_cost
from algorithms import ALL_ALGORITHMS

SOURCE         = 0
DESTINATION    = 8
DEPARTURE_TIMES = [0, 30, 50]


def fmt_path(path):
    return " -> ".join(f"v{n}" for n in path) if path else "no path"


def sep(c="-", w=96): print(c * w)


def run_all():
    print("=" * 96)
    print("  TDSPP Benchmark  |  9-node TDRN  |  Q(v0, v8)")
    print("  Algorithms: TD-Dijkstra | TD-A* | TD-G-Tree")
    print("=" * 96)

    results = {}
    for algo in ALL_ALGORITHMS:
        results[algo.ALGORITHM_NAME] = []
        for t0 in DEPARTURE_TIMES:
            g = build_wang_graph()
            n_repeats = 20
            t_start = time.perf_counter()
            for _ in range(n_repeats):
                r = algo.run(g, SOURCE, DESTINATION, t0)
            r["elapsed_ms"] = (time.perf_counter() - t_start) / n_repeats * 1000
            r["t0"] = t0
            results[algo.ALGORITHM_NAME].append(r)

    print(f"\n  {'t0':>4}  ", end="")
    for algo in ALL_ALGORITHMS:
        print(f"  {algo.ALGORITHM_NAME:<22} {'ms':>8}  ", end="")
    print()
    sep()

    for idx, t0 in enumerate(DEPARTURE_TIMES):
        print(f"  {t0:>4}  ", end="")
        for algo in ALL_ALGORITHMS:
            r = results[algo.ALGORITHM_NAME][idx]
            tt = r["travel_time"]
            tt_str = f"{tt:.2f} min" if tt != float('inf') else "inf"
            print(f"  {tt_str:<22} {r['elapsed_ms']:>7.4f}ms  ", end="")
        print()

    sep()
    print("\n  Summary")
    sep()
    print(f"  {'Algorithm':<20} {'Avg Travel Time':>16} {'Avg Nodes':>12} {'Avg ms':>10} {'Correct':>8}")
    sep()

    ref = results["TD-Dijkstra"]
    for algo in ALL_ALGORITHMS:
        rs = results[algo.ALGORITHM_NAME]
        valid  = [r for r in rs if r["travel_time"] != float('inf')]
        avg_tt = sum(r["travel_time"] for r in valid) / len(valid) if valid else 0
        avg_ns = sum(r["nodes_settled"] for r in rs) / len(rs)
        avg_ms = sum(r["elapsed_ms"]    for r in rs) / len(rs)
        correct = sum(1 for i, r in enumerate(rs)
                      if abs(r["travel_time"] - ref[i]["travel_time"]) < 0.1)
        print(f"  {algo.ALGORITHM_NAME:<20} {avg_tt:>14.2f} min {avg_ns:>12.1f} {avg_ms:>9.4f}ms  {correct}/{len(rs)}")

    sep()
    print("\n  Paths for each departure time:")
    for idx, t0 in enumerate(DEPARTURE_TIMES):
        print(f"\n  t = {t0}")
        for algo in ALL_ALGORITHMS:
            r = results[algo.ALGORITHM_NAME][idx]
            print(f"    {algo.ALGORITHM_NAME:<14} {fmt_path(r['path'])}  [{r['travel_time']:.2f} min]")

    print("\nBenchmark complete.")


if __name__ == "__main__":
    run_all()
