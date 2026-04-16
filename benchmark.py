# benchmark.py
# Terminal benchmark for TDSPP algorithms on the 9-node TDRN.
# Covers three classical algorithms + quantum annealing.
# Query: Q(v0, v8) at t = 0, 30, 50.

import time, sys, os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from algorithms.graph import build_wang_graph, compute_path_cost
from algorithms import ALL_ALGORITHMS, QUANTUM_ALGORITHM

SOURCE          = 0
DESTINATION     = 8
DEPARTURE_TIMES = [0, 30, 50]


def fmt_path(path):
    return " -> ".join(f"v{n}" for n in path) if path else "no path"


def sep(c="-", w=96):
    print(c * w)


def run_classical():
    print("=" * 96)
    print("  TDSPP Benchmark  |  9-node TDRN  |  Q(v0, v8)")
    print("  Classical Algorithms: TD-Dijkstra | TD-A* | TD-G-Tree")
    print("=" * 96)

    results = {}
    for algo in ALL_ALGORITHMS:
        results[algo.ALGORITHM_NAME] = []
        for t0 in DEPARTURE_TIMES:
            g = build_wang_graph()
            n_rep = 20
            t_start = time.perf_counter()
            for _ in range(n_rep):
                r = algo.run(g, SOURCE, DESTINATION, t0)
            r["elapsed_ms"] = (time.perf_counter() - t_start) / n_rep * 1000
            r["t0"] = t0
            results[algo.ALGORITHM_NAME].append(r)

    # Per departure time table
    print(f"\n  {'t0':>4}  ", end="")
    for algo in ALL_ALGORITHMS:
        print(f"  {algo.ALGORITHM_NAME:<22} {'ms':>8}  ", end="")
    print()
    sep()

    for idx, t0 in enumerate(DEPARTURE_TIMES):
        print(f"  {t0:>4}  ", end="")
        for algo in ALL_ALGORITHMS:
            r  = results[algo.ALGORITHM_NAME][idx]
            tt = r["travel_time"]
            tt_str = f"{tt:.2f} min" if tt != float('inf') else "inf"
            print(f"  {tt_str:<22} {r['elapsed_ms']:>7.4f}ms  ", end="")
        print()

    sep()

    # Summary
    print("\n  Summary")
    sep()
    print(f"  {'Algorithm':<20} {'Avg Travel Time':>16} {'Avg Nodes':>12}"
          f" {'Avg ms':>10} {'Correct':>8}")
    sep()
    ref = results["TD-Dijkstra"]
    for algo in ALL_ALGORITHMS:
        rs      = results[algo.ALGORITHM_NAME]
        avg_tt  = sum(r["travel_time"] for r in rs) / len(rs)
        avg_ns  = sum(r["nodes_settled"] for r in rs) / len(rs)
        avg_ms  = sum(r["elapsed_ms"]    for r in rs) / len(rs)
        correct = sum(1 for i, r in enumerate(rs)
                      if abs(r["travel_time"] - ref[i]["travel_time"]) < 0.1)
        print(f"  {algo.ALGORITHM_NAME:<20} {avg_tt:>14.2f} min"
              f" {avg_ns:>12.1f} {avg_ms:>9.4f}ms  {correct}/{len(rs)}")
    sep()

    # Paths per departure
    print("\n  Optimal paths:")
    for idx, t0 in enumerate(DEPARTURE_TIMES):
        print(f"\n  t = {t0}")
        for algo in ALL_ALGORITHMS:
            r = results[algo.ALGORITHM_NAME][idx]
            print(f"    {algo.ALGORITHM_NAME:<14} {fmt_path(r['path'])}"
                  f"  [{r['travel_time']:.2f} min]")

    return results


def run_quantum():
    print("\n" + "=" * 96)
    print("  Quantum Annealing  |  QUBO Formulation  |  Q(v0, v8)")
    print("  Framework: D-Wave Ocean SDK  |  Solver: ExactSolver / SimulatedAnnealingSampler")
    print("=" * 96)

    for t0 in DEPARTURE_TIMES:
        print(f"\n{'─'*96}")
        print(f"  Departure t0 = {t0} min")
        print(f"{'─'*96}")

        g = build_wang_graph()
        r = QUANTUM_ALGORITHM.run(
            g, SOURCE, DESTINATION, t0,
            penalty=100.0, method="exact"
        )

        if "error" in r:
            print(f"  ERROR: {r['error']}")
            continue

        print(f"\n  QUBO variables  : {r['num_vars']}")
        print(f"  Time-exp states : {r['num_states']}")
        print(f"  Best energy     : {r['best_energy']}")
        print(f"  Solver          : {r['method']}")

        # Top samples table
        print(f"\n  Top energy solutions (up to 30):")
        print(f"  {'Rank':>4}  {'Energy':>8}  {'Occ':>4}  "
              f"{'Variables':>30}  {'Path':<35}  {'Cost':>8}")
        print(f"  {'-'*4}  {'-'*8}  {'-'*4}  "
              f"{'-'*30}  {'-'*35}  {'-'*8}")

        for s in r["top_samples"]:
            marker = " ◄" if s["is_best"] else ""
            var_str = str(s["variables"])[:28]
            path_str = s["path"][:33]
            cost_str = f"{s['cost']:.2f}" if s["cost"] else "—"
            print(f"  {s['rank']:>4}  {s['energy']:>8.1f}  {s['num_occ']:>4}  "
                  f"{var_str:>30}  {path_str:<35}  {cost_str:>8}{marker}")

        print(f"\n  ► Shortest Path: {fmt_path(r['path'])}  "
              f"[{r['travel_time']:.2f} min]")


if __name__ == "__main__":
    run_classical()
    run_quantum()
    print("\nBenchmark complete.")
