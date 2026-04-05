# Shortest Path With Changing Edge Weights

A research-grade benchmarking and visualization suite for **dynamic shortest-path algorithms** - algorithms that efficiently update shortest paths as edge weights change (increase, decrease, or go negative) without recomputing from scratch every time.

---

## Project Structure

```
├── algorithms/                  #   Algorithm package (logic separated from UI)
│   ├── __init__.py              #   Public API — import everything from here
│   ├── graph.py                 #   Shared Graph data structure
│   ├── dijkstra.py              #   Dijkstra Full Rerun (baseline)
│   ├── bellman_ford.py          #   Bellman-Ford Full Rerun + Dynamic variant
│   ├── ramalingam_reps.py       #   Ramalingam-Reps dynamic SSSP (RR-SSSP)
│   ├── lpa_star.py              #   LPA* — Lifelong Planning A*
│   └── quantum_sssp.py          #   Quantum SSSP stub (plug-in point)
│
├── dynamic_sp_benchmark.py      # CLI benchmark runner (Scenarios A-D)
├── sp_visualizer.py             # PyQt5 GUI with live charts
├── BENCHMARK.md                 # Full test-case catalogue & scoring matrix
├── requirements.txt             # Python dependencies
└── readme.md                    # This file
```

---

## Algorithms Implemented

| Algorithm | Type | Handles Negatives | Per-Update Cost |
|---|---|---|---|
| **Dijkstra Full Rerun** | Baseline | ❌ | O((V+E) log V) |
| **Bellman-Ford Full Rerun** | Baseline | ✅ | O(V · E) |
| **Dynamic Bellman-Ford** | Incremental | ✅ | O(k · E) |
| **Ramalingam-Reps (RR-SSSP)** | Fully Dynamic | ❌ | O(k log V) |
| **LPA\*** | Fully Dynamic | ❌ | O(k log V) |
| **Quantum SSSP** | Stub / Plug-in | ✅\* | O(√(V·E))\* |

> `k` = number of nodes whose shortest distance actually changes after an update.  
> \* Quantum entry is a stub backed by classical Dijkstra. See `algorithms/quantum_sssp.py` for the integration guide.

---

## Setup & Installation

### 1. Create a virtual environment (recommended)

```bash
python3 -m venv .venv
source .venv/bin/activate        # macOS / Linux
# .venv\Scripts\activate         # Windows
```

### 2. Install dependencies

```bash
pip install -r requirements.txt
```

---

## Running the Code

### CLI Benchmark (no GUI required)

Runs all 5 algorithms across 4 scenarios and prints a full comparison table:

```bash
python dynamic_sp_benchmark.py
```

**Output includes:**
- Per-algorithm timing (total ms, average ms per update)
- Nodes visited (average and max per update)
- Correctness % vs Dijkstra ground truth
- Speedup bar chart vs the Dijkstra baseline

### GUI Visualizer (interactive)

Opens a PyQt5 window with live charts, algorithm toggles, and a summary dashboard:

```bash
python sp_visualizer.py
```

**GUI features:**
- Checkboxes to enable/disable individual algorithms
- Sliders for node count, edge count, and number of updates
- Dropdown for weight-change mode: mixed / decrease / increase / traffic
- Toggle for negative weights (enables BF-family only)
- Per-algorithm panels with live time and nodes-visited charts
- Summary tab: bar charts + speedup comparison vs Dijkstra

### Troubleshooting: Zero Nodes Visited

If you see **0 nodes visited** in the UI panels, it means the algorithm encountered an error, usually because of an incompatible configuration.

**Limitations & Negative Weights:**
- **Dijkstra Full Rerun, Ramalingam-Reps, and LPA\*** do **not** support negative weight edges.
- If the **"Allow Negative Weights"** checkbox is selected in the GUI while these algorithms are enabled, they will instantly fail and report 0 nodes.
- **Solution:** Either uncheck the "**Allow Negative Weights**" box, or disable the incompatible algorithms when simulating graphs with negative costs. Bellman-Ford variants are designed to correctly handle negative weights!

---

## What Is Happening in the Codebase

### `algorithms/` — The Algorithm Package

All algorithm logic lives here, **completely separated** from the GUI and runner code. Every algorithm exposes the same interface:

```python
# Initialise (runs SSSP once from scratch)
algo = DijkstraRerun(graph, source=0)

# Apply an edge weight change incrementally
nodes_visited = algo.update(u=2, v=5, w_new=3.7)

# Read current shortest distances
print(algo.dist[7])   # distance from source to node 7
```

**`algorithms/graph.py`** — `Graph` class with `add_edge`, `update_weight`, `neighbors`, `predecessors`, and `clone`. Uses two adjacency dicts (forward + reverse) so predecessor lookups are O(degree).

**`algorithms/dijkstra.py`** — Baseline. Full Dijkstra rerun (`O((V+E) log V)`) after every change. Correct only for non-negative weights.

**`algorithms/bellman_ford.py`** — Two classes:
- `BellmanFordRerun` — full rerun, `O(VE)`, handles negatives.
- `DynamicBellmanFord` — incremental; weight decreases forward-propagate via BFS, weight increases invalidate the dependent subtree then recompute.

**`algorithms/ramalingam_reps.py`** — Maintains an explicit shortest-path tree. On decrease: Dijkstra-style local relaxation. On increase: subtree invalidation + seeded priority-queue repair. `O(k log V)` per update.

**`algorithms/lpa_star.py`** — Lifelong Planning A\*. Maintains `g` (current distance) and `rhs` (one-step lookahead) per node. Processes only *inconsistent* nodes (`g ≠ rhs`). `O(k log k)` per update.

**`algorithms/quantum_sssp.py`** — Plug-in stub. Classical Dijkstra with simulated 40% node-visit reduction. Replace `_quantum_run()` to wire in a real quantum backend.

### `dynamic_sp_benchmark.py` — CLI Benchmark Runner

Imports all algorithms from `algorithms/` and runs four scenarios:

| Scenario | Graph Type | Update Mode | Notes |
|---|---|---|---|
| A | Random (Erdős-Rényi) | mixed ↑↓ | Small / Medium / Large |
| B | Grid (road network) | traffic spikes | Simulates rush hour |
| C | Layered dense | mixed | Worst-case SPT disruption |
| D | Random with negatives | mixed | BF algorithms only |

Each scenario clones the original graph for every algorithm so they all receive identical edge-weight update sequences. Correctness is verified against a fresh Dijkstra ground truth after every single update.

### `sp_visualizer.py` — PyQt5 GUI

Imports algorithms from `algorithms/` and runs benchmarks in a background `QThread` so the UI stays responsive. Emits signals per update so each algorithm's panel updates in real time.

---

## Benchmark Test Cases

See **[BENCHMARK.md](BENCHMARK.md)** for:

- Per-algorithm **breaking test cases** (exact graph structures that expose bugs or limitations)
- **Suite A** — 10 small-graph correctness tests (V ≤ 10)
- **Suite B** — 10 large-graph scalability tests (V up to 1 M)
- **Suite C** — 10 dynamic weight-change scenarios
- **Suite D** — 10 negative-weight and negative-cycle tests
- **Scoring matrix** (40 tests × 8 algorithms) to fill after running benchmarks

---

## Adding a New Algorithm

1. Create `algorithms/my_algo.py` implementing the standard interface:
   ```python
   from algorithms.graph import Graph, INF

   class MyAlgo:
       name = "My Algorithm"
       supports_negative = False  # or True

       def __init__(self, graph: Graph, source: int): ...
       def update(self, u: int, v: int, w_new: float) -> int: ...
       # .dist must be a dict[int, float]
   ```

2. Export it from `algorithms/__init__.py`:
   ```python
   from algorithms.my_algo import MyAlgo
   __all__ = [..., "MyAlgo"]
   ```

3. Add it to `ALGORITHMS` in `dynamic_sp_benchmark.py` and to `ALGO_MAP` in `sp_visualizer.py`.

---

## References

- Ramalingam, G. & Reps, T. (1996). *An incremental algorithm for a generalization of the shortest-path problem.* J. Algorithms.
- Koenig, S. & Likhachev, M. (2002). *D\* Lite.* AAAI.
- Koenig, S. & Likhachev, M. (2004). *Lifelong Planning A\*.* Artificial Intelligence.
- Dürr, C. et al. (2006). *Quantum query complexity of some graph problems.* SIAM J. Comput.
- Johnson, D. B. (1977). *Efficient algorithms for shortest paths in sparse networks.* J. ACM.
