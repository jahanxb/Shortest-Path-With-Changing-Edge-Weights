# Shortest Paths with Changing Edge Weights
**Group 2 — CS 691 Algorithm Design & Analysis**
Muhammad Jahanzeb Khan · Md Nazmul Hoque · Hassan Mahmoud
University of Alabama

---

## What This Project Does

This project benchmarks three classical algorithms that solve the **Time-Dependent Shortest Path Problem (TDSPP)** — finding the fastest route between two nodes in a road network where edge costs change over time. We also include a quantum annealing formulation of the same problem using a QUBO encoding.

The key difference from regular shortest path: in a time-dependent network, the cost of an edge is not a fixed number. It is a **piecewise linear function (PLF)** of the time you arrive at that edge. So if you leave later, the same road might be faster or slower depending on traffic conditions at that moment.

---

## Project Structure

```
├── algorithms/
│   ├── __init__.py          — registers all three algorithms
│   ├── graph.py             — graph data structure + PLF evaluation
│   ├── td_dijkstra.py       — TD-Dijkstra (baseline)
│   ├── td_astar.py          — TD-A* (heuristic-guided)
│   └── td_g_tree.py         — TD-G-Tree (index-based)
│
├── benchmark.py             — terminal benchmark, prints results table
├── gui.py                   — PyQt5 GUI with graph visualization and charts
└── README.md                — this file
```

---

## The Graph We Used

A 9-node time-dependent road network with 11 bidirectional edges. The graph has four time-varying edges whose costs change depending on when you travel, and seven fixed-cost edges that always cost the same.

```
Nodes: v0, v1, v2, v3, v4, v5, v6, v7, v8
Source: v0
Destination: v8
```

**Edge weights — (Time, Weight) breakpoints:**

| Edge | PLF Points | Type |
|---|---|---|
| e(v0, v1) | {(0,4), (60,4)} | Fixed — always 4 min |
| e(v0, v2) | {(0,8), (60,8)} | Fixed — always 8 min |
| e(v1, v2) | {(0,8), (20,8), (35,20), (60,20)} | Time-varying — rises after t=20 |
| e(v1, v6) | {(0,5), (20,5), (30,18), (60,5)} | Time-varying — **spikes at t=30** |
| e(v2, v3) | {(0,15), (60,15)} | Fixed — always 15 min |
| e(v2, v5) | {(0,5), (60,5)} | Fixed — always 5 min |
| e(v3, v4) | {(0,6), (60,6)} | Fixed — always 6 min |
| e(v3, v5) | {(0,22), (20,22), (35,6), (60,6)} | Time-varying — drops after t=20 |
| e(v5, v6) | {(0,8), (25,8), (45,12), (60,12)} | Time-varying — rises from t=25 |
| e(v6, v7) | {(0,2), (60,2)} | Fixed — always 2 min |
| e(v7, v8) | {(0,3), (60,3)} | Fixed — always 3 min |

**How PLF costs work:** The breakpoints give you the cost at specific times. For any time in between, the cost is linearly interpolated. For example, e(v1,v6) at t=34 falls between (30,18) and (60,5), so the cost is 18 + (34-30) × (5-18)/(60-30) = **16.27 min**.

**We tested three departure times:**
- **t = 0** — off-peak, all edges cheap
- **t = 30** — peak congestion, e(v1,v6) spikes to ~16.27 min at actual arrival
- **t = 50** — recovering, e(v1,v6) costs 7.6 min at actual arrival

---

## Algorithms

### TD-Dijkstra — Dreyfus, 1969
The baseline algorithm. Works exactly like standard Dijkstra except edge costs are evaluated at the actual arrival time at each node, not the original departure time. Always picks the node reachable in the least time and expands from there.

**Complexity:** O((V + E) log V · f) where f = max PLF breakpoints per edge

**On our graph:** 7–9 nodes settled per query depending on departure time.

### TD-A* — Zhao et al., 2008
Same as TD-Dijkstra but adds a precomputed lower-bound heuristic h(v) for every node — the minimum possible remaining time to the destination from that node. This steers the search toward the destination and avoids wasting time expanding nodes that are clearly going the wrong way. Because h(v) never overestimates, the algorithm still returns the exact optimal answer.

The heuristic is built once using a reverse Dijkstra with the minimum possible cost per edge. Node priority = arrival time + h(v).

**Heuristic values for our graph:**

| Node | h(v) | Reason |
|---|---|---|
| v8 | 0 | Already at destination |
| v7 | 3 | v7→v8 minimum cost |
| v6 | 5 | v6→v7→v8 |
| v1 | 7 | v1→v6→v7→v8 |
| v5 | 13 | v5→v6→v7→v8 |
| v3, v4 | high | Far from destination — pruned |

**Complexity:** O((V + E) log V · f) — same asymptotic, fewer nodes in practice

**On our graph:** Consistently settles only 5 nodes regardless of departure time.

### TD-G-Tree — Wang, Li, Tang (PVLDB 2019)
Index-based algorithm. Instead of searching the full graph at every query, it does the expensive work once offline. The graph is split into clusters, border nodes (nodes whose edges cross cluster boundaries) are identified, and travel times between all border pairs are precomputed and stored in a matrix. At query time, only border nodes are touched — internal cluster nodes are never visited.

**Our graph has 4 clusters:**

| Cluster | Nodes | Border Nodes |
|---|---|---|
| Source cluster | v0, v1, v2 | v1, v2 |
| Destination cluster | v6, v7, v8 | v6 |
| Bottom cluster | v3, v4 | v3 |
| Connector | v5 | v5 |

**Why local_dijkstra is used:** The source and destination nodes (v0 and v8) are not border nodes themselves. A short Dijkstra restricted to nodes inside a single cluster is used to travel from v0 to its cluster's border nodes, and from the destination cluster's border node v6 to v8. These internal searches are fast because each cluster is small.

**Query steps:**
1. local_dijkstra from v0 to source cluster borders v1, v2
2. Matrix lookup: best arrival at destination cluster border v6 (comparing via v1 directly, or via v2→v5→v6)
3. local_dijkstra from v6 to v8 inside destination cluster

**Complexity:** O(log²(κf) · V · log²f) per query where κf = tree fanout

**On our graph:** v3, v4 are never visited at t=0 and t=50. At t=30, v4 is never visited.

---

## Results

All three algorithms find the same optimal path for all three departure times:

**Path: v0 → v1 → v6 → v7 → v8**

| | t = 0 | t = 30 | t = 50 |
|---|---|---|---|
| **TD-Dijkstra** | 14.00 min | 25.27 min | 16.60 min |
| **TD-A*** | 14.00 min | 25.27 min | 16.60 min |
| **TD-G-Tree** | 14.00 min | 25.27 min | 16.60 min |

**Why t=30 is the hardest case:** When departing at t=30, you arrive at v1 at t=34. The edge e(v1,v6) is evaluated at t=34 (not t=30), giving 16.27 min due to the PLF spike. The alternative route via v2→v5→v6 costs 8+5+11.6=24.6 min just to reach v6, making the direct v1→v6 route still faster at 4+16.27=20.27 min.

**Nodes visited comparison:**

| | t = 0 | t = 30 | t = 50 |
|---|---|---|---|
| TD-Dijkstra | v0,v1,v2,v6,v7,v5,v8 | all 9 nodes | v0,v1,v2,v6,v5,v7,v8 |
| TD-A* | v0,v1,v6,v7,v8 only | v0,v1,v6,v7,v8 only | v0,v1,v6,v7,v8 only |
| TD-G-Tree | skips v3,v4,v5 | skips v4 | skips v3,v4 |

---

## How to Run

### Requirements

```
PyQt5
matplotlib
numpy
```

Install with:
```bash
pip install PyQt5 matplotlib numpy
```

### Terminal Benchmark

Prints a full results table for all three algorithms at t=0, t=30, t=50:

```bash
python benchmark.py
```

Output includes travel time, nodes settled, and query time in milliseconds for each algorithm and each departure time.

### GUI

Opens an interactive window with the graph visualization, bar charts, and a per-departure-time results table:

```bash
python gui.py
```

**GUI features:**
- Graph panel showing node positions, time-varying edges (dashed orange), and optimal paths highlighted after running
- Algorithm checkboxes to enable or disable individual algorithms
- Bar charts for average query time (ms) and average nodes settled
- Summary table with travel time per departure time per algorithm
- Edge Weight Functions tab showing all four PLF curves with departure time markers

---

## Algorithm Complexity on Our Graph

| Algorithm | Formula | Our graph (V=9, E=11, f=4) |
|---|---|---|
| TD-Dijkstra | O((V+E)·log V·f) | ≈ 254 operations |
| TD-A* | O((V+E)·log V·f) | ≈ 141 in practice (5 nodes) |
| TD-G-Tree | O(log²(κf)·V·log²f) | ≈ 324 (saves on large graphs) |
| Quantum (QUBO) | O(\|E\|·\|T\|) variables | 660 QUBO variables |

Note: TD-G-Tree appears more expensive on 9 nodes because its advantage is at scale. On networks with thousands of nodes and many repeated queries, the border-only traversal becomes dramatically faster than full graph search.

---

## Quantum Annealing Approach — *[PENDING]*

The quantum formulation of TDSPP encodes the problem as a QUBO (Quadratic Unconstrained Binary Optimization) using time-indexed binary variables:

```
x_{ij}^t = 1  if edge (i,j) is used at time t
x_{ij}^t = 0  otherwise
```

The full Hamiltonian is:

```
H_P = H_cost + A·H_source + A·H_dest + A·H_flow
```

Where A is a penalty coefficient that enforces path validity constraints. The quantum annealer finds the ground state of H_P, which corresponds to the optimal path.

Framework used: **D-Wave Ocean SDK** (compatible with D-Wave quantum annealers).

**Current status:** QUBO formulation is complete and verified against classical TD-Dijkstra at t=0 and t=50. At t=30 the annealer returns a suboptimal path (v0→v2→v5→v6→v7→v8, 29.6 min vs optimal 25.27 min) due to the PLF spike on e(v1,v6) creating a complex energy landscape. Full quantum implementation and hardware testing are pending.
quantum
---

## References

- Wang, Y., Li, G., & Tang, N. (2019). Querying Shortest Paths on Time Dependent Road Networks. *PVLDB*, 12(11), 1249–1261.
- Zhao, X. et al. (2008). Time-dependent heuristic search for routing in dynamic road networks.
- Dreyfus, S. E. (1969). An appraisal of some shortest-path algorithms. *Operations Research*, 17(3), 395–412.
- Papalitsas, C. et al. (2019). A QUBO model for the traveling salesman problem with time windows. *Algorithms*, 12(11), 224.
- Krauss, T. & McCollum, J. (2020). Solving the network shortest path problem on a quantum annealer. *IEEE Transactions on Quantum Engineering*, 1, 1–12.
