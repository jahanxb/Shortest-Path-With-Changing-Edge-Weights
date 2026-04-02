# 📊 Shortest Path Algorithm Benchmark & Test Cases
> **Purpose:** Define breaking test cases, complexity analysis, and small/large benchmark suites for all algorithms dealing with dynamic, negative, and changing edge weights.

---

## 📋 Table of Contents
1. [Codebase & How to Run](#0-codebase--how-to-run)
2. [Algorithm Complexity Reference](#1-algorithm-complexity-reference)
3. [Breaking Test Cases Per Algorithm](#2-breaking-test-cases-per-algorithm)
4. [Small Edge Test Cases (Benchmark Suite A)](#3-small-edge-test-cases-benchmark-suite-a)
5. [Large Edge Test Cases (Benchmark Suite B)](#4-large-edge-test-cases-benchmark-suite-b)
6. [Dynamic Weight Change Test Cases (Benchmark Suite C)](#5-dynamic-weight-change-test-cases-benchmark-suite-c)
7. [Negative Weight & Negative Cycle Test Cases (Benchmark Suite D)](#6-negative-weight--negative-cycle-test-cases-benchmark-suite-d)
8. [Benchmark Scoring Matrix](#7-benchmark-scoring-matrix)

---

## 0. Codebase & How to Run

### Project Layout

```
algorithms/                   ← Algorithm logic (separated from UI)
  __init__.py                 ← Public API: from algorithms import Graph, DijkstraRerun, …
  graph.py                    ← Shared Graph data structure (adj + radj dicts)
  dijkstra.py                 ← DijkstraRerun           — baseline, non-negative only
  bellman_ford.py             ← BellmanFordRerun        — baseline, handles negatives
                              ← DynamicBellmanFord      — incremental, handles negatives
  ramalingam_reps.py          ← RamalingamReps (RR-SSSP)— dynamic, O(k log V)/update
  lpa_star.py                 ← LPAStar                 — dynamic, O(k log k)/update
  quantum_sssp.py             ← QuantumSSSP             — stub, replace _quantum_run()

dynamic_sp_benchmark.py       ← CLI runner: 4 scenarios, correctness + timing tables
sp_visualizer.py              ← PyQt5 GUI: live charts, per-algo panels, summary view
BENCHMARK.md                  ← This file
requirements.txt              ← numpy, matplotlib, PyQt5
readme.md                     ← Setup & full codebase walkthrough
```

### Running the Benchmark (CLI)

```bash
pip install -r requirements.txt
python dynamic_sp_benchmark.py
```

Runs **Scenarios A–D**, prints correctness % vs ground truth and speedup bars.

### Running the Visualizer (GUI)

```bash
python sp_visualizer.py
```

Opens the PyQt5 window. Use checkboxes to enable algorithms, sliders to configure graph size and update count, then click **▶ Run Benchmark**.

> **⚠️ Note on "0 nodes visited" in GUI:** If you enable the **"Allow Negative Weights"** checkbox, algorithms that do not support negative weights (like *Dijkstra Full Rerun*, *Ramalingam-Reps*, and *LPA\**) will fail to process them. They will safely return 0 nodes to avoid crashing the visualizer. To test negative weights, properly disable incompatible algorithms and ensure only algorithms that support negative weights (like Bellman-Ford variants) are checked.

### Implemented Algorithms (in `algorithms/`)

| Class | File | Neg. Weights | Dynamic |
|---|---|---|---|
| `DijkstraRerun` | `dijkstra.py` | ❌ | ❌ (full rerun) |
| `BellmanFordRerun` | `bellman_ford.py` | ✅ | ❌ (full rerun) |
| `DynamicBellmanFord` | `bellman_ford.py` | ✅ | ✅ incremental |
| `RamalingamReps` | `ramalingam_reps.py` | ❌ | ✅ SPT repair |
| `LPAStar` | `lpa_star.py` | ❌ | ✅ g/rhs repair |
| `QuantumSSSP` | `quantum_sssp.py` | stub | stub |

---

## 1. Algorithm Complexity Reference

| Algorithm | Time (Static) | Time (Per Update) | Space | Handles Negatives | Handles Negative Cycles | Dynamic |
|---|---|---|---|---|---|---|
| **Dijkstra** | O((V+E) log V) | O((V+E) log V) full rerun | O(V) | ❌ | ❌ | ❌ |
| **Bellman-Ford** | O(VE) | O(VE) | O(V) | ✅ | ✅ (detect) | ⚠️ slow |
| **SPFA** | O(VE) worst | O(VE) worst | O(V) | ✅ | ✅ (detect) | ⚠️ avg fast |
| **Johnson's** | O(V² log V + VE) | O(VE) repotential | O(V²) | ✅ | ✅ (detect) | ⚠️ partial |
| **Floyd-Warshall** | O(V³) | O(V³) full rerun | O(V²) | ✅ | ✅ (detect) | ❌ |
| **Ramalingam-Reps** | O(k log k) update | O(k log k) | O(V+E) | ✅* | ✅* extended | ✅ |
| **D\* Lite** | O(k log k) | O(k log k) | O(V) | ❌ | ❌ | ✅ |
| **LPA\*** | O(k log k) | O(k log k) | O(V) | ❌ | ❌ | ✅ |
| **Even-Shiloach** | O(mn) total | O(mn) amortized | O(V+E) | ❌ | ❌ | ✅ decrement |
| **Δ-Stepping** | O(V + E + d/Δ · V) | N/A | O(V) | ❌ | ❌ | ❌ |
| **Potential-Dijkstra** | O((V+E) log V) | O(k log k) | O(V) | ✅ | ✅ detect | ✅ |

> `k` = number of nodes whose shortest path actually changes after an update  
> `*` = requires Johnson-style potential extension for negative weights

---

## 2. Breaking Test Cases Per Algorithm

### 🔴 2.1 Dijkstra — Breaking Cases

**What breaks it:** Any negative edge weight.

#### Break Case 1: Single Negative Edge
```
Graph:
  A --5--> B
  A --2--> C
  C --(-4)--> B   ← negative edge

Source: A
Expected shortest path to B: A→C→B = 2 + (-4) = -2
Dijkstra gives: A→B = 5  ← WRONG (greedy locks in wrong answer)
```

#### Break Case 2: Negative Edge After Relaxation
```
Graph:
  A --1--> B --1--> D
  A --3--> C --(-5)--> D

Source: A
Correct: A→C→D = 3 + (-5) = -2
Dijkstra: A→B→D = 1 + 1 = 2  ← WRONG (already settled B before seeing C→D)
```

#### Break Case 3: Dynamic Weight Decrease Creates Negative
```
Initial:
  A --10--> B --10--> C
Update: A→B weight changes to -3
Dijkstra cannot handle the update without full rerun.
Even with full rerun, negative weight breaks it.
```

---

### 🔴 2.2 Bellman-Ford — Breaking Cases

**What breaks it:** Negative cycles (no finite answer); slow on large dense graphs.

#### Break Case 1: Negative Cycle (Infinite Loop)
```
Graph:
  A --1--> B
  B --(-3)--> C
  C --1--> B   ← forms cycle B→C→B with weight (-3+1) = -2

Source: A
Result: Shortest path to any node reachable via B = -∞
Expected behavior: DETECT and report no valid path
Bug if: algorithm loops forever or returns wrong finite value
```

#### Break Case 2: Negative Cycle Not Reachable From Source
```
Graph:
  A --2--> B --3--> D  (main path)
  X --(-5)--> Y --(-5)--> X  ← negative cycle, unreachable from A

Source: A
Expected: Correctly finds A→B→D = 5 (cycle is irrelevant)
Bug if: algorithm wrongly aborts due to the unreachable cycle
```

#### Break Case 3: Dense Graph Performance
```
V = 1000 nodes
E = 499,500 edges (complete graph)
Many weight updates per second

Bellman-Ford: O(VE) = O(1000 × 499500) ≈ 500M operations per update
Expected behavior: Correct but extremely slow
Break: Timeout / memory exhaustion
```

---

### 🔴 2.3 SPFA — Breaking Cases

**What breaks it:** Adversarial graph structures cause worst-case O(VE); negative cycles cause infinite loops.

#### Break Case 1: Adversarial Queue Cycling (Worst Case)
```
Graph structure: Long chain with back-edges that force maximum re-enqueueing
  A→B→C→D→...→Z with cross-edges causing repeated relaxations

Expected: O(VE) blowup in practice
Break: Stack overflow or TLE (time limit exceeded)
```

#### Break Case 2: Negative Cycle (No Termination)
```
Same as Bellman-Ford Break Case 1.
SPFA without cycle detection loops forever.
Fix: Count enqueue frequency — if node enqueued > V times → negative cycle
```

#### Break Case 3: All Zero-Weight Edges
```
Graph: All edges have weight 0
Source: A
SPFA gives correct answer (dist=0 everywhere) BUT
may loop checking nodes redundantly without proper visited tracking.
```

---

### 🔴 2.4 Johnson's Algorithm — Breaking Cases

**What breaks it:** Negative cycles (Bellman-Ford step fails); large V makes preprocessing expensive.

#### Break Case 1: Negative Cycle Detected at Potential Step
```
Graph contains a negative cycle reachable from super-source.
Johnson's Bellman-Ford step correctly detects it.
Expected: Report "no valid path" due to negative cycle.
Bug if: Continues to compute invalid potentials and gives wrong distances.
```

#### Break Case 2: Potential Update Lag on Dynamic Changes
```
Initial edge weights computed valid potentials h(v).
Edge (u,v) weight changes: w_new such that h(u) + w_new - h(v) < 0
Potentials are now INVALID. Dijkstra will give wrong answers.
Expected: Re-trigger potential recomputation.
Bug if: Uses stale potentials without revalidation.
```

#### Break Case 3: Disconnected Graph
```
Graph: Two separate components with no edges between them
Source: A (in component 1)
Target: X (in component 2)
Expected: dist = ∞ (unreachable)
Bug if: Johnson's super-source incorrectly connects components.
```

---

### 🔴 2.5 Floyd-Warshall — Breaking Cases

**What breaks it:** Negative cycles corrupt the entire distance matrix; O(V³) too slow for large graphs.

#### Break Case 1: Negative Cycle Corrupts All-Pairs Matrix
```
If any negative cycle exists:
  d[i][i] < 0 for some i
  All distances involving nodes on/reachable from cycle = -∞
Expected: Detect d[i][i] < 0 and flag affected nodes
Bug if: Returns finite incorrect distances to/from cycle nodes
```

#### Break Case 2: Large Graph Memory Explosion
```
V = 100,000 nodes
Space needed: V² × 8 bytes = 80 GB ← impossible
Expected: System runs out of memory
Break: Cannot allocate distance matrix
```

#### Break Case 3: Edge Weight Update Performance
```
Single edge weight changes.
Floyd-Warshall must rerun entirety: O(V³)
For V = 10,000: 10^12 operations per update ← completely infeasible
Expected: Correct answer after very long time
Break: Timeout
```

---

### 🔴 2.6 D\* Lite / LPA\* — Breaking Cases

**What breaks it:** Negative edge weights; graphs without an admissible heuristic.

#### Break Case 1: Negative Weight Edges
```
D* Lite uses a priority queue with costs.
Negative edges violate the non-decreasing cost assumption.
Expected: Wrong path is returned or priority queue behaves incorrectly.
```

#### Break Case 2: Non-Admissible Heuristic
```
If h(v) > actual_distance(v, goal) for any node v:
  A*/D* may skip the true shortest path
Expected: Sub-optimal path returned
```

#### Break Case 3: Massive Simultaneous Edge Updates
```
10,000 edges change weight simultaneously.
D* Lite processes updates one at a time.
Expected: O(10000 × k log k) — massive overhead
Break: Much slower than simply re-running from scratch
```

---

### 🔴 2.7 Ramalingam-Reps — Breaking Cases

#### Break Case 1: Cascading Updates (Large k)
```
A single edge weight change causes ALL nodes to update their distance.
k = V (worst case), so O(V log V) per update — same as Dijkstra rerun.
Example: Source edge weight changes in a chain graph of 1M nodes.
```

#### Break Case 2: Negative Weights Without Potential Extension
```
Base Ramalingam-Reps assumes non-negative weights.
Negative edge introduced → algorithm produces incorrect shortest path tree.
Fix requires Johnson-style potential reweighting extension.
```

---

## 3. Small Edge Test Cases (Benchmark Suite A)

> **Target:** V ≤ 10, E ≤ 20. Verify correctness on all algorithms.

---

### Test A-01: Simple Linear Chain
```
Nodes: A B C D E
Edges: A→B(1), B→C(2), C→D(3), D→E(4)
Source: A
Expected distances: B=1, C=3, D=6, E=10
Type: Basic correctness
```

### Test A-02: Triangle with Shortcut
```
Nodes: A B C
Edges: A→B(10), A→C(1), C→B(1)
Source: A
Expected: B=2 (via C), C=1
Type: Greedy trap (tests Dijkstra's priority handling)
```

### Test A-03: Single Negative Edge (Non-Cycle)
```
Nodes: A B C D
Edges: A→B(5), A→C(2), C→D(-1), B→D(1)
Source: A
Expected: D=1 (via C→D), B=5
Type: Negative edge, no cycle
Breaks: Dijkstra ❌
Passes: Bellman-Ford ✅, SPFA ✅, Johnson's ✅
```

### Test A-04: Negative Cycle Reachable from Source
```
Nodes: A B C
Edges: A→B(1), B→C(-3), C→B(1)
Source: A
Expected: Negative cycle detected, no valid SSSP
Type: Negative cycle
```

### Test A-05: Negative Cycle NOT Reachable from Source
```
Nodes: A B C X Y
Edges: A→B(2), B→C(3), X→Y(-10), Y→X(4)
Source: A
Expected: B=2, C=5. X,Y unreachable (cycle irrelevant)
Type: Isolated negative cycle
```

### Test A-06: Zero-Weight Edges
```
Nodes: A B C D
Edges: A→B(0), B→C(0), C→D(0), A→D(1)
Source: A
Expected: D=0 (via chain), not 1
Type: Zero weights (tests priority queue tie-breaking)
```

### Test A-07: Disconnected Graph
```
Nodes: A B C | X Y Z (two components)
Edges: A→B(1), B→C(2) | X→Y(3), Y→Z(4)
Source: A
Expected: B=1, C=3, X=∞, Y=∞, Z=∞
Type: Partial reachability
```

### Test A-08: Complete Graph (K5) with Mixed Weights
```
Nodes: 1 2 3 4 5 (all pairs connected, 20 directed edges)
Edge weights: mix of +, -, 0
Source: Node 1
Verify: All-pairs distances consistent
Type: Dense small graph stress test
```

### Test A-09: Self-Loop
```
Nodes: A B
Edges: A→A(5), A→B(3), B→B(-1)
Source: A
Expected: Self-loop on B is negative but B→B is a negative cycle
         A→B = 3
Type: Self-loop edge cases
```

### Test A-10: Single Node, No Edges
```
Nodes: A
Edges: none
Source: A
Expected: dist[A] = 0, no paths elsewhere
Type: Trivial edge case / empty graph
```

---

## 4. Large Edge Test Cases (Benchmark Suite B)

> **Target:** V from 1,000 to 1,000,000. Measures performance and scalability.

---

### Test B-01: Large Sparse Chain
```
V = 100,000 nodes (0 → 1 → 2 → ... → 99,999)
E = 99,999 edges, all weight = 1
Source: 0, Target: 99,999
Expected distance: 99,999
Type: Linear scalability test
Expect fast: D* Lite, Ramalingam-Reps, SPFA
Expect slow: Floyd-Warshall ❌ (infeasible)
```

### Test B-02: Large Dense Random Graph
```
V = 5,000
E = ~2,500,000 (dense, random weights in [1, 100])
Source: 0
Expected: Run all algorithms, compare results
Type: Dense graph — SSSP performance at scale
Bottleneck: Floyd-Warshall O(V³) = 125B ops ← avoid or timeout
```

### Test B-03: Grid Graph (Road Network Simulation)
```
V = 1,000 × 1,000 = 1,000,000 nodes
E = ~4,000,000 edges (4-directional grid)
Weights: Random [1, 50] simulating road lengths
Source: Top-left (0,0), Target: Bottom-right (999,999)
Expected: Shortest grid path
Type: Real-world road network simulation
Best algorithms: Δ-Stepping, A*, D* Lite
```

### Test B-04: Random Graph with Negative Edges (No Cycles)
```
V = 10,000
E = 50,000
~10% edges have negative weights (randomly assigned, no negative cycles guaranteed)
Source: 0
Expected: Valid SSSP from all negative-capable algorithms
Type: Large scale negative edge handling
Breaks: Dijkstra, D* Lite, LPA*, Even-Shiloach
```

### Test B-05: Scale-Free Graph (Power Law / Social Network)
```
V = 50,000
E = 500,000
Generated by Barabási–Albert model (few hubs with many edges)
Weights: [1, 1000]
Source: highest-degree hub node
Type: Social/web network topology test
Expected: Very fast query times due to hub structure
```

### Test B-06: Worst-Case Dijkstra (Dense, Many Same-Weight Edges)
```
V = 10,000
E = 99,990,000 (near-complete graph)
All weights = 1
Type: Forces maximum priority queue operations
Expected: Heap exhaustion / TLE for non-optimized Dijkstra
```

### Test B-07: Complete Bipartite Graph
```
V = 2000 (1000 on each side)
E = 1,000,000 (all left nodes connected to all right nodes)
Weights: Random [1, 100]
Source: Left node 0
Type: Bipartite structure scalability
```

### Test B-08: Negative Weight Large Graph (Bellman-Ford Stress)
```
V = 50,000
E = 200,000
All edge weights in [-100, 100]
No negative cycles guaranteed
Source: 0
Expected: Correct SSSP
Type: Stresses Bellman-Ford's O(VE) = 10B operations
Baseline metric: Time for Bellman-Ford vs SPFA vs Johnson's
```

### Test B-09: Layered DAG (Topological Order)
```
V = 500,000 nodes in 1000 layers of 500 nodes each
E = ~2,500,000 (each node connects to ~5 in next layer)
Weights: Mix of positive and negative (DAG = no negative cycles)
Source: Layer 0
Type: Large DAG — can use topological sort + relaxation for O(V+E)
Expected: Fastest correct answer via topological sort approach
```

### Test B-10: Adversarial SPFA Graph
```
V = 1,000
Constructed to maximize SPFA re-enqueue operations
E = ~50,000 back-edges that repeatedly improve distances
Type: Worst-case SPFA killer graph
Expected: SPFA approaches O(VE) — compare actual time to Bellman-Ford
```

---

## 5. Dynamic Weight Change Test Cases (Benchmark Suite C)

> **Target:** Tests algorithms that handle edge weight changes over time

---

### Test C-01: Single Edge Decrease (Small Graph)
```
Graph: A→B(10), B→C(5), A→C(20)
Source: A
Initial distances: B=10, C=15

Update: A→C weight: 20 → 3
Expected new distances: C=3 (shortcut found)
Measures: How fast algorithm propagates the improvement
```

### Test C-02: Single Edge Increase (Small Graph)
```
Graph: A→B(2), B→C(3), A→C(10)
Source: A
Initial: B=2, C=5

Update: B→C weight: 3 → 15
Expected new: C=10 (must find alternate path)
Measures: How fast algorithm abandons invalidated path
```

### Test C-03: Alternating Weight Changes (Oscillation)
```
Graph: 100 nodes in a ring
One edge oscillates between weight 1 and weight 100 every round
Run 1000 rounds of changes
Measures: Cumulative overhead of repeated dynamic updates
Best: Ramalingam-Reps, D* Lite
Worst: Algorithms requiring full rerun
```

### Test C-04: Cascade Update (High k Value)
```
Graph: Star topology — Source connected to all 10,000 leaf nodes
Source→leaf_i = weight_i

Update: Source→leaf_0 weight increases dramatically
Expected: Only leaf_0 distance changes (k=1)
Update: Source→leaf_0 weight decreases to 0.0001
Expected: If leaf_0 was relay to others, cascades through graph

Measures: k sensitivity of incremental algorithms
```

### Test C-05: Massive Simultaneous Updates
```
Graph: 100,000 nodes, 500,000 edges
Simultaneous update of 10,000 edges
Expected: All algorithms give correct final state
Measures: Batch update capability
Best approach: Process all changes, then run one SSSP pass
```

### Test C-06: Edge Addition (New Edge Appears)
```
Graph: A→B(5), B→C(5)
Source: A
Initial: C=10

New edge added: A→C(3)
Expected: C=3
Measures: Edge insertion handling
```

### Test C-07: Edge Deletion (Edge Removed)
```
Graph: A→B(1), B→C(1), A→C(10)
Source: A
Initial: C=2 (via B)

Edge B→C deleted
Expected: C=10 (must fall back to A→C)
Measures: Edge deletion — hardest dynamic case
```

### Test C-08: Weight Flips Positive↔Negative
```
Graph: A→B(5), B→C(-2)
Source: A
Initial: C=3

Update: B→C changes to +8 then back to -2 repeatedly
Measures: Algorithms handling sign changes in weights
Breaks: D* Lite, LPA*, Dijkstra-based
```

### Test C-09: Real-Time Stream of 1M Updates
```
Graph: 10,000-node road network simulation
Stream of 1,000,000 weight change events at ~100 updates/sec
Measure: Latency of each query answer after each update
Target: <10ms per query
Best: D* Lite, LPA*, Ramalingam-Reps
```

### Test C-10: Update on Bridge Edge (Cut Edge)
```
Graph: Two clusters of 500 nodes connected by a single bridge edge
Bridge weight changes from 1 → 1,000,000
Expected: All cross-cluster distances increase dramatically
Measures: How efficiently algorithm handles structural bottlenecks
```

---

## 6. Negative Weight & Negative Cycle Test Cases (Benchmark Suite D)

> **Target:** Correctness under negative weights and cycle detection.

---

### Test D-01: Simple Negative Edge Chain
```
A→B(3), B→C(-5), C→D(2)
Source: A
Expected: B=3, C=-2, D=0
Fails: Dijkstra
Passes: Bellman-Ford, SPFA, Johnson's
```

### Test D-02: All Negative Edges (No Cycles)
```
DAG: A→B(-1), B→C(-2), C→D(-3)
Source: A
Expected: B=-1, C=-3, D=-6
Note: Must process in topological order
```

### Test D-03: Negative Edge Bypasses Longer Positive Path
```
A→B(100), A→C(10), C→D(-50), D→B(1)
Source: A
Expected: B = min(100, 10 + (-50) + 1) = -39
Fails: Dijkstra (already settled B=100 by the time it sees -39)
```

### Test D-04: Reachable Negative Cycle
```
A→B(2), B→C(-5), C→B(1)
Source: A
Expected: Negative cycle B→C→B = -4. Report INVALID, no finite path.
All compliant algorithms must: detect and report this.
```

### Test D-05: Unreachable Negative Cycle
```
A→B(2), B→C(3)        (main component)
X→Y(-10), Y→Z(3), Z→X(2)  (separate cycle, unreachable from A)
Source: A
Expected: B=2, C=5. X,Y,Z=∞. Cycle does NOT affect answer.
Bug check: Algorithm shouldn't abort due to unreachable cycle.
```

### Test D-06: Near-Zero Negative Edge
```
A→B(0.0001), A→C(1), C→B(-0.9998)
Source: A
Expected: B = 1 + (-0.9998) = 0.0002 (via C is shorter)
Type: Floating point precision test
```

### Test D-07: Negative Weight on Back Edge in DFS Tree
```
Graph inducing DFS tree where negative weight is on a back-edge
Tests whether algorithm correctly handles relaxation order
Expected: Correct SSSP despite non-tree structure of negative edges
```

### Test D-08: Negative Cycle Added Dynamically
```
Initially valid graph with no negative cycles.
Dynamic update: An edge weight change creates a new negative cycle.
Expected: Algorithm detects cycle after update and reports invalidation.
Measures: Reactive negative cycle detection in dynamic algorithms.
```

### Test D-09: Negative Cycle Removed Dynamically
```
Graph has a negative cycle initially.
Dynamic update: Edge weight change eliminates the negative cycle.
Expected: Algorithm re-enables valid path computation.
Measures: Recovery from negative cycle state.
```

### Test D-10: Large Graph with 1% Negative Edges
```
V = 100,000, E = 500,000
5,000 edges have negative weights, carefully placed to avoid cycles
Source: 0
Expected: Correct SSSP
Measures: Negative-capable algorithm performance at scale
Baseline: SPFA time vs Johnson's vs Bellman-Ford
```

---

## 7. Benchmark Scoring Matrix

> Use this matrix to score each algorithm after running all suites.

### Scoring Criteria
| Score | Meaning |
|---|---|
| ✅ **PASS** | Correct result within time limit |
| ⚠️ **SLOW** | Correct result but exceeds time budget |
| ❌ **FAIL** | Wrong result or crash |
| 💀 **BREAK** | Infinite loop, OOM, or undefined behavior |

### Time Budgets
| Suite | Graph Size | Time Budget per test |
|---|---|---|
| Suite A (Small) | V≤10 | < 1ms |
| Suite B (Large) | V≤1M | < 30s |
| Suite C (Dynamic) | V≤100k | < 100ms per update |
| Suite D (Negative) | V≤100k | < 10s |

---

### Master Benchmark Table (Fill After Testing)

| Test | Dijkstra | Bellman-Ford | SPFA | Johnson's | Floyd-Warshall | Ramalingam-Reps | D\* Lite | LPA\* |
|---|---|---|---|---|---|---|---|---|
| A-01 | | | | | | | | |
| A-02 | | | | | | | | |
| A-03 | | | | | | | | |
| A-04 | | | | | | | | |
| A-05 | | | | | | | | |
| A-06 | | | | | | | | |
| A-07 | | | | | | | | |
| A-08 | | | | | | | | |
| A-09 | | | | | | | | |
| A-10 | | | | | | | | |
| B-01 | | | | | | | | |
| B-02 | | | | | | | | |
| B-03 | | | | | | | | |
| B-04 | | | | | | | | |
| B-05 | | | | | | | | |
| B-06 | | | | | | | | |
| B-07 | | | | | | | | |
| B-08 | | | | | | | | |
| B-09 | | | | | | | | |
| B-10 | | | | | | | | |
| C-01 | | | | | | | | |
| C-02 | | | | | | | | |
| C-03 | | | | | | | | |
| C-04 | | | | | | | | |
| C-05 | | | | | | | | |
| C-06 | | | | | | | | |
| C-07 | | | | | | | | |
| C-08 | | | | | | | | |
| C-09 | | | | | | | | |
| C-10 | | | | | | | | |
| D-01 | | | | | | | | |
| D-02 | | | | | | | | |
| D-03 | | | | | | | | |
| D-04 | | | | | | | | |
| D-05 | | | | | | | | |
| D-06 | | | | | | | | |
| D-07 | | | | | | | | |
| D-08 | | | | | | | | |
| D-09 | | | | | | | | |
| D-10 | | | | | | | | |

---

### Expected Known Results (Pre-filled)

| Test | Dijkstra | Bellman-Ford | SPFA | Johnson's | Floyd-Warshall | Ramalingam-Reps | D\* Lite |
|---|---|---|---|---|---|---|---|
| A-03 (negative edge) | ❌ FAIL | ✅ PASS | ✅ PASS | ✅ PASS | ✅ PASS | ✅ PASS | ❌ FAIL |
| A-04 (neg cycle) | 💀 BREAK | ✅ detect | ✅ detect | ✅ detect | ✅ detect | ✅ detect | 💀 BREAK |
| B-03 (1M grid) | ⚠️ SLOW | 💀 OOM | ⚠️ SLOW | ⚠️ SLOW | 💀 OOM | ✅ PASS | ✅ PASS |
| B-04 (neg edges large) | ❌ FAIL | ⚠️ SLOW | ✅ PASS | ✅ PASS | 💀 OOM | ✅ PASS | ❌ FAIL |
| C-08 (sign flip) | ❌ FAIL | ✅ PASS | ✅ PASS | ✅ PASS | ✅ PASS | ✅ PASS | ❌ FAIL |
| D-04 (neg cycle) | 💀 BREAK | ✅ detect | ✅ detect | ✅ detect | ✅ detect | ✅ detect | 💀 BREAK |

---

## Appendix: Key Formulas

### Bellman-Ford Negative Cycle Detection
```
After V-1 relaxations:
For each edge (u,v,w):
  if dist[u] + w < dist[v]:
    → Negative cycle exists
```

### Johnson's Potential Validity Check
```
For edge (u,v,w), potentials are valid if:
  h(u) + w - h(v) >= 0
If violated after weight update → must recompute potentials
```

### SPFA Cycle Detection
```
Track enqueue_count[v]
If enqueue_count[v] > V:
  → Negative cycle exists
```

### Complexity Quick Reference
```
Dijkstra:        O((V + E) log V)     ← fastest for positive weights
Bellman-Ford:    O(V × E)             ← correct for negatives, slow
SPFA:            O(V × E) worst       ← faster than BF in practice
Johnson's:       O(VE + V² log V)     ← all-pairs with negatives
Floyd-Warshall:  O(V³)                ← all-pairs, small graphs only
D* Lite:         O(k log k) / update  ← dynamic, no negatives
Ramalingam-Reps: O(k log k) / update  ← dynamic, with extension for neg
```

---

*Document version: 1.0 | Status: Ready for Implementation Phase*
