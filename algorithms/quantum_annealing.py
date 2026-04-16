# quantum_annealing.py
# TDSPP via Quantum Annealing — QUBO formulation.
# Logic follows the original notebook exactly:
#   - Step function for edge costs (not linear interpolation)
#   - BFS time-expanded graph with absolute horizon
#   - Same constraint encoding (H_source, H_dest, H_flow)
#   - Same path decoder logic
# Only additions: module interface (run function, top_samples for GUI)

ALGORITHM_NAME = "Quantum Annealing"
ALGORITHM_YEAR = "D-Wave QUBO, 2024"
COLOR          = "#7b1fa2"

try:
    from dimod import BinaryQuadraticModel, ExactSolver
    from dwave.samplers import SimulatedAnnealingSampler
    _DWAVE_AVAILABLE = True
except ImportError:
    _DWAVE_AVAILABLE = False


# ── Edge cost — step function (exactly as in notebook) ───────────────────

def w_of_t(edge_weights, edge, t):
    """
    Return the weight of edge=(u,v) when departing at time t.
    Uses a step function: scans breakpoints left to right,
    returns the last value where t >= breakpoint.
    Assumes breakpoints are sorted increasingly.
    """
    pw = edge_weights.get(edge) or edge_weights.get((edge[1], edge[0]))
    if pw is None:
        return float('inf')
    current = pw[0][1]
    for bp, wt in pw:
        if t >= bp:
            current = wt
        else:
            break
    return current


# ── Time-expanded graph (exactly as in notebook) ──────────────────────────

def build_time_expanded_graph(edge_weights, adj, source, dest, t0, horizon=100):
    """
    Build the reachable time-expanded graph from (source, t0).
    Returns:
      states      — set of reachable states (node, time)
      transitions — list of directed transitions [((u,t), (v,t2), cost), ...]
    """
    states      = set()
    transitions = []

    frontier = [(source, t0)]
    states.add((source, t0))

    while frontier:
        u, t = frontier.pop(0)

        if u == dest:
            continue

        for v in adj.get(u, []):
            cost = w_of_t(edge_weights, (u, v), t)
            t2   = t + cost

            if t2 <= horizon:
                s1 = (u, t)
                s2 = (v, t2)
                transitions.append((s1, s2, cost))
                if s2 not in states:
                    states.add(s2)
                    frontier.append(s2)

    return states, transitions


# ── Constraint builder (exactly as in notebook) ───────────────────────────

def add_squared_constraint(Q, offset, coeffs, rhs, penalty):
    """
    coeffs: dict {var_index: coefficient}
    Adds penalty * (sum coeffs[i]*x_i - rhs)^2 to Q and offset.
    """
    vars_list = sorted(coeffs.keys())

    # diagonal / linear terms
    for i in vars_list:
        ai = coeffs[i]
        Q[(i, i)] = Q.get((i, i), 0.0) + penalty * (ai * ai - 2.0 * rhs * ai)

    # quadratic terms
    for p in range(len(vars_list)):
        for q in range(p + 1, len(vars_list)):
            i, j   = vars_list[p], vars_list[q]
            ai, aj = coeffs[i], coeffs[j]
            Q[(i, j)] = Q.get((i, j), 0.0) + penalty * (2.0 * ai * aj)

    # constant offset
    offset += penalty * (rhs * rhs)
    return offset


# ── QUBO builder (exactly as in notebook) ────────────────────────────────

def build_qubo_for_t0(edge_weights, adj, source='v0', dest='v8',
                      t0=0, penalty=100.0):
    """
    Build QUBO for one fixed departure time t0.

    Variables:
      One binary variable for each transition in the reachable
      time-expanded graph.

    Objective:
      Minimize total travel time.

    Constraints:
      (a) source state has one more outgoing than incoming
      (b) all destination states together satisfy destination balance
      (c) every intermediate state has equal incoming/outgoing flow
    """
    states, transitions = build_time_expanded_graph(
        edge_weights, adj, source, dest, t0)

    var_to_transition = {}
    transition_to_var = {}
    for idx, tr in enumerate(transitions):
        transition_to_var[tr] = idx
        var_to_transition[idx] = tr

    Q      = {}
    offset = 0.0

    # ── H_cost: minimize total travel time ───────────────────────────────
    for idx, tr in var_to_transition.items():
        _, _, cost = tr
        Q[(idx, idx)] = Q.get((idx, idx), 0.0) + float(cost)

    # ── Build incoming / outgoing transition lists per state ──────────────
    outgoing = {s: [] for s in states}
    incoming = {s: [] for s in states}
    for idx, tr in var_to_transition.items():
        s1, s2, _ = tr
        outgoing[s1].append(idx)
        incoming[s2].append(idx)

    source_state = (source, t0)
    dest_states  = [s for s in states if s[0] == dest]

    # ── (A) Source constraint: sum_out - sum_in = 1 ───────────────────────
    coeffs = {}
    for idx in outgoing.get(source_state, []):
        coeffs[idx] = coeffs.get(idx, 0.0) + 1.0
    for idx in incoming.get(source_state, []):
        coeffs[idx] = coeffs.get(idx, 0.0) - 1.0
    offset = add_squared_constraint(Q, offset, coeffs, rhs=1.0, penalty=penalty)

    # ── (B) Destination constraint: sum_out - sum_in = -1 ─────────────────
    coeffs = {}
    for ds in dest_states:
        for idx in outgoing.get(ds, []):
            coeffs[idx] = coeffs.get(idx, 0.0) + 1.0
        for idx in incoming.get(ds, []):
            coeffs[idx] = coeffs.get(idx, 0.0) - 1.0
    offset = add_squared_constraint(Q, offset, coeffs, rhs=-1.0, penalty=penalty)

    # ── (C) Flow conservation for intermediate states ─────────────────────
    for s in states:
        if s == source_state:
            continue
        if s[0] == dest:
            continue
        coeffs = {}
        for idx in outgoing.get(s, []):
            coeffs[idx] = coeffs.get(idx, 0.0) + 1.0
        for idx in incoming.get(s, []):
            coeffs[idx] = coeffs.get(idx, 0.0) - 1.0
        if coeffs:
            offset = add_squared_constraint(
                Q, offset, coeffs, rhs=0.0, penalty=penalty)

    return Q, offset, var_to_transition, states, transitions


# ── Path decoder (exactly as in notebook) ────────────────────────────────

def decode_path(sample, var_to_transition, source='v0', t0=0):
    """
    Given the best binary assignment, extract transitions with x=1
    and follow them from the source state.
    """
    selected = []
    for idx, val in sample.items():
        if val == 1:
            selected.append(var_to_transition[idx])

    # build map from start-state -> transition
    next_map = {}
    for tr in selected:
        s1, s2, cost = tr
        next_map[s1] = (s2, cost)

    # follow path
    path_states   = []
    path_vertices = []
    total_cost    = 0

    current = (source, t0)
    visited = set()

    while current in next_map and current not in visited:
        visited.add(current)
        path_states.append(current)
        if not path_vertices:
            path_vertices.append(current[0])

        nxt, c = next_map[current]
        total_cost += c
        path_vertices.append(nxt[0])
        current = nxt

    path_states.append(current)

    return selected, path_states, path_vertices, total_cost


# ── Module interface (run + top_samples for GUI/benchmark) ───────────────

def run(graph, source, destination, departure_time,
        penalty=100.0, num_reads=2000, method="exact"):
    """
    Solve Q(source, destination, departure_time) via QUBO/quantum annealing.
    Wraps the notebook logic for use in benchmark.py and gui.py.

    Returns standard result dict plus:
      top_samples — list of top 30 solutions {rank, energy, path, cost, ...}
      num_vars    — number of QUBO variables
      num_states  — number of time-expanded states
    """
    if not _DWAVE_AVAILABLE:
        return {
            "travel_time": float('inf'), "path": [],
            "nodes_settled": 0, "nodes_relaxed": 0,
            "error": "dimod / dwave-samplers not installed. "
                     "Run: pip install dimod dwave-samplers",
            "top_samples": [], "num_vars": 0, "num_states": 0,
        }

    # Build string-keyed edge weights and adjacency from graph object.
    # IMPORTANT: graph.edges stores both (u,v) and (v,u) for each edge.
    # The notebook uses forward-only adjacency (matching the original edge
    # definitions). We reconstruct this by only taking edges where u < v
    # (the canonical direction), matching how add_edge was called in graph.py.
    edge_weights = {}
    adj = {}
    seen = set()
    for (u, v), plf in graph.edges.items():
        # Store both directions in edge_weights for lookup
        edge_weights[(f"v{u}", f"v{v}")] = plf
        # Only add forward direction to adj (u < v = canonical direction)
        key = (min(u,v), max(u,v))
        if key not in seen:
            seen.add(key)
            adj.setdefault(f"v{u}", []).append(f"v{v}")

    src_str = f"v{source}"
    dst_str = f"v{destination}"

    Q, offset, var_to_transition, states, transitions = build_qubo_for_t0(
        edge_weights, adj,
        source=src_str, dest=dst_str,
        t0=departure_time, penalty=penalty
    )

    bqm = BinaryQuadraticModel.from_qubo(Q, offset=offset)

    # ── Solve ─────────────────────────────────────────────────────────────
    if method == "exact":
        sampler   = ExactSolver()
        sampleset = sampler.sample(bqm)
    else:
        sampler   = SimulatedAnnealingSampler()
        sampleset = sampler.sample(bqm, num_reads=num_reads)

    best_sample = sampleset.first.sample
    best_energy = sampleset.first.energy

    # ── Decode best path ──────────────────────────────────────────────────
    _, _, path_vertices, total_cost = decode_path(
        best_sample, var_to_transition,
        source=src_str, t0=departure_time
    )

    path_int = [int(n[1:]) for n in path_vertices]

    # ── Top-30 sample list for GUI ────────────────────────────────────────
    top_samples = []
    for rank, row in enumerate(
            sampleset.data(['sample', 'energy', 'num_occurrences'])):
        if rank >= 30:
            break
        _, _, p_verts, p_cost = decode_path(
            row.sample, var_to_transition,
            source=src_str, t0=departure_time
        )
        top_samples.append({
            "rank":      rank,
            "energy":    round(row.energy, 2),
            "num_occ":   row.num_occurrences,
            "variables": sorted([i for i, v in row.sample.items() if v == 1]),
            "path":      " → ".join(p_verts) if p_verts else "—",
            "cost":      p_cost,
            "is_best":   rank == 0,
        })

    return {
        "travel_time":   total_cost,
        "path":          path_int,
        "nodes_settled": len(var_to_transition),
        "nodes_relaxed": 0,
        "top_samples":   top_samples,
        "num_vars":      len(var_to_transition),
        "num_states":    len(states),
        "best_energy":   round(best_energy, 2),
        "method":        method,
    }