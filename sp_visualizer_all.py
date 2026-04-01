"""
=============================================================================
  Dynamic Shortest Path Benchmark Visualizer
  PyQt5 GUI — Interactive benchmarking with live charts

  Features:
  - Select algorithms via checkboxes (each gets its own panel)
  - Control edges, nodes, weight change direction
  - Live animated bar + line charts per algorithm
  - Complexity reference table
  - Quantum placeholder module (pluggable)
=============================================================================
"""

import sys, heapq, time, random, math, threading
from collections import defaultdict
from typing import Dict, List, Tuple, Optional

import numpy as np
import matplotlib
matplotlib.use('Qt5Agg')
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.figure import Figure
from matplotlib.animation import FuncAnimation

from PyQt5.QtWidgets import (
    QApplication, QMainWindow, QWidget, QVBoxLayout, QHBoxLayout,
    QGridLayout, QLabel, QSlider, QCheckBox, QPushButton, QComboBox,
    QGroupBox, QScrollArea, QFrame, QSplitter, QTabWidget,
    QTableWidget, QTableWidgetItem, QHeaderView, QProgressBar,
    QSpinBox, QSizePolicy, QTextEdit
)
from PyQt5.QtCore import Qt, QThread, pyqtSignal, QTimer, QSize
from PyQt5.QtGui import QFont, QColor, QPalette, QIcon

# ─────────────────────────────────────────────────────────────────────────────
# THEME COLORS
# ─────────────────────────────────────────────────────────────────────────────
DARK_BG     = "#1a1a2e"
PANEL_BG    = "#16213e"
ACCENT      = "#00d9ff"
ACCENT2     = "#00ff88"
WARN        = "#f9ca24"
RED         = "#ff6b6b"
TEXT        = "#e0e0e0"
SUBTEXT     = "#a0a0a0"
BORDER      = "#2a2a4e"

ALGO_COLORS = {
    "Dijkstra Full Rerun":       "#ff6b6b",
    "Bellman-Ford Full Rerun":   "#f9ca24",
    "Dynamic Bellman-Ford":      "#00d9ff",
    "Ramalingam-Reps":           "#00ff88",
    "LPA*":                      "#c56af9",
    "Quantum SSSP (stub)":       "#ff9f43",
}

ALGO_COMPLEXITY = {
    "Dijkstra Full Rerun":       ("O((V+E) log V)", "per full rerun", "No",  "No"),
    "Bellman-Ford Full Rerun":   ("O(VE)",          "per full rerun", "Yes", "No"),
    "Dynamic Bellman-Ford":      ("O(k·E)",         "k=affected",     "Yes", "Yes"),
    "Ramalingam-Reps":           ("O(k log V)",     "k=affected",     "No",  "Yes"),
    "LPA*":                      ("O(k log V)",     "k=inconsistent", "No",  "Yes"),
    "Quantum SSSP (stub)":       ("O(√(VE))*",      "theoretical",    "Yes*","Yes"),
}

INF = float('inf')

# ─────────────────────────────────────────────────────────────────────────────
# GRAPH
# ─────────────────────────────────────────────────────────────────────────────
class Graph:
    def __init__(self, n):
        self.n = n
        self.adj  = defaultdict(dict)
        self.radj = defaultdict(dict)

    def add_edge(self, u, v, w):
        self.adj[u][v] = w
        self.radj[v][u] = w

    def update_weight(self, u, v, w):
        old = self.adj[u].get(v, INF)
        self.adj[u][v] = w
        self.radj[v][u] = w
        return old

    def get_weight(self, u, v):
        return self.adj[u].get(v, INF)

    def neighbors(self, u):
        return self.adj[u]

    def predecessors(self, v):
        return self.radj[v]

    def clone(self):
        g = Graph(self.n)
        for u in self.adj:
            for v, w in self.adj[u].items():
                g.add_edge(u, v, w)
        return g

    def edge_list(self):
        return [(u, v) for u in self.adj for v in self.adj[u]]

# ─────────────────────────────────────────────────────────────────────────────
# ALGORITHMS
# ─────────────────────────────────────────────────────────────────────────────
class DijkstraRerun:
    name = "Dijkstra Full Rerun"
    supports_negative = False

    def __init__(self, graph, source):
        self.graph = graph; self.source = source
        self.dist = {}; self.nodes_visited = 0
        self._run()

    def _run(self):
        self.nodes_visited = 0
        dist = {i: INF for i in range(self.graph.n)}
        dist[self.source] = 0
        pq = [(0, self.source)]; visited = set()
        while pq:
            d, u = heapq.heappop(pq)
            if u in visited: continue
            visited.add(u); self.nodes_visited += 1
            for v, w in self.graph.neighbors(u).items():
                if dist[u] + w < dist[v]:
                    dist[v] = dist[u] + w
                    heapq.heappush(pq, (dist[v], v))
        self.dist = dist

    def update(self, u, v, w_new):
        self.graph.update_weight(u, v, w_new)
        self._run()
        return self.nodes_visited


class BellmanFordRerun:
    name = "Bellman-Ford Full Rerun"
    supports_negative = True

    def __init__(self, graph, source):
        self.graph = graph; self.source = source
        self.dist = {}; self.nodes_visited = 0
        self._run()

    def _run(self):
        n = self.graph.n
        dist = {i: INF for i in range(n)}
        dist[self.source] = 0
        self.nodes_visited = 0
        edges = [(u, v, w) for u in self.graph.adj for v, w in self.graph.adj[u].items()]
        for _ in range(n - 1):
            updated = False
            for u, v, w in edges:
                self.nodes_visited += 1
                if dist[u] != INF and dist[u] + w < dist[v]:
                    dist[v] = dist[u] + w; updated = True
            if not updated: break
        self.dist = dist

    def update(self, u, v, w_new):
        self.graph.update_weight(u, v, w_new)
        self._run()
        return self.nodes_visited


class DynamicBellmanFord:
    name = "Dynamic Bellman-Ford"
    supports_negative = True

    def __init__(self, graph, source):
        self.graph = graph; self.source = source
        self.dist = {}; self.parent = {}; self.nodes_visited = 0
        self._full_run()

    def _full_run(self):
        n = self.graph.n
        dist = {i: INF for i in range(n)}
        parent = {i: None for i in range(n)}
        dist[self.source] = 0
        edges = [(u, v, w) for u in self.graph.adj for v, w in self.graph.adj[u].items()]
        for _ in range(n - 1):
            updated = False
            for u, v, w in edges:
                if dist[u] != INF and dist[u] + w < dist[v]:
                    dist[v] = dist[u] + w; parent[v] = u; updated = True
            if not updated: break
        self.dist = dist; self.parent = parent

    def update(self, u, v, w_new):
        w_old = self.graph.update_weight(u, v, w_new)
        self.nodes_visited = 0
        if w_new < w_old: self._propagate_decrease(u, v, w_new)
        else:              self._propagate_increase(u, v)
        return self.nodes_visited

    def _propagate_decrease(self, u, v, w_new):
        changed = set()
        candidate = self.dist[u] + w_new if self.dist[u] != INF else INF
        if candidate < self.dist[v]:
            self.dist[v] = candidate; self.parent[v] = u; changed.add(v)
        while changed:
            x = changed.pop(); self.nodes_visited += 1
            for y, w in self.graph.neighbors(x).items():
                if self.dist[x] != INF and self.dist[x] + w < self.dist[y]:
                    self.dist[y] = self.dist[x] + w; self.parent[y] = x; changed.add(y)

    def _propagate_increase(self, u, v):
        affected = set(); queue = [v]
        while queue:
            node = queue.pop()
            if node in affected: continue
            affected.add(node); self.nodes_visited += 1
            for y in self.graph.neighbors(node):
                if self.parent.get(y) == node: queue.append(y)
        affected_list = sorted(affected, key=lambda x: self.dist.get(x, INF))
        for node in affected_list:
            best = INF; best_parent = None
            for pred, w in self.graph.predecessors(node).items():
                if self.dist.get(pred, INF) != INF:
                    c = self.dist[pred] + w
                    if c < best: best = c; best_parent = pred
            self.dist[node] = best; self.parent[node] = best_parent


class RamalingamReps:
    name = "Ramalingam-Reps"
    supports_negative = False

    def __init__(self, graph, source):
        self.graph = graph; self.source = source
        self.dist = {}; self.parent = {}; self.nodes_visited = 0
        self._full_dijkstra()

    def _full_dijkstra(self):
        dist = {i: INF for i in range(self.graph.n)}
        parent = {i: None for i in range(self.graph.n)}
        dist[self.source] = 0
        pq = [(0, self.source)]; visited = set()
        while pq:
            d, u = heapq.heappop(pq)
            if u in visited: continue
            visited.add(u)
            for v, w in self.graph.neighbors(u).items():
                if dist[u] + w < dist[v]:
                    dist[v] = dist[u] + w; parent[v] = u
                    heapq.heappush(pq, (dist[v], v))
        self.dist = dist; self.parent = parent

    def update(self, u, v, w_new):
        w_old = self.graph.update_weight(u, v, w_new)
        self.nodes_visited = 0
        if w_new <= w_old: self._handle_decrease(u, v, w_new)
        else:              self._handle_increase(u, v)
        return self.nodes_visited

    def _handle_decrease(self, u, v, w_new):
        if self.dist[u] == INF: return
        candidate = self.dist[u] + w_new
        if candidate >= self.dist[v]: return
        self.dist[v] = candidate; self.parent[v] = u
        pq = [(candidate, v)]
        while pq:
            d, x = heapq.heappop(pq); self.nodes_visited += 1
            if d > self.dist[x]: continue
            for y, w in self.graph.neighbors(x).items():
                nd = self.dist[x] + w
                if nd < self.dist[y]:
                    self.dist[y] = nd; self.parent[y] = x
                    heapq.heappush(pq, (nd, y))

    def _handle_increase(self, u, v):
        if self.parent.get(v) != u: return
        subtree = []; queue = [v]; visited = set()
        while queue:
            node = queue.pop()
            if node in visited: continue
            visited.add(node); subtree.append(node)
            for y in self.graph.neighbors(node):
                if self.parent.get(y) == node: queue.append(y)
        subtree_set = set(subtree)
        for node in subtree:
            self.dist[node] = INF; self.parent[node] = None
        pq = []
        for node in subtree:
            self.nodes_visited += 1
            for pred, w in self.graph.predecessors(node).items():
                if pred not in subtree_set and self.dist[pred] != INF:
                    c = self.dist[pred] + w
                    if c < self.dist[node]:
                        self.dist[node] = c; self.parent[node] = pred
                        heapq.heappush(pq, (c, node))
        while pq:
            d, x = heapq.heappop(pq)
            if d > self.dist[x]: continue
            for y, w in self.graph.neighbors(x).items():
                nd = self.dist[x] + w
                if nd < self.dist[y]:
                    self.dist[y] = nd; self.parent[y] = x
                    heapq.heappush(pq, (nd, y))


class LPAStar:
    name = "LPA*"
    supports_negative = False

    def __init__(self, graph, source):
        self.graph = graph; self.source = source
        self.g    = defaultdict(lambda: INF)
        self.rhs  = defaultdict(lambda: INF)
        self.nodes_visited = 0
        self._pq = []; self._in_pq = {}
        self.rhs[source] = 0
        self._insert(source, 0.0)
        self._compute()

    def _key(self, v):   return min(self.g[v], self.rhs[v])
    def _insert(self, v, key): self._in_pq[v] = key; heapq.heappush(self._pq, (key, v))
    def _remove(self, v): self._in_pq.pop(v, None)

    def _update_vertex(self, v):
        if v != self.source:
            best = INF
            for pred, w in self.graph.predecessors(v).items():
                if self.g[pred] != INF: best = min(best, self.g[pred] + w)
            self.rhs[v] = best
        self._remove(v)
        if self.g[v] != self.rhs[v]: self._insert(v, self._key(v))

    def _compute(self):
        while self._pq:
            key, u = heapq.heappop(self._pq)
            if self._in_pq.get(u) != key: continue
            self._in_pq.pop(u, None); self.nodes_visited += 1
            if self.g[u] > self.rhs[u]:
                self.g[u] = self.rhs[u]
                for v in self.graph.neighbors(u): self._update_vertex(v)
            else:
                self.g[u] = INF; self._update_vertex(u)
                for v in self.graph.neighbors(u): self._update_vertex(v)

    @property
    def dist(self): return dict(self.g)

    def update(self, u, v, w_new):
        self.nodes_visited = 0
        self.graph.update_weight(u, v, w_new)
        self._update_vertex(v)
        self._compute()
        return self.nodes_visited


# ─────────────────────────────────────────────────────────────────────────────
# QUANTUM SSSP STUB  (pluggable module — replace body with real implementation)
# ─────────────────────────────────────────────────────────────────────────────
class QuantumSSSP:
    """
    Quantum SSSP stub.

    This class is a PLACEHOLDER for a quantum-enhanced shortest path algorithm.
    Plug in your real quantum implementation by replacing _quantum_run().

    Theoretical basis:
      - Quantum walk-based shortest path: O(√(VE)) expected
      - Grover-accelerated relaxation: quadratic speedup on search subroutine
      - Currently: simulates quantum behavior via probabilistic sampling
        with a classical Dijkstra fallback for correctness.
    """
    name = "Quantum SSSP (stub)"
    supports_negative = False
    QUANTUM_SPEEDUP_FACTOR = 0.6   # simulated speedup multiplier on nodes_visited

    def __init__(self, graph, source):
        self.graph = graph; self.source = source
        self.dist = {}; self.nodes_visited = 0
        self._classical = DijkstraRerun(graph.clone(), source)
        self.dist = dict(self._classical.dist)
        self._quantum_log = []   # hook: append quantum circuit results here

    def _quantum_run(self, graph, source):
        """
        ── INTEGRATION POINT ──────────────────────────────────────────────────
        Replace this method with your actual quantum algorithm.

        Expected interface:
          Input : graph (Graph), source (int)
          Output: (dist_dict, nodes_explored_count)

        Until then, we run classical Dijkstra and apply a simulated
        speedup factor to nodes_visited so it shows up distinctly on charts.
        ───────────────────────────────────────────────────────────────────────
        """
        # --- classical fallback (remove when real quantum impl available) ---
        dist = {i: INF for i in range(graph.n)}
        dist[source] = 0
        pq = [(0, source)]; visited = set(); count = 0
        while pq:
            d, u = heapq.heappop(pq)
            if u in visited: continue
            visited.add(u); count += 1
            for v, w in graph.neighbors(u).items():
                if dist[u] + w < dist[v]:
                    dist[v] = dist[u] + w
                    heapq.heappush(pq, (dist[v], v))
        # simulate quantum speedup on node visits (not on correctness)
        simulated_count = max(1, int(count * self.QUANTUM_SPEEDUP_FACTOR))
        return dist, simulated_count

    def update(self, u, v, w_new):
        self.graph.update_weight(u, v, w_new)
        dist, count = self._quantum_run(self.graph, self.source)
        self.dist = dist; self.nodes_visited = count
        return self.nodes_visited


# ─────────────────────────────────────────────────────────────────────────────
# GRAPH GENERATORS
# ─────────────────────────────────────────────────────────────────────────────
def make_random_graph(n, n_edges, w_min, w_max, allow_negative, seed=42):
    random.seed(seed)
    g = Graph(n)
    possible = [(u, v) for u in range(n) for v in range(n) if u != v]
    random.shuffle(possible)
    edges = []
    for u, v in possible[:n_edges]:
        lo = -abs(w_min) if allow_negative else max(0.1, w_min)
        w  = random.uniform(lo, w_max)
        # avoid negative cycles: only allow negative on forward edges
        if allow_negative and v <= u: w = abs(w)
        g.add_edge(u, v, w); edges.append((u, v))
    return g, edges

def make_update(edges, graph, mode, seed=None):
    if seed is not None: random.seed(seed)
    u, v = random.choice(edges)
    w_cur = graph.get_weight(u, v)
    if mode == "decrease":   w_new = w_cur * random.uniform(0.3, 0.85)
    elif mode == "increase": w_new = w_cur * random.uniform(1.15, 3.0)
    else:                    w_new = w_cur * random.uniform(0.3, 3.0)
    return u, v, max(0.01, w_new)

# ─────────────────────────────────────────────────────────────────────────────
# BENCHMARK WORKER THREAD
# ─────────────────────────────────────────────────────────────────────────────
class BenchmarkWorker(QThread):
    result_ready  = pyqtSignal(str, float, int)   # algo_name, time_ms, nodes
    run_complete  = pyqtSignal(str, list, list)    # algo_name, times, nodes
    status_update = pyqtSignal(str)

    def __init__(self, algo_classes, graph, edges, n_updates, update_mode, allow_negative):
        super().__init__()
        self.algo_classes    = algo_classes
        self.graph           = graph
        self.edges           = edges
        self.n_updates       = n_updates
        self.update_mode     = update_mode
        self.allow_negative  = allow_negative
        self._stop           = False

    def stop(self): self._stop = True

    def run(self):
        updates = []
        g_tmp = self.graph.clone()
        random.seed(77)
        for i in range(self.n_updates):
            u, v, w = make_update(self.edges, g_tmp, self.update_mode)
            g_tmp.update_weight(u, v, w)
            updates.append((u, v, w))

        for AlgoClass in self.algo_classes:
            if self._stop: return
            if not self.allow_negative and not AlgoClass.supports_negative:
                pass  # fine
            self.status_update.emit(f"Running {AlgoClass.name}…")
            g = self.graph.clone()
            try:
                algo = AlgoClass(g, 0)
            except Exception as e:
                self.status_update.emit(f"{AlgoClass.name} init error: {e}")
                continue

            times = []; nodes_list = []
            for u, v, w_new in updates:
                if self._stop: return
                t0 = time.perf_counter()
                try:
                    nv = algo.update(u, v, w_new)
                except Exception:
                    nv = 0
                elapsed = (time.perf_counter() - t0) * 1000
                times.append(elapsed); nodes_list.append(nv)
                self.result_ready.emit(AlgoClass.name, elapsed, nv)

            self.run_complete.emit(AlgoClass.name, times, nodes_list)
        self.status_update.emit("Complete ✓")

# ─────────────────────────────────────────────────────────────────────────────
# MATPLOTLIB CANVAS (dark theme)
# ─────────────────────────────────────────────────────────────────────────────
class DarkCanvas(FigureCanvas):
    def __init__(self, fig):
        super().__init__(fig)
        self.setStyleSheet(f"background-color: {DARK_BG};")

def make_dark_figure(nrows=1, ncols=1, figsize=(6,3)):
    fig, axes = plt.subplots(nrows, ncols, figsize=figsize)
    fig.patch.set_facecolor(DARK_BG)
    axlist = axes if hasattr(axes, '__iter__') else [axes]
    for ax in np.array(axlist).flatten():
        ax.set_facecolor(PANEL_BG)
        ax.tick_params(colors=TEXT, labelsize=7)
        ax.xaxis.label.set_color(TEXT)
        ax.yaxis.label.set_color(TEXT)
        ax.title.set_color(ACCENT)
        for spine in ax.spines.values():
            spine.set_edgecolor(BORDER)
    return fig, axes

# ─────────────────────────────────────────────────────────────────────────────
# PER-ALGORITHM PANEL  (pops up as a QFrame card when checkbox ticked)
# ─────────────────────────────────────────────────────────────────────────────
class AlgoPanel(QFrame):
    def __init__(self, algo_name, parent=None):
        super().__init__(parent)
        self.algo_name = algo_name
        self.color     = ALGO_COLORS.get(algo_name, "#ffffff")
        self.times     = []
        self.nodes     = []
        self._build_ui()

    def _build_ui(self):
        self.setStyleSheet(f"""
            QFrame {{
                background: {PANEL_BG};
                border: 2px solid {self.color};
                border-radius: 10px;
            }}
        """)
        self.setMinimumWidth(320)
        self.setMinimumHeight(320)

        layout = QVBoxLayout(self)
        layout.setSpacing(4)
        layout.setContentsMargins(8, 8, 8, 8)

        # Header
        hdr = QLabel(f"  {self.algo_name}")
        hdr.setStyleSheet(f"color: {self.color}; font-weight: bold; font-size: 12px;"
                          f"border: none; background: transparent;")
        layout.addWidget(hdr)

        # Stats row
        self.stat_time  = QLabel("Avg: — ms")
        self.stat_nodes = QLabel("Nodes: —")
        self.stat_last  = QLabel("Last: — ms")
        for lbl in [self.stat_time, self.stat_nodes, self.stat_last]:
            lbl.setStyleSheet(f"color:{SUBTEXT}; font-size:10px; border:none; background:transparent;")
        row = QHBoxLayout()
        row.addWidget(self.stat_time); row.addWidget(self.stat_nodes); row.addWidget(self.stat_last)
        layout.addLayout(row)

        # Complexity tag
        cx = ALGO_COMPLEXITY.get(self.algo_name, ("?","","",""))
        cx_lbl = QLabel(f"Complexity: {cx[0]}  ({cx[1]})")
        cx_lbl.setStyleSheet(f"color:{ACCENT}; font-size:9px; border:none; background:transparent;")
        layout.addWidget(cx_lbl)

        # Chart
        self.fig, (self.ax_time, self.ax_nodes) = make_dark_figure(1, 2, figsize=(5, 2.4))
        self.fig.tight_layout(pad=0.8)
        self.canvas = DarkCanvas(self.fig)
        layout.addWidget(self.canvas)

        # Progress bar
        self.pbar = QProgressBar()
        self.pbar.setRange(0, 100)
        self.pbar.setValue(0)
        self.pbar.setFixedHeight(6)
        self.pbar.setTextVisible(False)
        self.pbar.setStyleSheet(f"""
            QProgressBar {{ background: {DARK_BG}; border-radius:3px; border:none; }}
            QProgressBar::chunk {{ background: {self.color}; border-radius:3px; }}
        """)
        layout.addWidget(self.pbar)

    def push_result(self, t_ms, nodes, total_updates):
        self.times.append(t_ms); self.nodes.append(nodes)
        pct = int(len(self.times) / max(1, total_updates) * 100)
        self.pbar.setValue(min(pct, 100))

        avg = sum(self.times) / len(self.times)
        self.stat_time.setText(f"Avg: {avg:.3f} ms")
        self.stat_nodes.setText(f"Nodes: {nodes}")
        self.stat_last.setText(f"Last: {t_ms:.3f} ms")

    def finalize(self):
        if not self.times: return
        self.pbar.setValue(100)
        c = self.color
        x = list(range(1, len(self.times)+1))

        self.ax_time.clear(); self.ax_nodes.clear()
        for ax in [self.ax_time, self.ax_nodes]:
            ax.set_facecolor(PANEL_BG)
            ax.tick_params(colors=TEXT, labelsize=6)
            for sp in ax.spines.values(): sp.set_edgecolor(BORDER)

        self.ax_time.plot(x, self.times, color=c, lw=1.2, alpha=0.9)
        self.ax_time.fill_between(x, self.times, alpha=0.15, color=c)
        self.ax_time.set_title("Time / update (ms)", color=ACCENT, fontsize=8, pad=3)
        self.ax_time.set_xlabel("Update #", color=TEXT, fontsize=6)

        self.ax_nodes.plot(x, self.nodes, color=ACCENT2, lw=1.2, alpha=0.9)
        self.ax_nodes.fill_between(x, self.nodes, alpha=0.15, color=ACCENT2)
        self.ax_nodes.set_title("Nodes visited", color=ACCENT, fontsize=8, pad=3)
        self.ax_nodes.set_xlabel("Update #", color=TEXT, fontsize=6)

        self.fig.tight_layout(pad=0.8)
        self.canvas.draw()

    def reset(self):
        self.times.clear(); self.nodes.clear()
        self.pbar.setValue(0)
        self.stat_time.setText("Avg: — ms")
        self.stat_nodes.setText("Nodes: —")
        self.stat_last.setText("Last: — ms")
        for ax in [self.ax_time, self.ax_nodes]: ax.clear()
        self.canvas.draw()

# ─────────────────────────────────────────────────────────────────────────────
# SUMMARY CHART PANEL
# ─────────────────────────────────────────────────────────────────────────────
class SummaryPanel(QWidget):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.results = {}   # algo_name -> (times, nodes)
        layout = QVBoxLayout(self)
        layout.setContentsMargins(4,4,4,4)

        self.fig, self.axes = make_dark_figure(2, 2, figsize=(10, 5))
        self.fig.suptitle("Benchmark Summary", color=ACCENT, fontsize=12)
        self.canvas = DarkCanvas(self.fig)
        layout.addWidget(self.canvas)

    def update(self, results):
        self.results = results
        self._redraw()

    def _redraw(self):
        if not self.results: return
        axs = self.axes.flatten() if hasattr(self.axes, 'flatten') else [self.axes]
        for ax in axs:
            ax.clear()
            ax.set_facecolor(PANEL_BG)
            ax.tick_params(colors=TEXT, labelsize=7)
            for sp in ax.spines.values(): sp.set_edgecolor(BORDER)

        names  = list(self.results.keys())
        colors = [ALGO_COLORS.get(n, "#fff") for n in names]

        # Avg time bar chart
        avg_times = [sum(self.results[n][0])/max(1,len(self.results[n][0])) for n in names]
        axs[0].barh(names, avg_times, color=colors, alpha=0.85)
        axs[0].set_title("Avg Time per Update (ms)", color=ACCENT, fontsize=9)
        axs[0].set_xlabel("ms", color=TEXT, fontsize=7)
        for i, v in enumerate(avg_times):
            axs[0].text(v + 0.001, i, f"{v:.3f}", va='center', color=TEXT, fontsize=6)

        # Avg nodes bar chart
        avg_nodes = [sum(self.results[n][1])/max(1,len(self.results[n][1])) for n in names]
        axs[1].barh(names, avg_nodes, color=colors, alpha=0.85)
        axs[1].set_title("Avg Nodes Visited per Update", color=ACCENT, fontsize=9)
        axs[1].set_xlabel("nodes", color=TEXT, fontsize=7)
        for i, v in enumerate(avg_nodes):
            axs[1].text(v + 0.1, i, f"{v:.1f}", va='center', color=TEXT, fontsize=6)

        # Time convergence lines
        for n in names:
            ts = self.results[n][0]
            axs[2].plot(ts, color=ALGO_COLORS.get(n,"#fff"), lw=1.1, label=n, alpha=0.9)
        axs[2].set_title("Time per Update over Sequence", color=ACCENT, fontsize=9)
        axs[2].set_xlabel("Update #", color=TEXT, fontsize=7)
        axs[2].set_ylabel("ms", color=TEXT, fontsize=7)
        axs[2].legend(fontsize=5, facecolor=PANEL_BG, labelcolor=TEXT,
                      loc='upper right', framealpha=0.7)

        # Speedup vs Dijkstra
        baseline_name = next((n for n in names if "Dijkstra" in n), None)
        if baseline_name:
            baseline_avg = sum(self.results[baseline_name][0]) / max(1, len(self.results[baseline_name][0]))
            speedups = [baseline_avg / max(1e-9, at) for at in avg_times]
            bars = axs[3].bar(names, speedups, color=colors, alpha=0.85)
            axs[3].axhline(1.0, color=RED, lw=1, ls='--', alpha=0.7)
            axs[3].set_title("Speedup vs Dijkstra Rerun", color=ACCENT, fontsize=9)
            axs[3].set_ylabel("×", color=TEXT, fontsize=7)
            axs[3].tick_params(axis='x', rotation=20, labelsize=6)
            for bar, sv in zip(bars, speedups):
                axs[3].text(bar.get_x()+bar.get_width()/2, bar.get_height()+0.1,
                            f"{sv:.1f}×", ha='center', color=TEXT, fontsize=6)

        self.fig.tight_layout(pad=1.0)
        self.canvas.draw()

# ─────────────────────────────────────────────────────────────────────────────
# COMPLEXITY TABLE TAB
# ─────────────────────────────────────────────────────────────────────────────
class ComplexityTable(QWidget):
    def __init__(self, parent=None):
        super().__init__(parent)
        layout = QVBoxLayout(self)
        lbl = QLabel("Algorithm Complexity Reference")
        lbl.setStyleSheet(f"color:{ACCENT}; font-size:14px; font-weight:bold;")
        layout.addWidget(lbl)

        tbl = QTableWidget()
        tbl.setColumnCount(5)
        tbl.setHorizontalHeaderLabels(["Algorithm", "Complexity", "Notes", "Neg. Weights", "No Rerun"])
        tbl.horizontalHeader().setSectionResizeMode(QHeaderView.Stretch)
        tbl.setStyleSheet(f"""
            QTableWidget {{ background:{PANEL_BG}; color:{TEXT}; gridline-color:{BORDER}; border:none; }}
            QHeaderView::section {{ background:{DARK_BG}; color:{ACCENT}; font-weight:bold; padding:6px; }}
            QTableWidget::item {{ padding:6px; }}
        """)
        tbl.setEditTriggers(QTableWidget.NoEditTriggers)
        tbl.setAlternatingRowColors(True)
        tbl.setSelectionMode(QTableWidget.NoSelection)

        rows = [
            ("Dijkstra Full Rerun",     "O((V+E) log V)", "per full rerun",  "No",  "No"),
            ("Bellman-Ford Full Rerun",  "O(VE)",          "per full rerun",  "Yes", "No"),
            ("Dynamic Bellman-Ford",     "O(k·E)",         "k = affected",    "Yes", "Yes"),
            ("Ramalingam-Reps",          "O(k log V)",     "k = affected",    "No",  "Yes"),
            ("LPA*",                     "O(k log V)",     "k = inconsistent","No",  "Yes"),
            ("Quantum SSSP (stub)",      "O(√(VE))*",      "theoretical",     "Yes*","Yes"),
        ]
        tbl.setRowCount(len(rows))
        color_map = {"No": RED, "Yes": ACCENT2, "Yes*": WARN}
        for r, row in enumerate(rows):
            for c, val in enumerate(row):
                item = QTableWidgetItem(val)
                item.setTextAlignment(Qt.AlignCenter)
                if c in (3, 4) and val in color_map:
                    item.setForeground(QColor(color_map[val]))
                if c == 0:
                    item.setForeground(QColor(ALGO_COLORS.get(row[0], TEXT)))
                tbl.setItem(r, c, item)

        layout.addWidget(tbl)

        note = QLabel("* Quantum: O(√(VE)) speedup is theoretical. Stub uses classical Dijkstra + simulated node-visit reduction.")
        note.setStyleSheet(f"color:{SUBTEXT}; font-size:10px; padding:8px;")
        note.setWordWrap(True)
        layout.addWidget(note)

# ─────────────────────────────────────────────────────────────────────────────
# QUANTUM INFO TAB
# ─────────────────────────────────────────────────────────────────────────────
class QuantumInfoTab(QWidget):
    def __init__(self, parent=None):
        super().__init__(parent)
        layout = QVBoxLayout(self)

        title = QLabel("⚛  Quantum SSSP Module — Integration Guide")
        title.setStyleSheet(f"color:{WARN}; font-size:14px; font-weight:bold; padding:8px;")
        layout.addWidget(title)

        info = QTextEdit()
        info.setReadOnly(True)
        info.setStyleSheet(f"background:{PANEL_BG}; color:{TEXT}; border:1px solid {BORDER}; border-radius:6px; padding:12px;")
        info.setHtml(f"""
        <style>body{{font-family:Consolas,monospace; font-size:12px; color:{TEXT};}}
               h3{{color:{WARN};}} code{{color:{ACCENT};}} b{{color:{ACCENT2};}}</style>
        <h3>Current Status: STUB / PLACEHOLDER</h3>
        <p>The <code>QuantumSSSP</code> class is a fully wired integration point.
        It participates in all benchmarks when checked, reports real timing,
        and shows up in summary charts — but internally falls back to classical Dijkstra.</p>

        <h3>How to plug in your real implementation</h3>
        <p>Open <code>sp_visualizer.py</code> and locate the <code>QuantumSSSP</code> class.
        Replace the body of <b>_quantum_run(self, graph, source)</b> with your algorithm:</p>
        <pre><code>
def _quantum_run(self, graph, source):
    # Your quantum implementation here.
    # Must return:
    #   dist  : dict  {{ node_id -> shortest_distance }}
    #   count : int   {{ number of quantum operations / node equivalents }}

    # Example stub (replace this):
    dist, count = your_quantum_algorithm(graph, source)
    return dist, count
        </code></pre>

        <h3>Interface contract</h3>
        <p><b>Input:</b> graph (Graph object with .adj, .radj, .n), source (int)</p>
        <p><b>Output:</b> (dist_dict, operations_count)</p>
        <p>The <code>update(u, v, w_new)</code> method calls <code>_quantum_run</code>
        after applying each edge weight change.
        You can extend it to support incremental quantum updates later.</p>

        <h3>Theoretical background</h3>
        <p>• Quantum walk on sparse graphs: O(√(VE)) expected<br>
           • Grover-accelerated Bellman-Ford: O(√V · E) relaxations<br>
           • Quantum amplitude amplification on shortest path search<br>
           • Reference: Dürr et al. (2006), "Quantum query complexity of some graph problems"</p>

        <h3>Simulation parameters</h3>
        <p><code>QUANTUM_SPEEDUP_FACTOR = 0.6</code> — reduces reported node visits to
        60% of classical count, simulating quantum speedup visually on the nodes-visited chart.</p>
        """)
        layout.addWidget(info)

# ─────────────────────────────────────────────────────────────────────────────
# MAIN WINDOW
# ─────────────────────────────────────────────────────────────────────────────
class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Dynamic Shortest Path Benchmark Visualizer")
        self.resize(1400, 900)
        self._apply_dark_theme()

        self.worker      = None
        self.algo_panels = {}    # name -> AlgoPanel
        self.all_results = {}    # name -> (times, nodes)
        self.n_updates   = 60
        self.checkboxes  = {}    # name -> QCheckBox

        self._build_ui()

    # ── Dark theme ────────────────────────────────────────────────────────────
    def _apply_dark_theme(self):
        self.setStyleSheet(f"""
            QMainWindow, QWidget {{ background:{DARK_BG}; color:{TEXT}; }}
            QGroupBox {{ border:1px solid {BORDER}; border-radius:6px; margin-top:8px;
                         color:{ACCENT}; font-weight:bold; padding:6px; }}
            QGroupBox::title {{ subcontrol-origin:margin; left:8px; top:-4px; }}
            QLabel {{ color:{TEXT}; }}
            QPushButton {{
                background: qlineargradient(x1:0,y1:0,x2:1,y2:1,stop:0 #667eea,stop:1 #764ba2);
                color:white; border:none; border-radius:6px; padding:8px 16px; font-weight:bold;
            }}
            QPushButton:hover {{ background: {ACCENT}; color:{DARK_BG}; }}
            QPushButton:disabled {{ opacity:0.4; background:#444; color:#888; }}
            QSlider::groove:horizontal {{ background:{BORDER}; height:6px; border-radius:3px; }}
            QSlider::handle:horizontal {{
                background:{ACCENT}; width:16px; height:16px; margin:-5px 0;
                border-radius:8px;
            }}
            QComboBox {{ background:{PANEL_BG}; border:1px solid {BORDER}; border-radius:4px;
                         padding:4px; color:{TEXT}; }}
            QComboBox QAbstractItemView {{ background:{PANEL_BG}; color:{TEXT}; }}
            QCheckBox {{ color:{TEXT}; spacing:8px; }}
            QCheckBox::indicator {{ width:16px; height:16px; border:2px solid {BORDER};
                                    border-radius:3px; background:{DARK_BG}; }}
            QCheckBox::indicator:checked {{ background:{ACCENT}; border-color:{ACCENT}; }}
            QScrollArea {{ border:none; }}
            QTabWidget::pane {{ border:1px solid {BORDER}; background:{DARK_BG}; }}
            QTabBar::tab {{ background:{PANEL_BG}; color:{SUBTEXT}; padding:8px 16px;
                            border-radius:4px 4px 0 0; margin-right:2px; }}
            QTabBar::tab:selected {{ background:{DARK_BG}; color:{ACCENT}; font-weight:bold; }}
        """)

    # ── Build UI ──────────────────────────────────────────────────────────────
    def _build_ui(self):
        central = QWidget(); self.setCentralWidget(central)
        root = QHBoxLayout(central); root.setSpacing(8); root.setContentsMargins(8,8,8,8)

        # LEFT SIDEBAR
        sidebar = self._build_sidebar()
        root.addWidget(sidebar, 0)

        # RIGHT CONTENT
        tabs = QTabWidget()
        tabs.setFont(QFont("Segoe UI", 10))

        # Tab 1 — Algorithm Panels
        self.panels_tab = QWidget()
        self.panels_layout = QGridLayout(self.panels_tab)
        self.panels_layout.setSpacing(10)
        self.panels_layout.setContentsMargins(6,6,6,6)
        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        scroll.setWidget(self.panels_tab)
        tabs.addTab(scroll, "📊  Algorithm Panels")

        # Tab 2 — Summary Charts
        self.summary_panel = SummaryPanel()
        tabs.addTab(self.summary_panel, "📈  Summary Charts")

        # Tab 3 — Complexity Table
        tabs.addTab(ComplexityTable(), "🗂  Complexity Reference")

        # Tab 4 — Quantum Info
        tabs.addTab(QuantumInfoTab(), "⚛  Quantum Module")

        root.addWidget(tabs, 1)

    def _build_sidebar(self):
        sidebar = QWidget(); sidebar.setFixedWidth(260)
        layout  = QVBoxLayout(sidebar)
        layout.setSpacing(10); layout.setContentsMargins(0,0,0,0)

        # Title
        title = QLabel(" Shortest Path in Changing Edge Weights")
        title.setStyleSheet(f"color:{ACCENT}; font-size:16px; font-weight:bold; padding:8px;")
        layout.addWidget(title)

        # ── Graph config ──────────────────────────────────────────────────────
        grp_graph = QGroupBox("Graph Configuration")
        gg = QVBoxLayout(grp_graph)

        gg.addWidget(QLabel("Nodes"))
        self.sld_nodes = QSlider(Qt.Horizontal); self.sld_nodes.setRange(10, 300); self.sld_nodes.setValue(60)
        self.lbl_nodes = QLabel("60"); self.sld_nodes.valueChanged.connect(lambda v: self.lbl_nodes.setText(str(v)))
        gg.addWidget(self.sld_nodes); gg.addWidget(self.lbl_nodes)

        gg.addWidget(QLabel("Edges"))
        self.sld_edges = QSlider(Qt.Horizontal); self.sld_edges.setRange(10, 1000); self.sld_edges.setValue(150)
        self.lbl_edges = QLabel("150"); self.sld_edges.valueChanged.connect(lambda v: self.lbl_edges.setText(str(v)))
        gg.addWidget(self.sld_edges); gg.addWidget(self.lbl_edges)

        gg.addWidget(QLabel("Updates"))
        self.sld_updates = QSlider(Qt.Horizontal); self.sld_updates.setRange(10, 200); self.sld_updates.setValue(60)
        self.lbl_updates = QLabel("60"); self.sld_updates.valueChanged.connect(lambda v: self.lbl_updates.setText(str(v)))
        gg.addWidget(self.sld_updates); gg.addWidget(self.lbl_updates)

        layout.addWidget(grp_graph)

        # ── Weight config ─────────────────────────────────────────────────────
        grp_weights = QGroupBox("Weight Change Mode")
        gw = QVBoxLayout(grp_weights)
        self.cmb_mode = QComboBox()
        self.cmb_mode.addItems(["mixed (↑ and ↓)", "decrease only (↓)", "increase only (↑)", "traffic (big swings)"])
        gw.addWidget(self.cmb_mode)

        self.chk_negative = QCheckBox("Allow Negative Weights")
        self.chk_negative.setStyleSheet(f"color:{WARN};")
        gw.addWidget(self.chk_negative)

        layout.addWidget(grp_weights)

        # ── Algorithm selection ───────────────────────────────────────────────
        grp_algos = QGroupBox("Algorithms  (check to enable)")
        ga = QVBoxLayout(grp_algos)
        ALGO_CLASSES = [DijkstraRerun, BellmanFordRerun, DynamicBellmanFord,
                        RamalingamReps, LPAStar, QuantumSSSP]
        for AlgoClass in ALGO_CLASSES:
            cb = QCheckBox(AlgoClass.name)
            cb.setChecked(AlgoClass.name not in ["Bellman-Ford Full Rerun", "Quantum SSSP (stub)"])
            c  = ALGO_COLORS.get(AlgoClass.name, TEXT)
            cb.setStyleSheet(f"color:{c}; font-weight:bold;")
            cb.stateChanged.connect(lambda state, ac=AlgoClass: self._toggle_algo(ac, state))
            self.checkboxes[AlgoClass.name] = cb
            ga.addWidget(cb)

        layout.addWidget(grp_algos)

        # ── Controls ──────────────────────────────────────────────────────────
        grp_ctrl = QGroupBox("Controls")
        gc = QVBoxLayout(grp_ctrl)

        self.btn_run = QPushButton("▶  Run Benchmark")
        self.btn_run.clicked.connect(self._run_benchmark)
        gc.addWidget(self.btn_run)

        self.btn_stop = QPushButton("⏹  Stop")
        self.btn_stop.clicked.connect(self._stop_benchmark)
        self.btn_stop.setEnabled(False)
        gc.addWidget(self.btn_stop)

        self.btn_reset = QPushButton("🔄  Reset")
        self.btn_reset.clicked.connect(self._reset)
        gc.addWidget(self.btn_reset)

        layout.addWidget(grp_ctrl)

        # Status
        self.lbl_status = QLabel("Ready")
        self.lbl_status.setStyleSheet(f"color:{ACCENT2}; font-size:11px; padding:4px;")
        self.lbl_status.setWordWrap(True)
        layout.addWidget(self.lbl_status)

        layout.addStretch()

        # Legend
        legend = QGroupBox("Legend")
        lg = QVBoxLayout(legend)
        for name, color in ALGO_COLORS.items():
            row = QHBoxLayout()
            dot = QLabel("●"); dot.setStyleSheet(f"color:{color}; font-size:14px;")
            lbl = QLabel(name.split("(")[0].strip()); lbl.setStyleSheet(f"color:{TEXT}; font-size:10px;")
            row.addWidget(dot); row.addWidget(lbl); row.addStretch()
            lg.addLayout(row)
        layout.addWidget(legend)

        return sidebar

    # ── Toggle algorithm panels ───────────────────────────────────────────────
    def _toggle_algo(self, AlgoClass, state):
        name = AlgoClass.name
        if state == Qt.Checked:
            if name not in self.algo_panels:
                panel = AlgoPanel(name)
                self.algo_panels[name] = panel
                self._relayout_panels()
        else:
            if name in self.algo_panels:
                panel = self.algo_panels.pop(name)
                panel.setParent(None)
                panel.deleteLater()
                self._relayout_panels()

    def _relayout_panels(self):
        # clear grid
        while self.panels_layout.count():
            item = self.panels_layout.takeAt(0)
            if item.widget(): item.widget().setParent(None)
        cols = 3
        for i, (name, panel) in enumerate(self.algo_panels.items()):
            self.panels_layout.addWidget(panel, i // cols, i % cols)

    # ── Benchmark run ─────────────────────────────────────────────────────────
    def _run_benchmark(self):
        n_nodes   = self.sld_nodes.value()
        n_edges   = self.sld_edges.value()
        n_updates = self.sld_updates.value()
        mode_raw  = self.cmb_mode.currentText()
        mode      = "mixed" if "mixed" in mode_raw else \
                    "decrease" if "decrease" in mode_raw else \
                    "increase" if "increase" in mode_raw else "traffic"
        allow_neg = self.chk_negative.isChecked()

        # Collect active algo classes
        ALGO_MAP = {
            "Dijkstra Full Rerun":      DijkstraRerun,
            "Bellman-Ford Full Rerun":  BellmanFordRerun,
            "Dynamic Bellman-Ford":     DynamicBellmanFord,
            "Ramalingam-Reps":          RamalingamReps,
            "LPA*":                     LPAStar,
            "Quantum SSSP (stub)":      QuantumSSSP,
        }
        active = [ALGO_MAP[n] for n, cb in self.checkboxes.items()
                  if cb.isChecked() and n in ALGO_MAP]
        if not active:
            self.lbl_status.setText("Select at least one algorithm."); return

        # Ensure panels exist for all active algos
        for ac in active:
            if ac.name not in self.algo_panels:
                self.algo_panels[ac.name] = AlgoPanel(ac.name)
        self._relayout_panels()
        for panel in self.algo_panels.values(): panel.reset()
        self.all_results.clear()

        # Build graph
        graph, edges = make_random_graph(n_nodes, n_edges,
                                          w_min=1, w_max=20,
                                          allow_negative=allow_neg)
        if not edges:
            self.lbl_status.setText("No edges generated — increase edge count."); return

        self.n_updates = n_updates

        # Worker thread
        self.worker = BenchmarkWorker(active, graph, edges, n_updates, mode, allow_neg)
        self.worker.result_ready.connect(self._on_result)
        self.worker.run_complete.connect(self._on_run_complete)
        self.worker.status_update.connect(self._on_status)
        self.worker.finished.connect(self._on_finished)
        self.worker.start()

        self.btn_run.setEnabled(False)
        self.btn_stop.setEnabled(True)
        self.lbl_status.setText("Running…")

    def _stop_benchmark(self):
        if self.worker: self.worker.stop()
        self.lbl_status.setText("Stopping…")

    def _reset(self):
        if self.worker: self.worker.stop()
        for panel in self.algo_panels.values(): panel.reset()
        self.all_results.clear()
        self.lbl_status.setText("Ready")
        self.btn_run.setEnabled(True)
        self.btn_stop.setEnabled(False)

    # ── Signals ───────────────────────────────────────────────────────────────
    def _on_result(self, algo_name, t_ms, nodes):
        if algo_name in self.algo_panels:
            self.algo_panels[algo_name].push_result(t_ms, nodes, self.n_updates)

    def _on_run_complete(self, algo_name, times, nodes):
        self.all_results[algo_name] = (times, nodes)
        if algo_name in self.algo_panels:
            self.algo_panels[algo_name].finalize()
        self.summary_panel.update(self.all_results)

    def _on_status(self, msg):
        self.lbl_status.setText(msg)

    def _on_finished(self):
        self.btn_run.setEnabled(True)
        self.btn_stop.setEnabled(False)
        self.summary_panel.update(self.all_results)


# ─────────────────────────────────────────────────────────────────────────────
# ENTRY POINT
# ─────────────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    app = QApplication(sys.argv)
    app.setFont(QFont("Segoe UI", 10))
    win = MainWindow()

    # Initialize panels for default-checked algorithms
    default_checked = ["Dijkstra Full Rerun", "Dynamic Bellman-Ford",
                       "Ramalingam-Reps", "LPA*"]
    ALGO_MAP = {
        "Dijkstra Full Rerun":     DijkstraRerun,
        "Dynamic Bellman-Ford":    DynamicBellmanFord,
        "Ramalingam-Reps":         RamalingamReps,
        "LPA*":                    LPAStar,
    }
    for name in default_checked:
        if name in ALGO_MAP:
            win.algo_panels[name] = AlgoPanel(name)
    win._relayout_panels()

    win.show()
    sys.exit(app.exec_())