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

  Algorithm logic lives in algorithms/ — this file is UI only.
=============================================================================
"""

import sys, time, random, math
from typing import Dict, List, Tuple

# ── Algorithm package — all logic lives here ──────────────────────────────────
from algorithms import (
    Graph, INF,
    DijkstraRerun, BellmanFordRerun, DynamicBellmanFord,
    RamalingamReps, LPAStar, QuantumSSSP,
)

import json
from visualization import SimulationReplayTab

import numpy as np
import networkx as nx
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
    QSpinBox, QSizePolicy, QTextEdit, QMessageBox
)
from PyQt5.QtCore import Qt, QThread, pyqtSignal, QTimer, QSize
from PyQt5.QtGui import QFont, QColor, QPalette, QIcon

# ─────────────────────────────────────────────────────────────────────────────
# THEME — Clean white / high-contrast light mode
# ─────────────────────────────────────────────────────────────────────────────
DARK_BG     = "#f4f6fb"    # page background (light blue-grey)
PANEL_BG    = "#ffffff"    # card / panel background
ACCENT      = "#1d4ed8"    # primary blue
ACCENT2     = "#047857"    # green
WARN        = "#b45309"    # amber
RED         = "#b91c1c"    # red
TEXT        = "#111827"    # near-black
SUBTEXT     = "#6b7280"    # grey
BORDER      = "#d1d5db"    # light border

# Algorithm brand colors — vivid enough to stand out on white
ALGO_COLORS = {
    "Dijkstra Full Rerun":        "#e11d48",   # rose-600
    "Bellman-Ford Full Rerun":    "#ea580c",   # orange-600
    "Dynamic Bellman-Ford":       "#2563eb",   # blue-600
    "Ramalingam-Reps (RR-SSSP)": "#059669",   # emerald-600
    "LPA*":                       "#7c3aed",   # violet-600
    "Quantum SSSP (stub)":        "#0891b2",   # cyan-600
}

# ── name must exactly match AlgoClass.name ──────────────────────────────────
ALGO_COMPLEXITY = {
    "Dijkstra Full Rerun":        ("O((V+E) log V)", "per full rerun", "No",  "No"),
    "Bellman-Ford Full Rerun":    ("O(VE)",           "per full rerun", "Yes", "No"),
    "Dynamic Bellman-Ford":       ("O(k·E)",          "k=affected",     "Yes", "Yes"),
    "Ramalingam-Reps (RR-SSSP)": ("O(k log V)",      "k=affected",     "No",  "Yes"),
    "LPA*":                       ("O(k log V)",      "k=inconsistent", "No",  "Yes"),
    "Quantum SSSP (stub)":        ("O(√(VE))*",       "theoretical",    "Yes*","Yes"),
}

# ─────────────────────────────────────────────────────────────────────────────
# NOTE: Graph, INF, and all algorithm classes are imported from algorithms/
# This file contains only GUI / visualization logic.
# ─────────────────────────────────────────────────────────────────────────────


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

        simulation_data = {
            "n_nodes": self.graph.n,
            "edges": self.edges,
            "initial_weights": [[u, v, self.graph.get_weight(u, v)] for u, v in self.edges],
            "updates": [{"u": u, "v": v, "w_new": w, "algos": {}} for u, v, w in updates]
        }

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
            for i, (u, v, w_new) in enumerate(updates):
                if self._stop: return
                t0 = time.perf_counter()
                try:
                    nv = algo.update(u, v, w_new)
                    vl = getattr(algo, "visited_nodes_list", [])
                except Exception:
                    nv = 0
                    vl = []
                elapsed = (time.perf_counter() - t0) * 1000
                times.append(elapsed); nodes_list.append(nv)
                self.result_ready.emit(AlgoClass.name, elapsed, nv)

                simulation_data["updates"][i]["algos"][AlgoClass.name] = {
                    "time_ms": elapsed,
                    "nodes_visited_count": nv,
                    "visited_list": vl
                }

            self.run_complete.emit(AlgoClass.name, times, nodes_list)
        
        try:
            with open("simulation_pattern.json", "w") as f:
                json.dump(simulation_data, f, indent=2)
        except Exception:
            pass
            
        self.status_update.emit("Complete ✓")

# ─────────────────────────────────────────────────────────────────────────────
# MATPLOTLIB CANVAS (light theme)
# ─────────────────────────────────────────────────────────────────────────────
class DarkCanvas(FigureCanvas):   # named DarkCanvas for API compat, now light
    def __init__(self, fig):
        super().__init__(fig)
        self.setStyleSheet(f"background-color: {PANEL_BG}; border: none;")

def make_dark_figure(nrows=1, ncols=1, figsize=(6, 3)):
    """Create a matplotlib figure styled for the light theme."""
    fig, axes = plt.subplots(nrows, ncols, figsize=figsize)
    fig.patch.set_facecolor(PANEL_BG)
    axlist = np.array(axes).flatten() if hasattr(axes, '__iter__') else [axes]
    for ax in axlist:
        ax.set_facecolor(PANEL_BG)
        ax.tick_params(colors=SUBTEXT, labelsize=7)
        ax.xaxis.label.set_color(SUBTEXT)
        ax.yaxis.label.set_color(SUBTEXT)
        ax.title.set_color(TEXT)
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
            QProgressBar {{ background: {BORDER}; border-radius:3px; border:none; }}
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
            ax.tick_params(colors=SUBTEXT, labelsize=6)
            for sp in ax.spines.values(): sp.set_edgecolor(BORDER)

        self.ax_time.plot(x, self.times, color=c, lw=1.5, alpha=1.0)
        self.ax_time.fill_between(x, self.times, alpha=0.12, color=c)
        self.ax_time.set_title("Time / update (ms)", color=TEXT, fontsize=8, fontweight='bold', pad=3)
        self.ax_time.set_xlabel("Update #", color=SUBTEXT, fontsize=6)

        self.ax_nodes.plot(x, self.nodes, color=ACCENT, lw=1.5, alpha=1.0)
        self.ax_nodes.fill_between(x, self.nodes, alpha=0.10, color=ACCENT)
        self.ax_nodes.set_title("Nodes visited", color=TEXT, fontsize=8, fontweight='bold', pad=3)
        self.ax_nodes.set_xlabel("Update #", color=SUBTEXT, fontsize=6)

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
        self.fig.suptitle("Benchmark Summary", color=TEXT, fontsize=13, fontweight='bold')
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
            ax.tick_params(colors=SUBTEXT, labelsize=7)
            for sp in ax.spines.values(): sp.set_edgecolor(BORDER)

        names  = list(self.results.keys())
        colors = [ALGO_COLORS.get(n, "#fff") for n in names]

        # Avg time bar chart
        avg_times = [sum(self.results[n][0])/max(1,len(self.results[n][0])) for n in names]
        axs[0].barh(names, avg_times, color=colors, alpha=0.9, edgecolor='none')
        axs[0].set_title("Avg Time per Update (ms)", color=TEXT, fontsize=9, fontweight='bold')
        axs[0].set_xlabel("ms", color=SUBTEXT, fontsize=7)
        for i, v in enumerate(avg_times):
            axs[0].text(v + 0.001, i, f"{v:.3f}", va='center', color=TEXT, fontsize=7, fontweight='bold')

        # Avg nodes bar chart
        avg_nodes = [sum(self.results[n][1])/max(1,len(self.results[n][1])) for n in names]
        axs[1].barh(names, avg_nodes, color=colors, alpha=0.9, edgecolor='none')
        axs[1].set_title("Avg Nodes Visited per Update", color=TEXT, fontsize=9, fontweight='bold')
        axs[1].set_xlabel("nodes", color=SUBTEXT, fontsize=7)
        for i, v in enumerate(avg_nodes):
            axs[1].text(v + 0.1, i, f"{v:.1f}", va='center', color=TEXT, fontsize=7, fontweight='bold')

        # Time convergence lines
        for n in names:
            ts = self.results[n][0]
            axs[2].plot(ts, color=ALGO_COLORS.get(n, ACCENT), lw=1.5, label=n, alpha=1.0)
        axs[2].set_title("Time per Update over Sequence", color=TEXT, fontsize=9, fontweight='bold')
        axs[2].set_xlabel("Update #", color=SUBTEXT, fontsize=7)
        axs[2].set_ylabel("ms", color=SUBTEXT, fontsize=7)
        axs[2].legend(fontsize=6, facecolor=PANEL_BG, labelcolor=TEXT,
                      edgecolor=BORDER, loc='upper right', framealpha=0.9)

        # Speedup vs Dijkstra
        baseline_name = next((n for n in names if "Dijkstra" in n), None)
        if baseline_name:
            baseline_avg = sum(self.results[baseline_name][0]) / max(1, len(self.results[baseline_name][0]))
            speedups = [baseline_avg / max(1e-9, at) for at in avg_times]
            bars = axs[3].bar(names, speedups, color=colors, alpha=0.9, edgecolor='none')
            axs[3].axhline(1.0, color=RED, lw=1.5, ls='--', alpha=0.85)
            axs[3].set_title("Speedup vs Dijkstra Rerun", color=TEXT, fontsize=9, fontweight='bold')
            axs[3].set_ylabel("×", color=SUBTEXT, fontsize=8)
            axs[3].tick_params(axis='x', rotation=20, labelsize=6, colors=SUBTEXT)
            for bar, sv in zip(bars, speedups):
                axs[3].text(bar.get_x()+bar.get_width()/2, bar.get_height()+0.1,
                            f"{sv:.1f}×", ha='center', color=TEXT, fontsize=7, fontweight='bold')

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
            ("Dijkstra Full Rerun",          "O((V+E) log V)", "per full rerun",  "No",  "No"),
            ("Bellman-Ford Full Rerun",       "O(VE)",           "per full rerun",  "Yes", "No"),
            ("Dynamic Bellman-Ford",          "O(k·E)",          "k = affected",    "Yes", "Yes"),
            ("Ramalingam-Reps (RR-SSSP)",    "O(k log V)",      "k = affected",    "No",  "Yes"),
            ("LPA*",                          "O(k log V)",      "k = inconsistent","No",  "Yes"),
            ("Quantum SSSP (stub)",           "O(√(VE))*",       "theoretical",     "Yes*","Yes"),
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
        self.showMaximized()   # start full-screen so panels never overflow
        self._apply_dark_theme()

        self.worker      = None
        self.algo_panels = {}    # name -> AlgoPanel
        self.all_results = {}    # name -> (times, nodes)
        self.n_updates   = 60
        self.checkboxes  = {}    # name -> QCheckBox

        self._build_ui()

    # ── Light theme ───────────────────────────────────────────────────────────
    def _apply_dark_theme(self):   # kept the method name for compatibility
        self.setStyleSheet(f"""
            QMainWindow, QWidget {{
                background: {DARK_BG};
                color: {TEXT};
                font-family: 'Helvetica Neue', 'Arial', sans-serif;
            }}
            QGroupBox {{
                border: 1.5px solid {BORDER};
                border-radius: 8px;
                margin-top: 10px;
                background: {PANEL_BG};
                color: {ACCENT};
                font-weight: bold;
                padding: 8px;
            }}
            QGroupBox::title {{
                subcontrol-origin: margin;
                left: 10px;
                top: -5px;
                background: {PANEL_BG};
                padding: 0 4px;
            }}
            QLabel {{ color: {TEXT}; }}
            QPushButton {{
                background: {ACCENT};
                color: white;
                border: none;
                border-radius: 6px;
                padding: 8px 18px;
                font-weight: bold;
            }}
            QPushButton:hover {{ background: #1e40af; color: white; }}
            QPushButton:disabled {{ background: {BORDER}; color: {SUBTEXT}; }}
            QSlider::groove:horizontal {{
                background: {BORDER};
                height: 6px;
                border-radius: 3px;
            }}
            QSlider::handle:horizontal {{
                background: {ACCENT};
                width: 16px;
                height: 16px;
                margin: -5px 0;
                border-radius: 8px;
            }}
            QSlider::sub-page:horizontal {{
                background: {ACCENT};
                border-radius: 3px;
            }}
            QComboBox {{
                background: {PANEL_BG};
                border: 1.5px solid {BORDER};
                border-radius: 6px;
                padding: 5px 8px;
                color: {TEXT};
            }}
            QComboBox:hover {{ border-color: {ACCENT}; }}
            QComboBox QAbstractItemView {{
                background: {PANEL_BG};
                color: {TEXT};
                selection-background-color: {ACCENT};
                selection-color: white;
            }}
            QCheckBox {{ color: {TEXT}; spacing: 8px; }}
            QCheckBox::indicator {{
                width: 16px;
                height: 16px;
                border: 2px solid {BORDER};
                border-radius: 4px;
                background: white;
            }}
            QCheckBox::indicator:checked {{
                background: {ACCENT};
                border-color: {ACCENT};
            }}
            QScrollArea {{ border: none; background: {DARK_BG}; }}
            QTabWidget::pane {{
                border: 1.5px solid {BORDER};
                border-radius: 0 6px 6px 6px;
                background: {PANEL_BG};
            }}
            QTabBar::tab {{
                background: {DARK_BG};
                color: {SUBTEXT};
                padding: 9px 18px;
                border-radius: 6px 6px 0 0;
                margin-right: 3px;
                font-weight: 600;
                border: 1.5px solid {BORDER};
                border-bottom: none;
            }}
            QTabBar::tab:selected {{
                background: {PANEL_BG};
                color: {ACCENT};
                font-weight: bold;
            }}
            QTabBar::tab:hover:!selected {{ color: {TEXT}; }}
            QScrollBar:vertical {{
                background: {DARK_BG}; width: 8px; border-radius: 4px;
            }}
            QScrollBar::handle:vertical {{
                background: {BORDER}; border-radius: 4px; min-height: 24px;
            }}
            QScrollBar::handle:vertical:hover {{ background: {ACCENT}; }}
            QScrollBar::add-line:vertical, QScrollBar::sub-line:vertical {{ height: 0; }}
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
        tabs.setFont(QFont("Helvetica Neue", 10))

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

        # Tab 3 — Graph Visualization & Playback
        self.graph_tab = SimulationReplayTab()
        tabs.addTab(self.graph_tab, "🕸️  Simulation Replay")

        # Tab 3 — Complexity Table
        tabs.addTab(ComplexityTable(), "🗂  Complexity Reference")

        # Tab 4 — Quantum Info
        tabs.addTab(QuantumInfoTab(), "⚛  Quantum Module")

        root.addWidget(tabs, 1)

    def _build_sidebar(self):
        """Sidebar wrapped in QScrollArea — scrolls vertically, never clips."""
        # Outer fixed-width container
        outer = QWidget()
        outer.setFixedWidth(285)
        outer_layout = QVBoxLayout(outer)
        outer_layout.setContentsMargins(0, 0, 0, 0)
        outer_layout.setSpacing(0)

        # Scroll area wrapping all controls
        sidebar_scroll = QScrollArea()
        sidebar_scroll.setWidgetResizable(True)
        sidebar_scroll.setHorizontalScrollBarPolicy(Qt.ScrollBarAlwaysOff)
        sidebar_scroll.setVerticalScrollBarPolicy(Qt.ScrollBarAsNeeded)
        sidebar_scroll.setStyleSheet(
            f"QScrollArea {{ border: none; background: {DARK_BG}; }}"
        )

        interior = QWidget()
        layout = QVBoxLayout(interior)
        layout.setSpacing(10)
        layout.setContentsMargins(8, 8, 8, 16)

        # ── Title ─────────────────────────────────────────────────────────────
        title = QLabel("Shortest Path in Changing Edge Weights")
        title.setStyleSheet(f"color:{ACCENT}; font-size:16px; font-weight:bold; padding:4px 0;")
        layout.addWidget(title)

        sub = QLabel("Dynamic Shortest-Path Visualizer")
        sub.setStyleSheet(f"color:{SUBTEXT}; font-size:9px; padding-bottom:4px;")
        layout.addWidget(sub)

        # ── Graph config ──────────────────────────────────────────────────────
        grp_graph = QGroupBox("📊  Graph Configuration")
        gg = QVBoxLayout(grp_graph)
        gg.setSpacing(4)

        def add_slider(parent_layout, label, lo, hi, default):
            row = QHBoxLayout()
            lbl = QLabel(label)
            lbl.setStyleSheet(f"color:{TEXT}; font-size:10px; font-weight:600;")
            val = QLabel(str(default))
            val.setStyleSheet(f"color:{ACCENT}; font-size:10px; font-weight:bold;")
            val.setAlignment(Qt.AlignRight)
            row.addWidget(lbl); row.addStretch(); row.addWidget(val)
            sld = QSlider(Qt.Horizontal)
            sld.setRange(lo, hi); sld.setValue(default)
            sld.valueChanged.connect(lambda v, w=val: w.setText(str(v)))
            parent_layout.addLayout(row)
            parent_layout.addWidget(sld)
            return sld, val

        self.sld_nodes,   self.lbl_nodes   = add_slider(gg, "Nodes",   4,  300,  60)
        self.sld_edges,   self.lbl_edges   = add_slider(gg, "Edges",   4, 1000, 150)
        self.sld_updates, self.lbl_updates = add_slider(gg, "Updates", 4,  200,  60)
        layout.addWidget(grp_graph)

        # ── Weight config ─────────────────────────────────────────────────────
        grp_weights = QGroupBox("⚖️  Weight Change Mode")
        gw = QVBoxLayout(grp_weights)
        gw.setSpacing(6)
        self.cmb_mode = QComboBox()
        self.cmb_mode.addItems([
            "mixed  (↑ and ↓)",
            "decrease only  (↓)",
            "increase only  (↑)",
            "traffic  (big swings)",
        ])
        gw.addWidget(self.cmb_mode)
        self.chk_negative = QCheckBox("Allow Negative Weights")
        self.chk_negative.setStyleSheet(f"color:{WARN}; font-weight:bold;")
        self.chk_negative.stateChanged.connect(self._update_warning)
        gw.addWidget(self.chk_negative)
        layout.addWidget(grp_weights)

        self.lbl_warning = QLabel("")
        self.lbl_warning.setStyleSheet(f"color:{RED}; font-weight:bold; font-size:10px; padding:4px; border:1px solid {RED}; border-radius:4px; background:{PANEL_BG}; margin-top:5px;")
        self.lbl_warning.setWordWrap(True)
        self.lbl_warning.hide()
        layout.addWidget(self.lbl_warning)

        # ── Algorithm selection ───────────────────────────────────────────────
        grp_algos = QGroupBox("🧠  Algorithms  (check to enable)")
        ga = QVBoxLayout(grp_algos)
        ga.setSpacing(5)
        ALGO_CLASSES = [DijkstraRerun, BellmanFordRerun, DynamicBellmanFord,
                        RamalingamReps, LPAStar, QuantumSSSP]
        for AlgoClass in ALGO_CLASSES:
            display_name = AlgoClass.name if AlgoClass.supports_negative else f"{AlgoClass.name} (Non-negative only)"
            c = ALGO_COLORS.get(AlgoClass.name, TEXT)
            row = QHBoxLayout()
            dot = QLabel("●")
            dot.setStyleSheet(f"color:{c}; font-size:13px;")
            dot.setFixedWidth(16)
            cb = QCheckBox(display_name)
            cb.setChecked(AlgoClass.name not in ["Bellman-Ford Full Rerun", "Quantum SSSP (stub)"])
            cb.setStyleSheet(f"color:{c}; font-weight:bold; font-size:10px;")
            cb.stateChanged.connect(lambda state, ac=AlgoClass: self._toggle_algo(ac, state))
            cb.stateChanged.connect(self._update_warning)
            self.checkboxes[AlgoClass.name] = cb
            row.addWidget(dot); row.addWidget(cb); row.addStretch()
            ga.addLayout(row)
        layout.addWidget(grp_algos)
        
        self._update_warning()

        # ── Controls ──────────────────────────────────────────────────────────
        grp_ctrl = QGroupBox("▶  Controls")
        gc = QVBoxLayout(grp_ctrl)
        gc.setSpacing(6)

        self.btn_run = QPushButton("▶   Run Benchmark")
        self.btn_run.setMinimumHeight(36)
        self.btn_run.clicked.connect(self._run_benchmark)
        gc.addWidget(self.btn_run)

        self.btn_stop = QPushButton("⏹   Stop")
        self.btn_stop.setMinimumHeight(32)
        self.btn_stop.setEnabled(False)
        self.btn_stop.clicked.connect(self._stop_benchmark)
        self.btn_stop.setStyleSheet(
            f"background:{RED}; color:white; border-radius:6px; "
            f"font-weight:bold; padding:6px;"
        )
        gc.addWidget(self.btn_stop)

        self.btn_reset = QPushButton("🔄   Reset")
        self.btn_reset.setMinimumHeight(32)
        self.btn_reset.clicked.connect(self._reset)
        self.btn_reset.setStyleSheet(
            f"background:{SUBTEXT}; color:white; border-radius:6px; "
            f"font-weight:bold; padding:6px;"
        )
        gc.addWidget(self.btn_reset)
        layout.addWidget(grp_ctrl)

        # ── Status ────────────────────────────────────────────────────────────
        self.lbl_status = QLabel("Ready")
        self.lbl_status.setStyleSheet(
            f"color:{ACCENT2}; font-size:11px; font-weight:bold; "
            f"padding:6px 8px; background:{PANEL_BG}; "
            f"border-radius:6px; border:1px solid {BORDER};"
        )
        self.lbl_status.setWordWrap(True)
        layout.addWidget(self.lbl_status)

        # ── Legend ────────────────────────────────────────────────────────────
        legend = QGroupBox("Legend")
        lg = QVBoxLayout(legend)
        lg.setSpacing(3)
        for name, color in ALGO_COLORS.items():
            row = QHBoxLayout()
            dot = QLabel("●")
            dot.setStyleSheet(f"color:{color}; font-size:13px;")
            dot.setFixedWidth(18)
            lbl = QLabel(name.split("(")[0].strip())
            lbl.setStyleSheet(f"color:{TEXT}; font-size:9px;")
            row.addWidget(dot); row.addWidget(lbl); row.addStretch()
            lg.addLayout(row)
        layout.addWidget(legend)

        layout.addStretch(1)

        sidebar_scroll.setWidget(interior)
        outer_layout.addWidget(sidebar_scroll)
        return outer

    # ── Warning logic ─────────────────────────────────────────────────────────
    def _update_warning(self):
        if not self.chk_negative.isChecked():
            self.lbl_warning.hide()
            return
            
        incompatible_selected = []
        for name, cb in self.checkboxes.items():
            if cb.isChecked():
                if name in ["Dijkstra Full Rerun", "Ramalingam-Reps (RR-SSSP)", "LPA*"]:
                    incompatible_selected.append(name.split("(")[0].strip())
        
        if incompatible_selected:
            names_str = ', '.join(incompatible_selected)
            self.lbl_warning.setText(f"⚠️ {names_str} cannot handle negative weights. They will fail and show 0 nodes visited. Please uncheck them or disable negative weights.")
            self.lbl_warning.show()
        else:
            self.lbl_warning.hide()

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
        """Re-flow algo panels into a responsive grid.
        Each panel is ~370px wide; column count adapts to available space.
        """
        while self.panels_layout.count():
            item = self.panels_layout.takeAt(0)
            if item.widget():
                item.widget().setParent(None)

        # Sidebar is 285px; subtract it plus margins to find available width
        available = max(370, self.width() - 310)
        cols = max(1, available // 390)

        for i, (name, panel) in enumerate(self.algo_panels.items()):
            self.panels_layout.addWidget(panel, i // cols, i % cols)

        # stretch last row so panels don't spread unevenly
        self.panels_layout.setRowStretch(
            (len(self.algo_panels) - 1) // cols + 1, 1
        )

    def resizeEvent(self, event):
        """Re-flow panels whenever the window is resized."""
        super().resizeEvent(event)
        if hasattr(self, 'panels_layout'):
            self._relayout_panels()

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

        # Collect active algo classes — keys must match AlgoClass.name exactly
        ALGO_MAP = {
            "Dijkstra Full Rerun":        DijkstraRerun,
            "Bellman-Ford Full Rerun":    BellmanFordRerun,
            "Dynamic Bellman-Ford":       DynamicBellmanFord,
            "Ramalingam-Reps (RR-SSSP)": RamalingamReps,
            "LPA*":                       LPAStar,
            "Quantum SSSP (stub)":        QuantumSSSP,
        }
        active = [ALGO_MAP[n] for n, cb in self.checkboxes.items()
                  if cb.isChecked() and n in ALGO_MAP]
        if not active:
            self.lbl_status.setText("Select at least one algorithm."); return

        if allow_neg:
            incompatible = [ac.name for ac in active if not ac.supports_negative]
            if incompatible:
                QMessageBox.warning(
                    self, 
                    "Incompatible Settings", 
                    f"The following selected algorithms do not support negative weights:\n\n"
                    f"{', '.join(incompatible)}\n\n"
                    f"Please uncheck them or disable negative weights before running."
                )
                return

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
        if hasattr(self, 'graph_tab') and hasattr(self.graph_tab, 'load_simulation_file'):
            self.graph_tab.load_simulation_file()


# ─────────────────────────────────────────────────────────────────────────────
# ENTRY POINT
# ─────────────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    app = QApplication(sys.argv)
    app.setFont(QFont("Helvetica Neue", 10))   # macOS system font
    win = MainWindow()

    # Initialize panels for default-checked algorithms
    # Names must match AlgoClass.name exactly
    default_checked = ["Dijkstra Full Rerun", "Dynamic Bellman-Ford",
                       "Ramalingam-Reps (RR-SSSP)", "LPA*"]
    for name in default_checked:
        if name not in win.algo_panels:
            win.algo_panels[name] = AlgoPanel(name)
    win._relayout_panels()

    win.show()
    sys.exit(app.exec_())