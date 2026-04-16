# gui.py
# TDSPP Benchmark GUI — 9-node TDRN, Q(v0, v8)
# White background, millisecond timing, departure times t=0, t=30, t=50.

import sys, os, time
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from PyQt5.QtWidgets import (
    QApplication, QMainWindow, QWidget, QVBoxLayout, QHBoxLayout,
    QLabel, QCheckBox, QPushButton, QGroupBox,
    QTableWidget, QTableWidgetItem, QHeaderView, QTabWidget, QSplitter
)
from PyQt5.QtCore import Qt, QThread, pyqtSignal
from PyQt5.QtGui import QFont, QColor

import matplotlib
matplotlib.use("Qt5Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.figure import Figure
import numpy as np

from algorithms.graph import build_wang_graph
from algorithms import ALL_ALGORITHMS

SOURCE         = 0
DESTINATION    = 8
DEPARTURE_TIMES = [0, 30, 50]

# Node positions matching the graph image layout
NODE_POS = {
    0: (1.0, 5.2),   # v0 top-left
    1: (3.0, 5.2),   # v1 top-center
    2: (1.8, 3.8),   # v2 mid-left
    3: (1.4, 2.2),   # v3 bottom-left
    4: (2.4, 0.9),   # v4 bottom-center
    5: (3.2, 3.0),   # v5 center (connector)
    6: (4.8, 4.4),   # v6 right-top
    7: (5.6, 3.0),   # v7 right-mid
    8: (5.6, 1.6),   # v8 right-bottom (destination)
}

# Border nodes — connect clusters
BORDER_NODES = {1, 2, 3, 5, 6}

ALGO_COLORS = {
    "TD-Dijkstra": "#1976d2",
    "TD-A*":       "#388e3c",
    "TD-G-Tree":   "#e65100",
}

# Time-varying edges to label on graph
VARYING_EDGES = {
    (1, 2): "PLF",
    (1, 6): "PLF★",   # the critical edge
    (3, 5): "PLF",
    (5, 6): "PLF",
}


# ── Worker ────────────────────────────────────────────────────────────────

class BenchmarkWorker(QThread):
    result_ready = pyqtSignal(dict)
    status_msg   = pyqtSignal(str)

    def __init__(self, graph, selected_algos):
        super().__init__()
        self.graph          = graph
        self.selected_algos = selected_algos

    def run(self):
        all_results = {}
        for algo in self.selected_algos:
            self.status_msg.emit(f"Running {algo.ALGORITHM_NAME}...")
            rows = []
            for t0 in DEPARTURE_TIMES:
                g = build_wang_graph()
                n_repeats = 20
                t_start = time.perf_counter()
                for _ in range(n_repeats):
                    r = algo.run(g, SOURCE, DESTINATION, t0)
                r["elapsed_ms"] = (time.perf_counter() - t_start) / n_repeats * 1000
                r["t0"] = t0
                rows.append(r)
            all_results[algo.ALGORITHM_NAME] = rows
        self.result_ready.emit(all_results)
        self.status_msg.emit("Done.")


# ── Graph canvas ──────────────────────────────────────────────────────────

class GraphCanvas(FigureCanvas):

    def __init__(self):
        fig = Figure(figsize=(5, 4.5), facecolor="white")
        super().__init__(fig)
        self.ax    = fig.add_subplot(111)
        self.graph = build_wang_graph()
        self.redraw()

    def redraw(self, highlighted_paths=None):
        self.ax.clear()
        self.ax.set_facecolor("white")
        self.figure.patch.set_facecolor("white")
        self.ax.set_xlim(0.2, 6.6)
        self.ax.set_ylim(0.3, 6.0)
        self.ax.axis("off")
        self.ax.set_title("9-node TDRN  |  source v0  →  destination v8",
                           fontsize=9, color="#333333", pad=5, fontweight="bold")

        # Draw edges
        drawn = set()
        for (u, v) in self.graph.edges:
            key = tuple(sorted([u, v]))
            if key in drawn:
                continue
            drawn.add(key)
            x0, y0 = NODE_POS[u]
            x1, y1 = NODE_POS[v]
            is_varying = key in {tuple(sorted(k)) for k in VARYING_EDGES}
            color = "#e65100" if is_varying else "#cccccc"
            lw    = 1.8 if is_varying else 1.2
            ls    = (0, (5, 3)) if is_varying else "solid"
            self.ax.plot([x0, x1], [y0, y1], color=color,
                         linewidth=lw, linestyle=ls, zorder=2)

        # Highlighted paths
        if highlighted_paths:
            algo_list = list(highlighted_paths.keys())
            for algo_name, path in highlighted_paths.items():
                color = ALGO_COLORS.get(algo_name, "#333333")
                idx   = algo_list.index(algo_name)
                delta = (idx - len(algo_list) / 2) * 0.08
                for i in range(len(path) - 1):
                    u, v = path[i], path[i + 1]
                    x0, y0 = NODE_POS[u]
                    x1, y1 = NODE_POS[v]
                    self.ax.annotate("",
                        xy=(x1 + delta, y1 + delta),
                        xytext=(x0 + delta, y0 + delta),
                        arrowprops=dict(arrowstyle="->", color=color, lw=2.2,
                                        connectionstyle="arc3,rad=0.05"),
                        zorder=5)

        # PLF edge labels
        labeled = set()
        for (u, v), label in VARYING_EDGES.items():
            key = tuple(sorted([u, v]))
            if key in labeled:
                continue
            labeled.add(key)
            x0, y0 = NODE_POS[u]
            x1, y1 = NODE_POS[v]
            mx, my = (x0+x1)/2, (y0+y1)/2 + 0.22
            self.ax.text(mx, my, label, fontsize=6, color="#bf360c",
                         ha="center", va="center",
                         bbox=dict(facecolor="white", edgecolor="#bf360c",
                                   boxstyle="round,pad=0.12", alpha=0.9),
                         zorder=7)

        # Draw nodes
        for nid, (x, y) in NODE_POS.items():
            if nid == SOURCE:
                fc, ec, r, tc = "#1565c0", "#0d47a1", 0.27, "white"
            elif nid == DESTINATION:
                fc, ec, r, tc = "#b71c1c", "#7f0000", 0.27, "white"
            elif nid in BORDER_NODES:
                fc, ec, r, tc = "#00838f", "#004d40", 0.23, "white"
            else:
                fc, ec, r, tc = "#f5f5f5", "#9e9e9e", 0.20, "#333333"

            circle = plt.Circle((x, y), r, facecolor=fc,
                                 edgecolor=ec, linewidth=1.4, zorder=6)
            self.ax.add_patch(circle)
            self.ax.text(x, y, f"v{nid}", ha="center", va="center",
                         fontsize=7.5, fontweight="bold", color=tc, zorder=7)

        items = [
            mpatches.Patch(color="#1565c0", label="Source (v0)"),
            mpatches.Patch(color="#b71c1c", label="Dest (v8)"),
            mpatches.Patch(color="#00838f", label="Border node"),
            mpatches.Patch(color="#f5f5f5", label="Internal node"),
            mpatches.Patch(color="#e65100", label="Time-varying edge"),
        ]
        if highlighted_paths:
            for name in highlighted_paths:
                items.append(mpatches.Patch(
                    color=ALGO_COLORS.get(name, "#333"), label=name))
        self.ax.legend(handles=items, loc="lower right", fontsize=7,
                       framealpha=0.95, edgecolor="#cccccc", facecolor="white")
        self.figure.tight_layout()
        self.draw()

    def show_paths(self, paths_dict):
        self.redraw(highlighted_paths=paths_dict)


# ── Result bar chart ──────────────────────────────────────────────────────

class ResultChart(FigureCanvas):

    def __init__(self):
        fig = Figure(figsize=(5, 4.5), facecolor="white")
        super().__init__(fig)
        self.fig = fig
        self.ax1 = fig.add_subplot(211)
        self.ax2 = fig.add_subplot(212)
        self._style()
        fig.tight_layout(pad=1.8)

    def _style(self):
        for ax in [self.ax1, self.ax2]:
            ax.set_facecolor("#fafafa")
            ax.tick_params(labelsize=7, colors="#333333")
            for sp in ax.spines.values():
                sp.set_edgecolor("#dddddd")

    def refresh_chart(self, all_results):
        self.ax1.clear(); self.ax2.clear(); self._style()

        names  = list(all_results.keys())
        colors = [ALGO_COLORS.get(n, "#333") for n in names]
        short  = [n.replace("TD-Dijkstra","Dijkstra")
                   .replace("TD-A*","A*")
                   .replace("TD-G-Tree","G-Tree") for n in names]

        avg_ms    = [sum(r["elapsed_ms"]    for r in all_results[n]) / len(all_results[n]) for n in names]
        avg_nodes = [sum(r["nodes_settled"] for r in all_results[n]) / len(all_results[n]) for n in names]

        b1 = self.ax1.bar(range(len(names)), avg_ms, color=colors, alpha=0.85,
                           width=0.5, edgecolor="white")
        self.ax1.set_title("Avg Query Time (ms)", fontsize=9, color="#333333",
                            fontweight="bold", pad=4)
        self.ax1.set_xticks(range(len(names)))
        self.ax1.set_xticklabels(short, fontsize=8)
        self.ax1.set_ylabel("milliseconds", fontsize=7, color="#555555")
        margin = max(avg_ms) * 0.05 if avg_ms else 0.001
        for bar, val in zip(b1, avg_ms):
            self.ax1.text(bar.get_x() + bar.get_width()/2,
                           bar.get_height() + margin,
                           f"{val:.4f}", ha="center", fontsize=7)

        b2 = self.ax2.bar(range(len(names)), avg_nodes, color=colors, alpha=0.85,
                           width=0.5, edgecolor="white")
        self.ax2.set_title("Avg Nodes Settled", fontsize=9, color="#333333",
                            fontweight="bold", pad=4)
        self.ax2.set_xticks(range(len(names)))
        self.ax2.set_xticklabels(short, fontsize=8)
        self.ax2.set_ylabel("nodes", fontsize=7, color="#555555")
        margin2 = max(avg_nodes) * 0.05 if avg_nodes else 0.1
        for bar, val in zip(b2, avg_nodes):
            self.ax2.text(bar.get_x() + bar.get_width()/2,
                           bar.get_height() + margin2,
                           f"{val:.1f}", ha="center", fontsize=7)

        self.fig.tight_layout(pad=1.8)
        self.draw()


# ── PLF chart ─────────────────────────────────────────────────────────────

class PLFCanvas(FigureCanvas):

    def __init__(self, graph):
        fig = Figure(figsize=(6, 3.5), facecolor="white")
        super().__init__(fig)
        self.fig   = fig
        self.ax    = fig.add_subplot(111)
        self.graph = graph
        self._draw()

    def _draw(self):
        self.ax.clear()
        self.ax.set_facecolor("#fafafa")
        self.ax.tick_params(labelsize=8, colors="#333333")
        for sp in self.ax.spines.values():
            sp.set_edgecolor("#dddddd")

        t = np.linspace(0, 60, 300)
        lines = [
            ((1, 6), "#c62828", "e(v1,v6)  ★ critical — spikes at t=30"),
            ((1, 2), "#1565c0", "e(v1,v2)  rises from t=20"),
            ((3, 5), "#2e7d32", "e(v3,v5)  drops from t=20"),
            ((5, 6), "#e65100", "e(v5,v6)  rises from t=25"),
        ]
        for (u, v), color, label in lines:
            costs = [self.graph.get_cost(u, v, ti) for ti in t]
            self.ax.plot(t, costs, color=color, lw=2.0, label=label)

        for dep in DEPARTURE_TIMES:
            self.ax.axvline(x=dep, color="#aaaaaa", lw=0.8, ls="--", alpha=0.6)
            self.ax.text(dep+0.5, self.ax.get_ylim()[1] if self.ax.get_ylim()[1] > 5 else 22,
                         f"t={dep}", fontsize=7, color="#888888")

        self.ax.set_xlabel("Departure Time (minutes)", fontsize=8)
        self.ax.set_ylabel("Edge Cost (minutes)", fontsize=8)
        self.ax.set_title("Piecewise Linear Weight Functions — Time-Varying Edges",
                           fontsize=9, fontweight="bold")
        self.ax.legend(fontsize=7.5, framealpha=0.95,
                        edgecolor="#cccccc", facecolor="white")
        self.fig.tight_layout()
        self.draw()


# ── Main window ───────────────────────────────────────────────────────────

class MainWindow(QMainWindow):

    def __init__(self):
        super().__init__()
        self.setWindowTitle("TDSPP Benchmark  |  9-node TDRN  |  Q(v0, v8)")
        self.resize(1300, 800)
        self.setStyleSheet("QMainWindow, QWidget { background-color: white; color: #222222; }")
        self.graph  = build_wang_graph()
        self.worker = None
        self._build_ui()

    def _build_ui(self):
        central = QWidget()
        self.setCentralWidget(central)
        root = QHBoxLayout(central)
        root.setSpacing(10)
        root.setContentsMargins(10, 10, 10, 10)
        root.addWidget(self._sidebar(), 0)
        root.addWidget(self._tabs(), 1)

    def _grp_style(self):
        return """
            QGroupBox { border:1px solid #dddddd; border-radius:5px;
                        margin-top:6px; font-weight:bold; color:#333333; padding:5px; }
            QGroupBox::title { subcontrol-origin:margin; left:8px; top:-4px; }
        """

    def _sidebar(self):
        side = QWidget()
        side.setFixedWidth(220)
        side.setStyleSheet("QWidget { background: white; }")
        lay = QVBoxLayout(side)
        lay.setSpacing(8)
        lay.setContentsMargins(0, 0, 0, 0)

        title = QLabel("TDSPP Benchmark")
        title.setStyleSheet("font-size:14px; font-weight:bold; color:#1565c0; padding:4px;")
        lay.addWidget(title)

        info = QLabel("9-node TDRN\nQuery: Q(v0, v8)\nDepartures: t = 0, 30, 50")
        info.setStyleSheet("font-size:9px; color:#666666; padding:2px 4px;")
        lay.addWidget(info)

        grp_a = QGroupBox("Algorithms")
        grp_a.setStyleSheet(self._grp_style())
        ga = QVBoxLayout(grp_a)
        self.checkboxes = {}
        for algo in ALL_ALGORITHMS:
            cb = QCheckBox(algo.ALGORITHM_NAME)
            cb.setChecked(True)
            c = ALGO_COLORS.get(algo.ALGORITHM_NAME, "#333333")
            cb.setStyleSheet(f"""
                QCheckBox {{ color:{c}; font-weight:bold; font-size:10px; }}
                QCheckBox::indicator {{ width:13px; height:13px;
                    border:1px solid #aaaaaa; border-radius:2px; background:white; }}
                QCheckBox::indicator:checked {{ background:{c}; border-color:{c}; }}
            """)
            self.checkboxes[algo.ALGORITHM_NAME] = cb
            ga.addWidget(cb)
        lay.addWidget(grp_a)

        self.btn = QPushButton("Run Benchmark")
        self.btn.setStyleSheet("""
            QPushButton { background:#1565c0; color:white; border:none;
                          border-radius:5px; padding:9px; font-weight:bold; font-size:11px; }
            QPushButton:hover    { background:#1976d2; }
            QPushButton:disabled { background:#bbbbbb; color:#888888; }
        """)
        self.btn.clicked.connect(self._run)
        lay.addWidget(self.btn)

        self.lbl_status = QLabel("Ready")
        self.lbl_status.setStyleSheet("color:#2e7d32; font-size:10px; padding:3px;")
        self.lbl_status.setWordWrap(True)
        lay.addWidget(self.lbl_status)

        grp_ref = QGroupBox("Complexity")
        grp_ref.setStyleSheet(self._grp_style())
        gr = QVBoxLayout(grp_ref)
        for name, cx, color in [
            ("TD-Dijkstra", "O((V+E)logV·f)", "#1976d2"),
            ("TD-A*",       "O((V+E)logV·f)", "#388e3c"),
            ("TD-G-Tree",   "O(log²(kf)·V)",  "#e65100"),
        ]:
            rw = QWidget(); rl = QHBoxLayout(rw); rl.setContentsMargins(0,0,0,0)
            nl = QLabel(name); nl.setStyleSheet(f"color:{color}; font-size:8.5px; font-weight:bold;")
            cl = QLabel(cx);   cl.setStyleSheet("color:#777777; font-size:7.5px;")
            rl.addWidget(nl); rl.addStretch(); rl.addWidget(cl)
            gr.addWidget(rw)
        fn = QLabel("f = PLF breakpoints per edge")
        fn.setStyleSheet("color:#aaaaaa; font-size:7px;")
        gr.addWidget(fn)
        lay.addWidget(grp_ref)

        lay.addStretch()
        return side

    def _tbl_style(self):
        return """
            QTableWidget { background:white; color:#222222;
                           gridline-color:#eeeeee; border:1px solid #dddddd; }
            QTableWidget::item:alternate { background:#f9f9f9; }
            QHeaderView::section { background:#f5f5f5; color:#333333;
                                   font-weight:bold; padding:5px;
                                   border:none; border-bottom:1px solid #dddddd; }
        """

    def _tabs(self):
        tabs = QTabWidget()
        tabs.setStyleSheet("""
            QTabWidget::pane { border:1px solid #dddddd; background:white; }
            QTabBar::tab { background:#f5f5f5; color:#555555; padding:7px 14px;
                           border-radius:4px 4px 0 0; margin-right:2px;
                           border:1px solid #dddddd; }
            QTabBar::tab:selected { background:white; color:#1565c0;
                                    font-weight:bold; border-bottom:none; }
        """)

        # Tab 1 — Graph + charts
        tab1 = QWidget(); t1 = QVBoxLayout(tab1); t1.setContentsMargins(6,6,6,6)
        top = QWidget(); top_lay = QHBoxLayout(top); top_lay.setContentsMargins(0,0,0,0)
        self.graph_canvas = GraphCanvas()
        self.result_chart = ResultChart()
        top_lay.addWidget(self.graph_canvas, 1)
        top_lay.addWidget(self.result_chart, 1)

        self.summary_table = QTableWidget()
        self.summary_table.setColumnCount(5)
        self.summary_table.setHorizontalHeaderLabels([
            "Algorithm", "t=0", "t=30", "t=50", "Avg Query (ms)"])
        self.summary_table.horizontalHeader().setSectionResizeMode(QHeaderView.Stretch)
        self.summary_table.setEditTriggers(QTableWidget.NoEditTriggers)
        self.summary_table.setAlternatingRowColors(True)
        self.summary_table.setStyleSheet(self._tbl_style())

        splitter = QSplitter(Qt.Vertical)
        splitter.addWidget(top)
        splitter.addWidget(self.summary_table)
        splitter.setSizes([480, 180])
        t1.addWidget(splitter)
        tabs.addTab(tab1, "Graph and Results")

        # Tab 2 — PLF
        tab2 = QWidget(); t2 = QVBoxLayout(tab2); t2.setContentsMargins(6,6,6,6)
        self.plf_canvas = PLFCanvas(self.graph)
        t2.addWidget(self.plf_canvas)
        tabs.addTab(tab2, "Edge Weight Functions")

        # Tab 3 — Per departure table
        tab3 = QWidget(); t3 = QVBoxLayout(tab3); t3.setContentsMargins(6,6,6,6)
        t3.addWidget(QLabel("  Query time (ms) per departure time:"))
        self.tbl_by_time = QTableWidget()
        cols = ["t (min)"] + [a.ALGORITHM_NAME for a in ALL_ALGORITHMS]
        self.tbl_by_time.setColumnCount(len(cols))
        self.tbl_by_time.setHorizontalHeaderLabels(cols)
        self.tbl_by_time.horizontalHeader().setSectionResizeMode(QHeaderView.Stretch)
        self.tbl_by_time.setEditTriggers(QTableWidget.NoEditTriggers)
        self.tbl_by_time.setAlternatingRowColors(True)
        self.tbl_by_time.setStyleSheet(self._tbl_style())
        t3.addWidget(self.tbl_by_time)
        tabs.addTab(tab3, "Results by Departure Time")

        return tabs

    def _run(self):
        selected = [a for a in ALL_ALGORITHMS
                    if self.checkboxes[a.ALGORITHM_NAME].isChecked()]
        if not selected:
            self.lbl_status.setText("Select at least one algorithm.")
            return
        self.btn.setEnabled(False)
        self.lbl_status.setText("Running...")
        self.worker = BenchmarkWorker(self.graph, selected)
        self.worker.status_msg.connect(self.lbl_status.setText)
        self.worker.result_ready.connect(self._on_done)
        self.worker.start()

    def _on_done(self, all_results):
        self.btn.setEnabled(True)

        # Summary table — one row per algorithm, columns = t=0,30,50
        self.summary_table.setRowCount(len(all_results))
        for row, (name, rows) in enumerate(all_results.items()):
            color = ALGO_COLORS.get(name, "#222222")
            avg_ms = sum(r["elapsed_ms"] for r in rows) / len(rows)
            vals = [name] + [f"{r['travel_time']:.2f} min" for r in rows] + [f"{avg_ms:.4f} ms"]
            for col, val in enumerate(vals):
                item = QTableWidgetItem(val)
                item.setTextAlignment(Qt.AlignCenter)
                if col == 0:
                    item.setForeground(QColor(color))
                self.summary_table.setItem(row, col, item)

        # Per departure time table
        algo_names = list(all_results.keys())
        self.tbl_by_time.setRowCount(len(DEPARTURE_TIMES))
        for row, t0 in enumerate(DEPARTURE_TIMES):
            t0_item = QTableWidgetItem(str(t0))
            t0_item.setTextAlignment(Qt.AlignCenter)
            self.tbl_by_time.setItem(row, 0, t0_item)
            for col, name in enumerate(algo_names):
                r    = all_results[name][row]
                item = QTableWidgetItem(f"{r['elapsed_ms']:.4f} ms")
                item.setTextAlignment(Qt.AlignCenter)
                item.setForeground(QColor(ALGO_COLORS.get(name, "#222222")))
                self.tbl_by_time.setItem(row, col + 1, item)

        self.result_chart.refresh_chart(all_results)

        paths = {name: all_results[name][0]["path"]
                 for name in all_results if all_results[name][0]["path"]}
        self.graph_canvas.show_paths(paths)


def run_gui():
    app = QApplication(sys.argv)
    app.setFont(QFont("Segoe UI", 9))
    win = MainWindow()
    win.show()
    sys.exit(app.exec_())


if __name__ == "__main__":
    run_gui()
