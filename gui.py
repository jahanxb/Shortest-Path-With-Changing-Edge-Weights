# gui.py
# TDSPP Benchmark GUI — 9-node TDRN, Q(v0, v8)
# Animated step-by-step trace showing nodes being settled one at a time.
# Graph layout matches the experiment slide image exactly.

import sys, os, time
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from PyQt5.QtWidgets import (
    QApplication, QMainWindow, QWidget, QVBoxLayout, QHBoxLayout,
    QLabel, QCheckBox, QPushButton, QGroupBox, QComboBox,
    QTableWidget, QTableWidgetItem, QHeaderView, QTabWidget,
    QSplitter, QSlider, QFrame
)
from PyQt5.QtCore import Qt, QThread, QTimer, pyqtSignal
from PyQt5.QtGui import QFont, QColor

import matplotlib
matplotlib.use("Qt5Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.figure import Figure
import numpy as np
import heapq

from algorithms.graph import build_wang_graph
from algorithms import ALL_ALGORITHMS

SOURCE          = 0
DESTINATION     = 8
DEPARTURE_TIMES = [0, 30, 50]

# ── Node positions matching the graph image (v0 top-left, v8 bottom-right) ──
NODE_POS = {
    0: (1.0, 5.2),   # v0 top-left  (source)
    1: (3.2, 5.2),   # v1 top-center
    2: (1.8, 3.7),   # v2 mid-left
    3: (1.3, 2.1),   # v3 bottom-left
    4: (2.3, 0.9),   # v4 bottom-center
    5: (3.3, 2.9),   # v5 center connector
    6: (5.0, 4.3),   # v6 right-top
    7: (5.8, 2.9),   # v7 right-mid
    8: (5.8, 1.5),   # v8 right-bottom (destination)
}

BORDER_NODES = {1, 2, 3, 5, 6}

ALGO_COLORS = {
    "TD-Dijkstra": "#1565c0",
    "TD-A*":       "#2e7d32",
    "TD-G-Tree":   "#bf360c",
}

# Time-varying edges
VARYING_EDGES = {(1,2), (1,6), (3,5), (5,6)}

# Edge list (undirected pairs)
EDGES = [(0,1),(0,2),(1,2),(1,6),(2,3),(2,5),(3,4),(3,5),(5,6),(6,7),(7,8)]


# ── Compute detailed step trace for an algorithm ──────────────────────────

def get_cost_direct(edges_dict, u, v, t):
    key = (u,v) if (u,v) in edges_dict else (v,u)
    if key not in edges_dict: return float('inf')
    pts = edges_dict[key]
    if t <= pts[0][0]: return pts[0][1]
    if t >= pts[-1][0]: return pts[-1][1]
    for i in range(len(pts)-1):
        t0,w0=pts[i]; t1,w1=pts[i+1]
        if t0<=t<=t1: return round(w0+(w1-w0)/(t1-t0)*(t-t0),2)
    return pts[-1][1]


def compute_trace(algo_name, departure_time):
    """Returns list of step dicts for animation."""
    graph = build_wang_graph()
    edges_dict = graph.edges

    steps   = []  # {node, elapsed, via_node, cost, arrival, status}
    path    = []
    total   = 0.0

    if algo_name == "TD-Dijkstra":
        dist = {n: float('inf') for n in range(9)}
        prev = {n: None for n in range(9)}
        dist[0] = departure_time
        heap = [(departure_time, 0)]
        settled = set()

        while heap:
            arr, u = heapq.heappop(heap)
            if u in settled: continue
            settled.add(u)
            via = prev[u]
            cost_here = round(arr - (dist[prev[u]] if via is not None else departure_time), 2) if via is not None else 0
            steps.append({
                "node": u, "arrival": round(arr, 2),
                "elapsed": round(arr - departure_time, 2),
                "via": via, "cost": cost_here,
                "status": "settled"
            })
            if u == 8: break
            for (a,b) in EDGES:
                for nu, nv in [(a,b),(b,a)]:
                    if nu == u and nv not in settled:
                        c = get_cost_direct(edges_dict, nu, nv, arr)
                        if c != float('inf') and arr+c < dist[nv]:
                            dist[nv] = arr+c; prev[nv] = nu
                            heapq.heappush(heap, (arr+c, nv))

        cur = 8
        while cur is not None: path.append(cur); cur = prev[cur]
        path.reverse()
        total = round(dist[8] - departure_time, 2)

    elif algo_name == "TD-A*":
        # Build heuristic
        min_cost = {}
        for (a,b), plf in edges_dict.items():
            min_cost[(a,b)] = min(w for _,w in plf)
        h = {n: float('inf') for n in range(9)}
        h[8] = 0
        hheap = [(0, 8)]; hsettled = set()
        while hheap:
            c, u = heapq.heappop(hheap)
            if u in hsettled: continue
            hsettled.add(u)
            for n in range(9):
                if (n,u) in min_cost:
                    nc = c + min_cost[(n,u)]
                    if nc < h[n]: h[n]=nc; heapq.heappush(hheap,(nc,n))

        g   = {n: float('inf') for n in range(9)}
        prev= {n: None for n in range(9)}
        g[0] = departure_time
        heap = [(departure_time + h[0], departure_time, 0)]
        settled = set()

        while heap:
            f, arr, u = heapq.heappop(heap)
            if u in settled: continue
            settled.add(u)
            via = prev[u]
            cost_here = round(arr - (g[prev[u]] if via is not None else departure_time), 2) if via is not None else 0
            steps.append({
                "node": u, "arrival": round(arr, 2),
                "elapsed": round(arr - departure_time, 2),
                "via": via, "cost": cost_here,
                "f_score": round(arr + h[u], 2),
                "h": h[u],
                "status": "settled"
            })
            if u == 8: break
            for (a,b) in EDGES:
                for nu, nv in [(a,b),(b,a)]:
                    if nu==u and nv not in settled:
                        c = get_cost_direct(edges_dict, nu, nv, arr)
                        new_arr = arr + c
                        if c != float('inf') and new_arr < g[nv]:
                            g[nv]=new_arr; prev[nv]=nu
                            heapq.heappush(heap,(new_arr+h[nv], new_arr, nv))

        cur = 8
        while cur is not None: path.append(cur); cur = prev[cur]
        path.reverse()
        total = round(g[8] - departure_time, 2)

    elif algo_name == "TD-G-Tree":
        # G-Tree: mark which cluster each node is in
        clusters = {
            "source":      [0,1,2],
            "destination": [6,7,8],
            "bottom":      [3,4],
            "connector":   [5],
        }
        node_cluster = {}
        for cname, nodes in clusters.items():
            for n in nodes: node_cluster[n] = cname

        # Step 1: reach source cluster borders
        src_allowed = set(clusters["source"])
        dist = {n: float('inf') for n in range(9)}
        prev_map = {n: None for n in range(9)}
        dist[0] = departure_time
        heap = [(departure_time, 0)]
        settled = set()

        while heap:
            arr, u = heapq.heappop(heap)
            if u in settled: continue
            settled.add(u)
            via = prev_map[u]
            cost_here = round(arr - (dist[prev_map[u]] if via is not None else departure_time), 2) if via is not None else 0
            cluster = node_cluster.get(u,"?")
            note = "(border)" if u in BORDER_NODES else f"[{cluster}]"
            steps.append({
                "node": u, "arrival": round(arr,2),
                "elapsed": round(arr-departure_time,2),
                "via": via, "cost": cost_here,
                "note": note, "status": "settled"
            })
            if u == 8: break
            for (a,b) in EDGES:
                for nu, nv in [(a,b),(b,a)]:
                    if nu==u:
                        nc = node_cluster.get(nv,"?")
                        # Skip deep bottom cluster nodes not on border-to-border path
                        if nv == 4 and u not in {3}: continue
                        if nv in settled: continue
                        c = get_cost_direct(edges_dict, nu, nv, arr)
                        if c != float('inf') and arr+c < dist[nv]:
                            dist[nv] = arr+c; prev_map[nv] = nu
                            heapq.heappush(heap,(arr+c,nv))

        # Add skipped nodes
        never_visited = [n for n in range(9) if n not in {s["node"] for s in steps}]
        for n in never_visited:
            steps.append({"node": n, "arrival": None, "elapsed": None,
                           "via": None, "cost": None, "status": "skipped"})

        cur = 8
        while cur is not None: path.append(cur); cur = prev_map[cur]
        path.reverse()
        total = round(dist[8] - departure_time, 2)

    return steps, path, total


# ── Worker thread ─────────────────────────────────────────────────────────

class BenchmarkWorker(QThread):
    all_done = pyqtSignal(dict)
    status   = pyqtSignal(str)

    def __init__(self, selected_algos, departure_time):
        super().__init__()
        self.selected_algos  = selected_algos
        self.departure_time  = departure_time

    def run(self):
        results = {}
        for algo in self.selected_algos:
            self.status.emit(f"Computing {algo.ALGORITHM_NAME}...")
            steps, path, total = compute_trace(algo.ALGORITHM_NAME, self.departure_time)
            # Also time it properly
            g = build_wang_graph()
            n_rep = 50
            t0 = time.perf_counter()
            for _ in range(n_rep):
                algo.run(g, SOURCE, DESTINATION, self.departure_time)
            ms = (time.perf_counter()-t0)/n_rep*1000
            results[algo.ALGORITHM_NAME] = {
                "steps": steps, "path": path,
                "total": total, "ms": round(ms, 4)
            }
        self.all_done.emit(results)
        self.status.emit("Done — watch the animation")


# ── Graph canvas ──────────────────────────────────────────────────────────

class GraphCanvas(FigureCanvas):

    def __init__(self):
        fig = Figure(figsize=(5.5, 4.8), facecolor="white")
        super().__init__(fig)
        self.ax    = fig.add_subplot(111)
        self.graph = build_wang_graph()
        self._settled   = set()
        self._skipped   = set()
        self._path_nodes= []
        self._active    = None
        self._algo_color= "#1565c0"
        self._draw()

    def reset(self, algo_color="#1565c0"):
        self._settled    = set()
        self._skipped    = set()
        self._path_nodes = []
        self._active     = None
        self._algo_color = algo_color
        self._draw()

    def settle_node(self, node, is_path=False):
        self._settled.add(node)
        self._active = node
        if is_path: self._path_nodes.append(node)
        self._draw()

    def skip_node(self, node):
        self._skipped.add(node)
        self._draw()

    def show_final_path(self, path, algo_color):
        self._algo_color  = algo_color
        self._path_nodes  = path
        self._active      = None
        self._draw()

    def _draw(self):
        self.ax.clear()
        self.ax.set_facecolor("white")
        self.figure.patch.set_facecolor("white")
        self.ax.set_xlim(0.2, 6.8)
        self.ax.set_ylim(0.3, 6.1)
        self.ax.axis("off")

        # Draw edges — highlight path edges
        path_edges = set()
        if len(self._path_nodes) > 1:
            for i in range(len(self._path_nodes)-1):
                path_edges.add(tuple(sorted([self._path_nodes[i], self._path_nodes[i+1]])))

        for (u, v) in EDGES:
            key = tuple(sorted([u,v]))
            x0,y0 = NODE_POS[u]; x1,y1 = NODE_POS[v]
            is_vary = key in {tuple(sorted(k)) for k in VARYING_EDGES}
            is_path_e = key in path_edges

            if is_path_e:
                color = self._algo_color; lw = 2.8; ls = "solid"; alpha = 1.0
                # Draw arrow
                self.ax.annotate("",
                    xy=(x1, y1), xytext=(x0, y0),
                    arrowprops=dict(arrowstyle="->", color=color, lw=2.0,
                                    connectionstyle="arc3,rad=0.0"),
                    zorder=4)
            elif is_vary:
                color="#e65100"; lw=1.6; ls=(0,(5,3)); alpha=0.7
                self.ax.plot([x0,x1],[y0,y1],color=color,lw=lw,ls=ls,alpha=alpha,zorder=2)
            else:
                color="#cccccc"; lw=1.2; ls="solid"; alpha=1.0
                self.ax.plot([x0,x1],[y0,y1],color=color,lw=lw,ls=ls,alpha=alpha,zorder=2)

        # PLF labels on time-varying edges
        labeled = set()
        vary_labels = {(1,2):"PLF",(1,6):"PLF★",(3,5):"PLF",(5,6):"PLF"}
        for (u,v), lbl in vary_labels.items():
            key = tuple(sorted([u,v]))
            if key in labeled: continue
            labeled.add(key)
            x0,y0=NODE_POS[u]; x1,y1=NODE_POS[v]
            mx,my=(x0+x1)/2,(y0+y1)/2+0.25
            self.ax.text(mx,my,lbl,fontsize=5.5,color="#bf360c",
                         ha="center",va="center",
                         bbox=dict(facecolor="white",edgecolor="#bf360c",
                                   boxstyle="round,pad=0.1",alpha=0.9),zorder=8)

        # Draw nodes
        for nid,(x,y) in NODE_POS.items():
            if nid == SOURCE:
                base_fc,base_ec = "#1565c0","#0d47a1"
                r = 0.27; tc = "white"
            elif nid == DESTINATION:
                base_fc,base_ec = "#b71c1c","#7f0000"
                r = 0.27; tc = "white"
            elif nid in BORDER_NODES:
                base_fc,base_ec = "#00838f","#004d40"
                r = 0.23; tc = "white"
            else:
                base_fc,base_ec = "#f0f0f0","#999999"
                r = 0.20; tc = "#333333"

            # Override based on animation state
            if nid in self._skipped:
                fc,ec = "#dddddd","#aaaaaa"; tc2 = "#aaaaaa"
            elif nid == self._active:
                fc,ec = "#f9a825","#e65100"; tc2 = "white"
            elif nid in self._settled:
                if nid in self._path_nodes:
                    fc,ec = self._algo_color, self._algo_color; tc2 = "white"
                else:
                    fc,ec = "#78909c","#546e7a"; tc2 = "white"
            else:
                fc,ec = base_fc,base_ec; tc2 = tc

            # Pulse ring for active node
            if nid == self._active:
                ring = plt.Circle((x,y), r+0.06, facecolor="none",
                                   edgecolor="#f9a825", linewidth=2.0,
                                   linestyle="--", zorder=5, alpha=0.7)
                self.ax.add_patch(ring)

            circle = plt.Circle((x,y), r, facecolor=fc,
                                 edgecolor=ec, linewidth=1.6, zorder=6)
            self.ax.add_patch(circle)
            self.ax.text(x,y,f"v{nid}",ha="center",va="center",
                         fontsize=7.5,fontweight="bold",color=tc2,zorder=7)

        # Legend
        legend_items = [
            mpatches.Patch(color="#1565c0", label="Source (v0)"),
            mpatches.Patch(color="#b71c1c", label="Dest (v8)"),
            mpatches.Patch(color="#00838f", label="Border node"),
            mpatches.Patch(color="#f9a825", label="Currently settling"),
            mpatches.Patch(color="#78909c", label="Settled (not on path)"),
            mpatches.Patch(color=self._algo_color, label="Optimal path"),
            mpatches.Patch(color="#dddddd", label="Skipped"),
        ]
        self.ax.legend(handles=legend_items, loc="lower left",
                       fontsize=6.5, framealpha=0.95,
                       edgecolor="#dddddd", facecolor="white", ncol=1)

        self.figure.tight_layout(pad=0.5)
        self.draw()


# ── Step table ────────────────────────────────────────────────────────────

class StepTable(QTableWidget):

    def __init__(self):
        super().__init__()
        self.setColumnCount(5)
        self.setHorizontalHeaderLabels(["Node","Status","Via","Cost","Elapsed"])
        self.horizontalHeader().setSectionResizeMode(QHeaderView.Stretch)
        self.setEditTriggers(QTableWidget.NoEditTriggers)
        self.setAlternatingRowColors(True)
        self.setSelectionMode(QTableWidget.NoSelection)
        self.setMaximumHeight(220)
        self.setStyleSheet("""
            QTableWidget { background:white; color:#222222;
                           gridline-color:#eeeeee; border:1px solid #dddddd;
                           font-size:11px; }
            QTableWidget::item:alternate { background:#f9f9f9; }
            QHeaderView::section { background:#8b1a1a; color:white;
                                   font-weight:bold; padding:5px;
                                   border:none; border-bottom:1px solid #dddddd; }
        """)

    def clear_rows(self):
        self.setRowCount(0)

    def add_row(self, step, algo_color):
        row = self.rowCount()
        self.insertRow(row)
        node = f"v{step['node']}"
        status = step["status"].upper()
        via  = f"v{step['via']}" if step["via"] is not None else "—"
        cost = f"{step['cost']} min" if step["cost"] is not None and step["cost"] != 0 else "—"
        elapsed = f"{step['elapsed']} min" if step["elapsed"] is not None else "—"

        vals = [node, status, via, cost, elapsed]
        colors = {
            "SETTLED": algo_color,
            "SKIPPED": "#999999",
        }
        bg_colors = {
            "SETTLED": "#fff",
            "SKIPPED": "#f5f5f5",
        }

        for col, val in enumerate(vals):
            item = QTableWidgetItem(val)
            item.setTextAlignment(Qt.AlignCenter)
            if col == 0:
                item.setForeground(QColor(colors.get(status, "#333333")))
                item.setFont(QFont("Courier New", 10, QFont.Bold))
            if status == "SKIPPED":
                item.setForeground(QColor("#aaaaaa"))
            self.setItem(row, col, item)

        self.scrollToBottom()


# ── PLF chart ─────────────────────────────────────────────────────────────

class PLFCanvas(FigureCanvas):

    def __init__(self, graph):
        fig = Figure(figsize=(6, 3.5), facecolor="white")
        super().__init__(fig)
        self.fig=fig; self.ax=fig.add_subplot(111); self.graph=graph
        self._draw(None)

    def _draw(self, dep_time):
        self.ax.clear()
        self.ax.set_facecolor("#fafafa")
        self.ax.tick_params(labelsize=8, colors="#333333")
        for sp in self.ax.spines.values(): sp.set_edgecolor("#dddddd")

        t = np.linspace(0,60,300)
        lines = [
            ((1,6),"#c62828","e(v1,v6) ★ critical — spikes at t=30"),
            ((1,2),"#1565c0","e(v1,v2) — rises after t=20"),
            ((3,5),"#2e7d32","e(v3,v5) — drops after t=20"),
            ((5,6),"#e65100","e(v5,v6) — rises after t=25"),
        ]
        for (u,v),color,label in lines:
            costs=[self.graph.get_cost(u,v,ti) for ti in t]
            self.ax.plot(t,costs,color=color,lw=2.0,label=label)

        for dep,label in [(0,"t=0"),(30,"t=30"),(50,"t=50")]:
            self.ax.axvline(x=dep,color="#aaaaaa",lw=1.0,ls="--",alpha=0.7)
            self.ax.text(dep+0.8,19.5,label,fontsize=7,color="#888888")

        if dep_time is not None:
            self.ax.axvline(x=dep_time,color="#8b1a1a",lw=1.8,ls="-",alpha=0.9)

        self.ax.set_xlabel("Departure Time (minutes)",fontsize=8)
        self.ax.set_ylabel("Edge Cost (minutes)",fontsize=8)
        self.ax.set_title("Piecewise Linear Weight Functions",
                           fontsize=9,fontweight="bold")
        self.ax.legend(fontsize=7.5,framealpha=0.95,
                        edgecolor="#dddddd",facecolor="white")
        self.fig.tight_layout()
        self.draw()

    def highlight_departure(self, t):
        self._draw(t)


# ── Main window ───────────────────────────────────────────────────────────

class MainWindow(QMainWindow):

    def __init__(self):
        super().__init__()
        self.setWindowTitle("TDSPP Benchmark  |  9-node TDRN  |  Q(v0 → v8)")
        self.resize(1380, 860)
        self.setStyleSheet("""
            QMainWindow, QWidget { background-color: white; color: #222222; }
            QLabel { color: #222222; }
        """)
        self.graph        = build_wang_graph()
        self.worker       = None
        self._anim_timer  = QTimer()
        self._anim_steps  = []
        self._anim_idx    = 0
        self._anim_color  = "#1565c0"
        self._anim_path   = []
        self._all_results = {}
        self._anim_timer.timeout.connect(self._anim_tick)
        self._build_ui()

    # ── UI construction ───────────────────────────────────────────────────

    def _grp(self, title):
        g = QGroupBox(title)
        g.setStyleSheet("""
            QGroupBox { border:1px solid #dddddd; border-radius:5px;
                        margin-top:8px; font-weight:bold; color:#8b1a1a; padding:6px; }
            QGroupBox::title { subcontrol-origin:margin; left:8px; top:-5px; }
        """)
        return g

    def _build_ui(self):
        central = QWidget()
        self.setCentralWidget(central)
        root = QHBoxLayout(central)
        root.setSpacing(10)
        root.setContentsMargins(10,10,10,10)
        root.addWidget(self._sidebar(), 0)
        root.addWidget(self._main_area(), 1)

    def _sidebar(self):
        side = QWidget()
        side.setFixedWidth(230)
        lay = QVBoxLayout(side)
        lay.setSpacing(8)
        lay.setContentsMargins(0,0,0,0)

        title = QLabel("TDSPP Benchmark")
        title.setStyleSheet("font-size:15px;font-weight:bold;color:#8b1a1a;padding:4px;")
        lay.addWidget(title)

        sub = QLabel("9-node TDRN  ·  Q(v0 → v8)\nWang et al., PVLDB 2019")
        sub.setStyleSheet("font-size:9px;color:#888888;padding:2px 4px;")
        lay.addWidget(sub)

        # Departure time picker
        grp_t = self._grp("Departure Time")
        gt = QVBoxLayout(grp_t)
        self.t_combo = QComboBox()
        self.t_combo.addItems(["t = 0  (off-peak)","t = 30  (peak congestion)","t = 50  (recovering)"])
        self.t_combo.setStyleSheet("""
            QComboBox { border:1px solid #dddddd; border-radius:4px; padding:5px;
                        font-size:11px; background:white; }
            QComboBox::drop-down { border:none; }
        """)
        gt.addWidget(self.t_combo)

        self.t_note = QLabel("e(v1,v6) = 5 min at t=4")
        self.t_note.setStyleSheet("font-size:9px;color:#888888;padding:2px;")
        self.t_note.setWordWrap(True)
        gt.addWidget(self.t_note)
        self.t_combo.currentIndexChanged.connect(self._on_t_changed)
        lay.addWidget(grp_t)

        # Algorithm selection
        grp_a = self._grp("Algorithms")
        ga = QVBoxLayout(grp_a)
        self.checkboxes = {}
        for algo in ALL_ALGORITHMS:
            cb = QCheckBox(algo.ALGORITHM_NAME)
            cb.setChecked(True)
            c = ALGO_COLORS.get(algo.ALGORITHM_NAME,"#333")
            cb.setStyleSheet(f"""
                QCheckBox {{ color:{c};font-weight:bold;font-size:10px; }}
                QCheckBox::indicator {{ width:13px;height:13px;
                    border:1px solid #aaaaaa;border-radius:2px;background:white; }}
                QCheckBox::indicator:checked {{ background:{c};border-color:{c}; }}
            """)
            self.checkboxes[algo.ALGORITHM_NAME] = cb
            ga.addWidget(cb)
        lay.addWidget(grp_a)

        # Animation speed
        grp_sp = self._grp("Animation Speed")
        gsp = QVBoxLayout(grp_sp)
        self.speed_lbl = QLabel("Step delay: 600 ms")
        self.speed_lbl.setStyleSheet("font-size:9px;color:#555555;")
        self.speed_slider = QSlider(Qt.Horizontal)
        self.speed_slider.setRange(100,2000)
        self.speed_slider.setValue(600)
        self.speed_slider.setStyleSheet("""
            QSlider::groove:horizontal{background:#e0e0e0;height:5px;border-radius:3px;}
            QSlider::handle:horizontal{background:#8b1a1a;width:14px;height:14px;
                                        margin:-5px 0;border-radius:7px;}
        """)
        self.speed_slider.valueChanged.connect(
            lambda v: self.speed_lbl.setText(f"Step delay: {v} ms"))
        gsp.addWidget(self.speed_lbl)
        gsp.addWidget(self.speed_slider)
        lay.addWidget(grp_sp)

        # Run button
        self.btn_run = QPushButton("▶  Run Benchmark")
        self.btn_run.setStyleSheet("""
            QPushButton{background:#8b1a1a;color:white;border:none;
                        border-radius:5px;padding:10px;font-weight:bold;font-size:12px;}
            QPushButton:hover{background:#a52020;}
            QPushButton:disabled{background:#cccccc;color:#888888;}
        """)
        self.btn_run.clicked.connect(self._run)
        lay.addWidget(self.btn_run)

        self.btn_stop = QPushButton("⏹  Stop Animation")
        self.btn_stop.setEnabled(False)
        self.btn_stop.setStyleSheet("""
            QPushButton{background:#555555;color:white;border:none;
                        border-radius:5px;padding:7px;font-weight:bold;font-size:11px;}
            QPushButton:hover{background:#333333;}
            QPushButton:disabled{background:#dddddd;color:#aaaaaa;}
        """)
        self.btn_stop.clicked.connect(self._stop_anim)
        lay.addWidget(self.btn_stop)

        self.lbl_status = QLabel("Select departure time and click Run")
        self.lbl_status.setStyleSheet("color:#555555;font-size:10px;padding:3px;")
        self.lbl_status.setWordWrap(True)
        lay.addWidget(self.lbl_status)

        # Complexity reference
        grp_c = self._grp("Complexity")
        gc = QVBoxLayout(grp_c)
        for name,cx,color in [
            ("TD-Dijkstra","O((V+E)logV·f)","#1565c0"),
            ("TD-A*","O((V+E)logV·f)","#2e7d32"),
            ("TD-G-Tree","O(log²(κf)·V)","#bf360c"),
        ]:
            rw=QWidget(); rl=QHBoxLayout(rw); rl.setContentsMargins(0,0,0,0)
            nl=QLabel(name); nl.setStyleSheet(f"color:{color};font-size:8.5px;font-weight:bold;")
            cl=QLabel(cx);   cl.setStyleSheet("color:#777777;font-size:7.5px;")
            rl.addWidget(nl); rl.addStretch(); rl.addWidget(cl)
            gc.addWidget(rw)
        fn=QLabel("f = PLF breakpoints per edge")
        fn.setStyleSheet("color:#aaaaaa;font-size:7px;")
        gc.addWidget(fn)
        lay.addWidget(grp_c)

        lay.addStretch()
        return side

    def _main_area(self):
        tabs = QTabWidget()
        tabs.setStyleSheet("""
            QTabWidget::pane{border:1px solid #dddddd;background:white;}
            QTabBar::tab{background:#f5f5f5;color:#555555;padding:8px 16px;
                         border-radius:4px 4px 0 0;margin-right:2px;
                         border:1px solid #dddddd;}
            QTabBar::tab:selected{background:white;color:#8b1a1a;
                                   font-weight:bold;border-bottom:none;}
        """)

        # ── Tab 1: Graph + Animation ──────────────────────────────────────
        tab1 = QWidget()
        t1   = QVBoxLayout(tab1)
        t1.setContentsMargins(8,8,8,8)

        # Algorithm selector for which algo to animate
        algo_row = QWidget()
        algo_rl  = QHBoxLayout(algo_row)
        algo_rl.setContentsMargins(0,0,0,0)
        algo_rl.addWidget(QLabel("Animate:"))
        self.anim_algo_combo = QComboBox()
        self.anim_algo_combo.addItems([a.ALGORITHM_NAME for a in ALL_ALGORITHMS])
        self.anim_algo_combo.setStyleSheet("""
            QComboBox{border:1px solid #dddddd;border-radius:4px;
                      padding:4px 8px;font-size:11px;background:white;min-width:160px;}
        """)
        self.anim_algo_combo.currentIndexChanged.connect(self._switch_anim_algo)
        algo_rl.addWidget(self.anim_algo_combo)
        algo_rl.addStretch()

        self.lbl_result = QLabel("—")
        self.lbl_result.setStyleSheet(
            "font-size:13px;font-weight:bold;color:#8b1a1a;padding:0 8px;")
        algo_rl.addWidget(self.lbl_result)
        t1.addWidget(algo_row)

        # Graph + step table side by side
        mid = QWidget()
        mid_lay = QHBoxLayout(mid)
        mid_lay.setContentsMargins(0,0,0,0)
        mid_lay.setSpacing(10)

        self.graph_canvas = GraphCanvas()
        mid_lay.addWidget(self.graph_canvas, 3)

        right_panel = QWidget()
        rp_lay = QVBoxLayout(right_panel)
        rp_lay.setContentsMargins(0,0,0,0)

        step_lbl = QLabel("Step-by-step trace:")
        step_lbl.setStyleSheet("font-size:11px;font-weight:bold;color:#333333;padding:2px;")
        rp_lay.addWidget(step_lbl)

        self.step_table = StepTable()
        rp_lay.addWidget(self.step_table)

        # Summary results table
        summary_lbl = QLabel("All algorithms — summary:")
        summary_lbl.setStyleSheet("font-size:11px;font-weight:bold;color:#333333;padding:6px 2px 2px;")
        rp_lay.addWidget(summary_lbl)

        self.summary_table = QTableWidget()
        self.summary_table.setColumnCount(4)
        self.summary_table.setHorizontalHeaderLabels(
            ["Algorithm","Travel Time","Nodes Settled","Query Time (ms)"])
        self.summary_table.horizontalHeader().setSectionResizeMode(QHeaderView.Stretch)
        self.summary_table.setEditTriggers(QTableWidget.NoEditTriggers)
        self.summary_table.setAlternatingRowColors(True)
        self.summary_table.setMaximumHeight(130)
        self.summary_table.setStyleSheet("""
            QTableWidget{background:white;color:#222222;
                         gridline-color:#eeeeee;border:1px solid #dddddd;font-size:11px;}
            QTableWidget::item:alternate{background:#f9f9f9;}
            QHeaderView::section{background:#8b1a1a;color:white;font-weight:bold;
                                  padding:5px;border:none;}
        """)
        rp_lay.addWidget(self.summary_table)
        mid_lay.addWidget(right_panel, 2)
        t1.addWidget(mid, 1)
        tabs.addTab(tab1, "Graph and Animation")

        # ── Tab 2: PLF chart ──────────────────────────────────────────────
        tab2 = QWidget()
        t2   = QVBoxLayout(tab2)
        t2.setContentsMargins(8,8,8,8)
        self.plf_canvas = PLFCanvas(self.graph)
        t2.addWidget(self.plf_canvas)
        tabs.addTab(tab2, "Edge Weight Functions")

        # ── Tab 3: Edge weight table ──────────────────────────────────────
        tab3 = QWidget()
        t3   = QVBoxLayout(tab3)
        t3.setContentsMargins(8,8,8,8)
        t3.addWidget(QLabel("  All edge weights from the experiment table:"))
        edge_tbl = QTableWidget()
        edge_tbl.setColumnCount(3)
        edge_tbl.setHorizontalHeaderLabels(["Edge","PLF Points {(Time, Weight)}","Type"])
        edge_tbl.horizontalHeader().setSectionResizeMode(QHeaderView.Stretch)
        edge_tbl.setEditTriggers(QTableWidget.NoEditTriggers)
        edge_tbl.setAlternatingRowColors(True)
        edge_tbl.setStyleSheet("""
            QTableWidget{background:white;color:#222222;
                         gridline-color:#eeeeee;border:1px solid #dddddd;}
            QTableWidget::item:alternate{background:#f9f9f9;}
            QHeaderView::section{background:#8b1a1a;color:white;font-weight:bold;
                                  padding:5px;border:none;}
        """)
        edge_data = [
            ("e(v1, v2)", "{(0,8), (20,8), (35,20), (60,20)}", "Time-varying"),
            ("e(v0, v2)", "{(0,8), (60,8)}",                   "Fixed"),
            ("e(v0, v1)", "{(0,4), (60,4)}",                   "Fixed"),
            ("e(v1, v6)", "{(0,5), (20,5), (30,18), (60,5)}",  "Time-varying ★"),
            ("e(v5, v6)", "{(0,8), (25,8), (45,12), (60,12)}", "Time-varying"),
            ("e(v2, v3)", "{(0,15), (60,15)}",                 "Fixed"),
            ("e(v3, v4)", "{(0,6), (60,6)}",                   "Fixed"),
            ("e(v3, v5)", "{(0,22), (20,22), (35,6), (60,6)}", "Time-varying"),
            ("e(v2, v5)", "{(0,5), (60,5)}",                   "Fixed"),
            ("e(v6, v7)", "{(0,2), (60,2)}",                   "Fixed"),
            ("e(v7, v8)", "{(0,3), (60,3)}",                   "Fixed"),
        ]
        edge_tbl.setRowCount(len(edge_data))
        for r,(edge,plf,etype) in enumerate(edge_data):
            for c,val in enumerate([edge,plf,etype]):
                item = QTableWidgetItem(val)
                item.setTextAlignment(Qt.AlignCenter)
                if "varying" in etype:
                    item.setForeground(QColor("#bf360c"))
                    item.setFont(QFont("Courier New", 10, QFont.Bold))
                edge_tbl.setItem(r,c,item)
            edge_tbl.setRowHeight(r,32)
        t3.addWidget(edge_tbl)
        tabs.addTab(tab3, "Edge Weight Table")

        return tabs

    # ── Event handlers ────────────────────────────────────────────────────

    def _on_t_changed(self, idx):
        notes = [
            "e(v1,v6) = 5 min at t=4  (off-peak)",
            "e(v1,v6) = 16.27 min at t=34  (PLF spike!)",
            "e(v1,v6) = 7.6 min at t=54  (recovering)",
        ]
        self.t_note.setText(notes[idx])
        self.plf_canvas.highlight_departure([0,30,50][idx])

    def _get_departure(self):
        return [0, 30, 50][self.t_combo.currentIndex()]

    def _run(self):
        selected = [a for a in ALL_ALGORITHMS
                    if self.checkboxes[a.ALGORITHM_NAME].isChecked()]
        if not selected:
            self.lbl_status.setText("Select at least one algorithm.")
            return
        self._stop_anim()
        self.btn_run.setEnabled(False)
        self.lbl_status.setText("Computing traces...")
        t0 = self._get_departure()
        self.worker = BenchmarkWorker(selected, t0)
        self.worker.status.connect(self.lbl_status.setText)
        self.worker.all_done.connect(self._on_results_ready)
        self.worker.start()

    def _on_results_ready(self, all_results):
        self.btn_run.setEnabled(True)
        self._all_results = all_results

        # Update summary table immediately
        self.summary_table.setRowCount(len(all_results))
        for row,(name,data) in enumerate(all_results.items()):
            color = ALGO_COLORS.get(name,"#222222")
            n_settled = len([s for s in data["steps"] if s["status"]=="settled"])
            vals = [name, f"{data['total']:.2f} min",
                    str(n_settled), f"{data['ms']:.4f} ms"]
            for col,val in enumerate(vals):
                item = QTableWidgetItem(val)
                item.setTextAlignment(Qt.AlignCenter)
                if col==0: item.setForeground(QColor(color))
                self.summary_table.setItem(row,col,item)

        # Update combo to show only available algos
        current_algos = list(all_results.keys())
        self.anim_algo_combo.blockSignals(True)
        self.anim_algo_combo.clear()
        self.anim_algo_combo.addItems(current_algos)
        self.anim_algo_combo.blockSignals(False)

        # Start animating first algorithm
        self._start_anim(current_algos[0])

    def _switch_anim_algo(self, idx):
        if not self._all_results: return
        names = list(self._all_results.keys())
        if idx < len(names):
            self._stop_anim()
            self._start_anim(names[idx])

    def _start_anim(self, algo_name):
        if algo_name not in self._all_results: return
        data = self._all_results[algo_name]
        self._anim_steps = data["steps"]
        self._anim_path  = data["path"]
        self._anim_color = ALGO_COLORS.get(algo_name,"#1565c0")
        self._anim_idx   = 0
        self._anim_name  = algo_name

        self.graph_canvas.reset(self._anim_color)
        self.step_table.clear_rows()
        self.lbl_result.setText(f"{algo_name}: animating...")
        self.btn_stop.setEnabled(True)

        delay = self.speed_slider.value()
        self._anim_timer.start(delay)

    def _anim_tick(self):
        if self._anim_idx >= len(self._anim_steps):
            self._anim_timer.stop()
            self.btn_stop.setEnabled(False)
            total = self._all_results[self._anim_name]["total"]
            path_str = " → ".join(f"v{n}" for n in self._anim_path)
            self.lbl_result.setText(
                f"{self._anim_name}: {path_str} = {total} min")
            self.graph_canvas.show_final_path(self._anim_path, self._anim_color)
            self.lbl_status.setText("Animation complete. Select another algorithm to compare.")
            return

        step = self._anim_steps[self._anim_idx]
        self._anim_idx += 1

        # Update timer in case speed changed
        self._anim_timer.setInterval(self.speed_slider.value())

        if step["status"] == "settled":
            is_on_path = step["node"] in self._anim_path
            self.graph_canvas.settle_node(step["node"], is_on_path)
        else:
            self.graph_canvas.skip_node(step["node"])

        self.step_table.add_row(step, self._anim_color)

    def _stop_anim(self):
        self._anim_timer.stop()
        self.btn_stop.setEnabled(False)


def run_gui():
    app = QApplication(sys.argv)
    app.setFont(QFont("Segoe UI", 9))
    win = MainWindow()
    win.show()
    sys.exit(app.exec_())

if __name__ == "__main__":
    run_gui()
