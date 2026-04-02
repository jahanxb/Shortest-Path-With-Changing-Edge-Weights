import math
import json
import networkx as nx
import matplotlib.pyplot as plt
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas

from PyQt5.QtWidgets import (
    QWidget, QVBoxLayout, QHBoxLayout, QPushButton, QLabel, QSlider, QComboBox
)
from PyQt5.QtCore import Qt

# Color Constants (from sp_visualizer theme)
DARK_BG     = "#f4f6fb"
PANEL_BG    = "#ffffff"
TEXT        = "#2e3440"
SUBTEXT     = "#4c566a"
BORDER      = "#e5e9f0"
ACCENT      = "#5e81ac"
ACCENT2     = "#88c0d0"
GREEN       = "#a3be8c"
RED         = "#bf616a"
GOLD        = "#ebcb8b"

class DarkCanvas(FigureCanvas):
    def __init__(self, fig):
        super().__init__(fig)
        self.setStyleSheet(f"background-color: {PANEL_BG}; border: none;")

def make_dark_figure(nrows=1, ncols=1, figsize=(6, 3)):
    import numpy as np
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


class SimulationReplayTab(QWidget):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.simulation_data = None
        self.current_step = 0
        
        layout = QVBoxLayout(self)
        layout.setContentsMargins(4, 4, 4, 4)
        
        # Tool bar 
        tool_layout = QHBoxLayout()
        
        self.combo_algo = QComboBox()
        self.combo_algo.setFixedWidth(200)
        self.combo_algo.currentTextChanged.connect(self._redraw)
        
        self.slider = QSlider(Qt.Horizontal)
        self.slider.setMinimum(0)
        self.slider.setMaximum(0)
        self.slider.setTickPosition(QSlider.TicksBelow)
        self.slider.valueChanged.connect(self._on_slider_change)
        
        self.lbl_step = QLabel("Step 0 / 0")
        self.lbl_step.setFixedWidth(100)
        
        self.btn_load = QPushButton("Load Last Run")
        self.btn_load.clicked.connect(self.load_simulation_file)
        
        tool_layout.addWidget(QLabel("Algorithm:"))
        tool_layout.addWidget(self.combo_algo)
        tool_layout.addWidget(QLabel("Timeline:"))
        tool_layout.addWidget(self.slider)
        tool_layout.addWidget(self.lbl_step)
        tool_layout.addWidget(self.btn_load)
        layout.addLayout(tool_layout)

        # Plot Canvas
        self.fig, self.ax = make_dark_figure(1, 1, figsize=(8, 6))
        self.canvas = DarkCanvas(self.fig)
        layout.addWidget(self.canvas)

    def load_simulation_file(self):
        try:
            with open("simulation_pattern.json", "r") as f:
                self.simulation_data = json.load(f)
            
            # Setup UI based on data
            updates = self.simulation_data.get("updates", [])
            self.slider.setMinimum(0)
            self.slider.setMaximum(len(updates))
            self.slider.setValue(0)
            
            # Collect algos from first update
            self.combo_algo.blockSignals(True)
            self.combo_algo.clear()
            if updates and "algos" in updates[0]:
                self.combo_algo.addItems(list(updates[0]["algos"].keys()))
            self.combo_algo.blockSignals(False)
            
            self._on_slider_change(0)
        except Exception as e:
            self.lbl_step.setText("No run found.")
            print(f"Error loading simulation: {e}")

    def _on_slider_change(self, val):
        if not self.simulation_data: return
        updates = self.simulation_data.get("updates", [])
        self.current_step = val
        self.lbl_step.setText(f"Step {val} / {len(updates)}")
        self._redraw()

    def _redraw(self):
        if not self.simulation_data: return
        
        algo_name = self.combo_algo.currentText()
        edges_init = self.simulation_data.get("initial_weights", [])
        updates = self.simulation_data.get("updates", [])
        step_idx = self.current_step
        
        # Build graph state up to current step
        active_weights = {}
        for u, v, w in edges_init:
            active_weights[(u, v)] = w
            
        highlight_edge = None
        for i in range(step_idx):
            u, v, w_new = updates[i]["u"], updates[i]["v"], updates[i]["w_new"]
            active_weights[(u, v)] = w_new
            if i == step_idx - 1:
                highlight_edge = (u, v)
        
        # Get visited nodes for this exact step and algo
        visited_nodes = []
        if step_idx > 0 and step_idx - 1 < len(updates):
            curr_update = updates[step_idx - 1]
            if "algos" in curr_update and algo_name in curr_update["algos"]:
                visited_nodes = curr_update["algos"][algo_name].get("visited_list", [])
        
        self.ax.clear()
        self.ax.set_facecolor(PANEL_BG)
        for sp in self.ax.spines.values():
            sp.set_edgecolor(BORDER)
            
        n_nodes = self.simulation_data.get("n_nodes", 0)
        if n_nodes > 300:
            self.ax.text(0.5, 0.5, f"Graph is too large to render ({n_nodes} nodes > 300 limit)",
                         ha='center', va='center', transform=self.ax.transAxes, color=TEXT)
            self.canvas.draw()
            return
            
        G = nx.DiGraph()
        G.add_nodes_from(range(n_nodes))
        
        pos_edges = []
        neg_edges = []
        for (u, v), w in active_weights.items():
            G.add_edge(u, v, weight=w)
            if w < 0: neg_edges.append((u, v))
            else: pos_edges.append((u, v))
            
        # Layout
        pos = nx.spring_layout(G, k=1/math.sqrt(n_nodes + 1), seed=42)
        
        # Nodes: Standard vs Visited
        standard_nodes = [n for n in G.nodes() if n not in visited_nodes]
        
        nx.draw_networkx_nodes(G, pos, ax=self.ax, nodelist=standard_nodes, node_color=ACCENT2, node_size=100, alpha=0.8, edgecolors=PANEL_BG)
        nx.draw_networkx_nodes(G, pos, ax=self.ax, nodelist=visited_nodes, node_color=GOLD, node_size=150, alpha=1.0, edgecolors=TEXT)
        
        nx.draw_networkx_labels(G, pos, ax=self.ax, font_size=8, font_color="white")
        edge_labels = {(u, v): f"{w:.1f}" for (u, v), w in active_weights.items()}
        nx.draw_networkx_edge_labels(G, pos, ax=self.ax, edge_labels=edge_labels, font_size=7, label_pos=0.3, font_color=ACCENT)
            
        nx.draw_networkx_edges(G, pos, ax=self.ax, edgelist=pos_edges, width=1.0, alpha=0.5, edge_color=SUBTEXT, arrows=True, arrowsize=8, connectionstyle='arc3,rad=0.1')
        nx.draw_networkx_edges(G, pos, ax=self.ax, edgelist=neg_edges, width=1.5, alpha=0.8, edge_color=RED, arrows=True, arrowsize=10, connectionstyle='arc3,rad=0.1')
        
        if highlight_edge and highlight_edge in G.edges:
            nx.draw_networkx_edges(G, pos, ax=self.ax, edgelist=[highlight_edge], width=2.5, alpha=1.0, edge_color=GREEN, arrows=True, arrowsize=12, connectionstyle='arc3,rad=0.1')
        
        self.ax.set_xticks([])
        self.ax.set_yticks([])
        
        title_str = "Status: Initial State" if step_idx == 0 else f"Highlighting Edge Update & Nodes Visited by {algo_name}"
        self.ax.set_title(title_str, color=TEXT, fontsize=10, fontweight='bold')
        self.fig.tight_layout(pad=1.0)
        self.canvas.draw()
