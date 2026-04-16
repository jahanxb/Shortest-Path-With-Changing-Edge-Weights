from algorithms import td_dijkstra
from algorithms import td_astar
from algorithms import td_g_tree
from algorithms import quantum_annealing

ALL_ALGORITHMS = [
    td_dijkstra,
    td_astar,
    td_g_tree,
]

# Quantum kept separate — shown in its own tab in GUI
QUANTUM_ALGORITHM = quantum_annealing
