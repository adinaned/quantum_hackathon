import random
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import RegularPolygon

def draw_catan_terrain_map():
    # --- Parameters ---
    radius = 1.0  # hex side length
    hex_radius = radius

    # axial coordinates for the 7-hex (2–3–2) layout
    axial_coords = [(0, 0),
                    (1, 0), (1, -1), (0, -1),
                    (-1, 0), (-1, 1), (0, 1)]

    # convert axial to cartesian
    def axial_to_cart(q, r):
        x = hex_radius * (np.sqrt(3) * q + np.sqrt(3)/2 * r)
        y = hex_radius * (1.5 * r)
        return (x, y)

    hex_centers = [axial_to_cart(q, r) for q, r in axial_coords]

    # --- Random terrain + number assignment ---
    terrain_types = {
        "Forest": "#2E8B57",
        "Field": "#F4E04D",
        "Pasture": "#9ACD32",
        "Hill": "#D2691E",
        "Mountain": "#A9A9A9",
        # "Desert": "#EEDD82"
    }

    terrain_list = random.choices(list(terrain_types.keys()), k=len(hex_centers))
    dice_numbers = random.sample([2, 3, 4, 5, 6, 8, 9, 10, 11, 12], len(hex_centers))

    # --- Plot ---
    fig, ax = plt.subplots(figsize=(8, 8))
    ax.set_aspect('equal')
    ax.axis('off')

    for (hx, hy), terrain, number in zip(hex_centers, terrain_list, dice_numbers):
        color = terrain_types[terrain]
        hex_patch = RegularPolygon(
            (hx, hy),
            numVertices=6,
            radius=hex_radius,
            orientation=np.radians(0),
            facecolor=color,
            alpha=1,
            edgecolor='k'
        )
        ax.add_patch(hex_patch)
        # Center text: dice number
        ax.text(hx, hy, str(number), ha='center', va='center',
                fontsize=16, fontweight='bold', color='black')
        # Terrain label
        ax.text(hx, hy - 0.6, terrain, ha='center', va='center',
                fontsize=9, color='black', alpha=0.7)

    ax.scatter(
        [hx for hx, hy in hex_centers],
        [hy for hx, hy in hex_centers],
        c=[terrain_types[t] for t in terrain_list],
        s=40,
        alpha=0
    )
    plt.title("Quantum Catan Challenge — Random Terrain Map", fontsize=14)
    plt.show()

    return terrain_list, dice_numbers

# Run the generator
terrains, numbers = draw_catan_terrain_map()

from matplotlib.patches import RegularPolygon
import numpy as np
import networkx as nx

def draw_connected_catan_board_aligned():
    # --- Parameters ---
    radius = 1.0  # hex side length
    hex_radius = radius
    # axial coordinates for 7-hex layout (center + 6 around)
    axial_coords = [(0, 0),
                    (1, 0), (1, -1), (0, -1),
                    (-1, 0), (-1, 1), (0, 1)]

    # convert axial to cartesian
    def axial_to_cart(q, r):
        x = hex_radius * (np.sqrt(3) * q + np.sqrt(3)/2 * r)
        y = hex_radius * (1.5 * r)
        return (x, y)

    hex_centers = [axial_to_cart(q, r) for q, r in axial_coords]

    # collect all vertices and edges
    vertices = []
    edges = []
    for (hx, hy) in hex_centers:
        for i in range(6):
            # rotate by +30° so that flat edges line up horizontally
            angle1 = np.radians(60 * i + 30)
            angle2 = np.radians(60 * (i + 1) + 30)
            x1, y1 = hx + hex_radius * np.cos(angle1), hy + hex_radius * np.sin(angle1)
            x2, y2 = hx + hex_radius * np.cos(angle2), hy + hex_radius * np.sin(angle2)
            vertices.append((x1, y1))
            vertices.append((x2, y2))
            edges.append(((x1, y1), (x2, y2)))

    # deduplicate nearby vertices
    unique_vertices = []
    tol = 1e-2
    def find_or_add(v):
        for u in unique_vertices:
            if np.linalg.norm(np.array(u) - np.array(v)) < tol:
                return u
        unique_vertices.append(v)
        return v

    merged_edges = []
    for a, b in edges:
        a2, b2 = find_or_add(a), find_or_add(b)
        if a2 != b2:
            merged_edges.append((a2, b2))

    # build graph
    G = nx.Graph()
    for v in unique_vertices:
        G.add_node(v)
    for a, b in merged_edges:
        G.add_edge(a, b)

    # plot
    fig, ax = plt.subplots(figsize=(8,8))
    ax.set_aspect('equal')
    ax.axis('off')

    # draw base hexes (rotated by 30°)
    for (hx, hy) in hex_centers:
        hex_patch = RegularPolygon(
            (hx, hy),
            numVertices=6,
            radius=hex_radius,
            orientation=np.radians(0),
            facecolor='lightgray',
            alpha=0.2,
            edgecolor='k'
        )
        ax.add_patch(hex_patch)

    pos = {v: v for v in G.nodes()}
    nx.draw_networkx_edges(G, pos, ax=ax, width=2, edge_color='brown', alpha=0.8)
    nx.draw_networkx_nodes(G, pos, ax=ax, node_color='orange', node_size=60)

    plt.title("Connected and Aligned 7-Hex Catan Road Network", fontsize=14)
    plt.show()

    print(f"Total vertices (settlement points): {len(unique_vertices)}")
    print(f"Total edges (roads): {len(merged_edges)}")

draw_connected_catan_board_aligned()

import pandas as pd

resources = ["Wood", "Brick", "Wheat", "Ore", "Sheep"]

actions = {
    "Settlement": {"Wood":1,"Brick":1,"Wheat":1,"Sheep":1,"Points":1},
    "Road": {"Wood":1,"Brick":1,"Points":0.5},
    "City": {"Wheat":2,"Ore":3,"Points":2}
}

np.random.seed(42)
resource_availability = {res: np.random.randint(3,10) for res in resources}

df_actions = pd.DataFrame(actions).T.fillna(0)
df_actions = df_actions[resources + ["Points"]]
df_actions["Points"] = df_actions["Points"].astype(float)

print("Available Resources:")
print(resource_availability)


# Quantum Circuit Simulation

import matplotlib.pyplot as plt
from qiskit import QuantumCircuit, transpile
from qiskit.visualization import plot_histogram
from qiskit_aer import AerSimulator


def measure_using_aer_simulator(qc, shots=500000, method='automatic'):
    simulator = AerSimulator(method=method)
    qc_transpiled = transpile(qc, simulator)
    result = simulator.run(qc_transpiled, shots=shots).result()
    counts = result.get_counts(qc_transpiled)
    return counts

backend = AerSimulator()

qc = QuantumCircuit(5)
qc.h(0)
for i in range(4):
    qc.cx(i, i + 1)
qc.measure_all()

# qc.draw(output='mpl')

job = measure_using_aer_simulator(qc, shots=1000)
plt.show()
print(job)
print(qc)

plot_histogram(job)
plt.show()
