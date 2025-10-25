
"""## 🛣️ Task 2 -- Quantum Longest Road

### Task Description

The *Longest Road* in Catan is a classic path-finding and connectivity problem.
Here, you’ll use the connected road network graph of the 7-hex board.

**Your goal:**
- Identify the **longest continuous path** that could be built on the current network.
- Formulate the problem as a **graph traversal or combinatorial optimization** suitable for a quantum algorithm.

**Possible approaches:**
- Encode the pathfinding problem as a **QUBO** or **Ising Hamiltonian**, using penalty terms to ensure continuity.
- Use QAOA or a similar method to explore valid road configurations.
- Alternatively, use quantum random walks or amplitude amplification for path search.

💡 *Extension idea:* Consider randomizing blocked roads (representing occupied edges) and finding the new optimal route under constraints.

### Basic Map Generation
"""

import matplotlib.pyplot as plt
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