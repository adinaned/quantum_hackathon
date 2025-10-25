import matplotlib.pyplot as plt
from matplotlib.patches import RegularPolygon
import numpy as np
import networkx as nx
import random

free_edges = []
occupied_edges = []

# --- Board and graph setup ---
def draw_connected_catan_board_no_overlap():
    radius = 1.0
    hex_radius = radius
    axial_coords = [(0, 0),
                    (1, 0), (1, -1), (0, -1),
                    (-1, 0), (-1, 1), (0, 1)]

    def axial_to_cart(q, r):
        x = hex_radius * (np.sqrt(3) * q + np.sqrt(3)/2 * r)
        y = hex_radius * (1.5 * r)
        return (x, y)

    hex_centers = [axial_to_cart(q, r) for q, r in axial_coords]

    vertices = []
    edges = []
    for (hx, hy) in hex_centers:
        for i in range(6):
            angle1 = np.radians(60 * i + 30)
            angle2 = np.radians(60 * (i + 1) + 30)
            x1, y1 = hx + hex_radius * np.cos(angle1), hy + hex_radius * np.sin(angle1)
            x2, y2 = hx + hex_radius * np.cos(angle2), hy + hex_radius * np.sin(angle2)
            vertices.append((x1, y1))
            vertices.append((x2, y2))
            edges.append(((x1, y1), (x2, y2)))

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
        # Ensure each edge is added only once
        if a2 != b2 and (a2, b2) not in merged_edges and (b2, a2) not in merged_edges:
            merged_edges.append((a2, b2))

    G = nx.Graph()
    for v in unique_vertices:
        G.add_node(v)
    for a, b in merged_edges:
        G.add_edge(a, b)

    # --- Randomly select 18 unique occupied edges ---
    occupied_edges = random.sample(merged_edges, 15)
    free_edges = [e for e in merged_edges if e not in occupied_edges]

    # --- Plotting ---
    fig, ax = plt.subplots(figsize=(8,8))
    ax.set_aspect('equal')
    ax.axis('off')

    for (hx, hy) in hex_centers:
        hex_patch = RegularPolygon(
            (hx, hy), numVertices=6, radius=hex_radius,
            orientation=np.radians(0), facecolor='lightgray', alpha=0.2, edgecolor='k')
        ax.add_patch(hex_patch)

    pos = {v: v for v in G.nodes()}

    # Draw free edges
    nx.draw_networkx_edges(G, pos, edgelist=free_edges, ax=ax, width=3, edge_color='lightgray', alpha=0.6)
    # Draw occupied edges
    nx.draw_networkx_edges(G, pos, edgelist=occupied_edges, ax=ax, width=3, edge_color='red', alpha=0.9)
    # Draw nodes
    nx.draw_networkx_nodes(G, pos, ax=ax, node_color='orange', node_size=60)

    plt.title("Catan 7-Hex Board: Red = Occupied Roads, Gray = Free Roads", fontsize=14)
    plt.show()

    print(f"Total vertices: {len(unique_vertices)}")
    print(f"Total edges: {len(merged_edges)} (Occupied: 18, Free: {len(free_edges)})")

# Run the drawing function
draw_connected_catan_board_no_overlap()
