import matplotlib.pyplot as plt
from matplotlib.patches import RegularPolygon, Circle
import numpy as np
import random
from typing import List, Set, Optional


# --- Tile class ---
class Tile:
    def __init__(self, tile_id: int, terrain: str, number: int):
        self.id: int = tile_id
        self.terrain: str = terrain
        self.number: int = number
        self.neighbors: List[int] = []  # adjacent tile IDs

    def __repr__(self):
        return f"Tile(Id={self.id}, Terrain={self.terrain}, Dice number={self.number}, Neighbour tiles={self.neighbors})"


# --- Vertex class ---
class Vertex:
    def __init__(self, tiles: List['Tile'], vertex_id: int):
        self.id: int = vertex_id
        self.tiles: List['Tile'] = tiles
        self.occupied: bool = False
        self.neighbors: Set[int] = set()  # neighboring vertex IDs
        self.score: Optional[float] = None  # computed via method

    def calculate_score(self, probability_map):
        """
        Compute vertex score as:
        - Sum of probabilities from dice numbers (probability_map)
        - +1 if at least 2 tiles contain wood (Forest) or brick (Hill)
        - +2 if at least 2 tiles contain grain (Field) or ore (Mountain)
        - +3 if tiles contain ore (Mountain), sheep (Pasture), and grain (Field)
        """
        # Base score from probability
        self.score = sum(probability_map.get(t.number, 0) for t in self.tiles)

        # Count tiles by terrain
        types = [t.terrain for t in self.tiles]

        # +1: wood and brick
        if "Forest" in types and "Hill" in types:
            self.score += 1

        # +2: grain and ore
        if "Field" in types and "Mountain" in types and "Pasture" not in types:
            self.score += 2

        # +3: ore + sheep + grain all present
        if "Mountain" in types and "Pasture" in types and "Field" in types:
            self.score += 3

    def __repr__(self):
        # Represent vertex as the list of adjacent tiles with terrain and number
        tile_strs = [f"{t.terrain}-{t.number}" for t in self.tiles]
        return f"Vertex(Id={self.id}, Adjacent tiles={tile_strs}, Occupied={self.occupied}), Score={self.score}"


# --- Board class ---
class Board:
    def __init__(self):
        self.tiles: List['Tile'] = []
        self.vertices: List['Vertex'] = []

    def add_tile(self, tile):
        self.tiles.append(tile)

    def build_vertices(self):
        """
        Build all vertices for a 7-tile Catan board:
        - 6 center corners (3 tiles each)
        - 6 edges between outer tiles (2 tiles each)
        - 12 outer corners (1 tile each)
        """
        if len(self.tiles) != 7:
            raise ValueError("This method assumes a 7-tile board (1 center + 6 surrounding).")

        vertices = []
        vertex_id = 0

        center = self.tiles[0]
        outer = self.tiles[1:7]

        # --- Step 1: 3-tile vertices (center corners) ---
        center_vertices = []
        for i in range(6):
            t1 = outer[i]
            t2 = outer[(i + 1) % 6]
            v = Vertex([center, t1, t2], vertex_id)
            center_vertices.append(v)
            vertices.append(v)
            vertex_id += 1

        # --- Step 2: 2-tile vertices (edges between neighboring outer tiles) ---
        edge_vertices = []
        for i in range(6):
            t1 = outer[i]
            t2 = outer[(i + 1) % 6]
            v = Vertex([t1, t2], vertex_id)
            edge_vertices.append(v)
            vertices.append(v)
            vertex_id += 1

        # --- Step 3: 1-tile vertices (outer corners) ---
        outer_corner_vertices = []
        for i in range(6):
            t = outer[i]
            for _ in range(2):
                v = Vertex([t], vertex_id)
                outer_corner_vertices.append(v)
                vertices.append(v)
                vertex_id += 1

        # --- Step 4: Set neighbors for center vertices ---
        for i in range(6):
            cv = center_vertices[i]
            # Neighboring center vertices in the ring
            cv.neighbors.add(center_vertices[(i - 1) % 6].id)
            cv.neighbors.add(center_vertices[(i + 1) % 6].id)
            # Neighboring edge vertex
            cv.neighbors.add(edge_vertices[i].id)

        # --- Step 5: Set neighbors for edge vertices ---
        for i in range(6):
            ev = edge_vertices[i]
            # Neighboring center vertex
            ev.neighbors.add(center_vertices[i].id)
            # Neighboring outer corners (exact ones on this edge)
            outer_i_corner = outer_corner_vertices[2 * i + 1]  # second corner of outer[i]
            outer_next_corner = outer_corner_vertices[2 * ((i + 1) % 6)]  # first corner of outer[i+1]
            ev.neighbors.add(outer_i_corner.id)
            ev.neighbors.add(outer_next_corner.id)

        # --- Step 6: Set neighbors for 1-tile outer corners (corrected) ---
        for i in range(6):
            left_edge = edge_vertices[i - 1]  # edge between outer[i-1] and outer[i]
            right_edge = edge_vertices[i]  # edge between outer[i] and outer[i+1]

            v1 = outer_corner_vertices[2 * i]  # first corner of outer[i]
            v2 = outer_corner_vertices[2 * i + 1]  # second corner of outer[i]

            # Connect corners correctly
            v1.neighbors.add(left_edge.id)  # connect to previous edge
            v1.neighbors.add(v2.id)  # connect to the other corner of the same tile

            v2.neighbors.add(v1.id)  # connect to other corner
            v2.neighbors.add(right_edge.id)  # connect to next edge

        self.vertices = vertices

    def compute_edges(self):
        """
        Build a list of undirected edges between vertices.
        Each edge is a tuple: (vertex_id1, vertex_id2)
        Ensures no duplicates and no missing edges.
        """
        edges = set()

        for v in self.vertices:
            for neighbor_id in v.neighbors:
                # Only add edge if neighbor is a Vertex object
                if isinstance(neighbor_id, int):
                    edge = (min(v.id, neighbor_id), max(v.id, neighbor_id))
                    edges.add(edge)

        return sorted(edges)

    def build_vertex_matrix(self, x: float) -> np.ndarray:
        n = len(self.vertices)
        matrix = np.zeros((n, n), dtype=float)

        for v in self.vertices:
            # main diagonal
            matrix[v.id, v.id] = -v.score if v.score is not None else 0
            # neighbors
            for nb_id in v.neighbors:
                matrix[v.id, nb_id] = x

        return matrix

    def __repr__(self):
        tiles_repr = "\n".join([repr(tile) for tile in self.tiles])
        vertices_repr = "\n".join([repr(v) for v in self.vertices])
        return f"Catan Board Tiles: \n{tiles_repr}\n\nVertices: \n{vertices_repr}\n"


# --- Dice probability map - pairs of dice and point value, where 1 point is 1 combination of dice that produces the number---
probability_map = {
    2: 1,
    3: 2,
    4: 3,
    5: 4,
    6: 5,
    7: 6,
    8: 5,
    9: 4,
    10: 3,
    11: 2,
    12: 1
}

def generate_lists():   
    radius = 1.0
    hex_radius = radius
    axial_coords = [(0, 0),
                    (1, 0), (1, -1), (0, -1),
                    (-1, 0), (-1, 1), (0, 1)]

    def axial_to_cart(q, r):
        x = hex_radius * (np.sqrt(3) * q + np.sqrt(3) / 2 * r)
        y = hex_radius * (1.5 * r)
        return x, y
    
    hex_centers = [axial_to_cart(q, r) for q, r in axial_coords]

    terrain_types = {
        "Forest": "#2E8B57",
        "Field": "#F4E04D",
        "Pasture": "#9ACD32",
        "Hill": "#D2691E",
        "Mountain": "#A9A9A9",
    }

    terrain_list = random.choices(list(terrain_types.keys()), k=len(hex_centers))
    dice_numbers = random.sample([2, 3, 4, 5, 6, 8, 9, 10, 11, 12], len(hex_centers))

    return terrain_list, dice_numbers


# --- Original draw function, now builds a Board object ---
def draw_catan_terrain_map(
        terrain_list: list[str],
        dice_numbers: list[int],
        top_pair: tuple[int, int] | None = None
):
    radius = 1.0
    hex_radius = radius
    axial_coords = [(0, 0),
                    (1, 0), (1, -1), (0, -1),
                    (-1, 0), (-1, 1), (0, 1)]

    def axial_to_cart(q, r):
        x = hex_radius * (np.sqrt(3) * q + np.sqrt(3) / 2 * r)
        y = hex_radius * (1.5 * r)
        return x, y

    hex_centers = [axial_to_cart(q, r) for q, r in axial_coords]

    terrain_types = {
        "Forest": "#2E8B57",
        "Field": "#F4E04D",
        "Pasture": "#9ACD32",
        "Hill": "#D2691E",
        "Mountain": "#A9A9A9",
    }


    # --- Build Board ---
    board = Board()
    tiles = []
    for i in range(len(hex_centers)):
        tile = Tile(i, terrain_list[i], dice_numbers[i])
        board.add_tile(tile)
        tiles.append(tile)

    # map axial coordinates to tile index for adjacency
    axial_to_index = {axial_coords[i]: i for i in range(len(axial_coords))}
    directions = [(1, 0), (1, -1), (0, -1), (-1, 0), (-1, 1), (0, 1)]  # axial neighbors
    for idx, (q, r) in enumerate(axial_coords):
        for dq, dr in directions:
            neighbor = (q + dq, r + dr)
            if neighbor in axial_to_index:
                tiles[idx].neighbors.append(axial_to_index[neighbor])

    # Build vertices
    board.build_vertices()
    for v in board.vertices:
        v.calculate_score(probability_map)

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
        ax.text(hx, hy, str(number), ha='center', va='center', fontsize=16, fontweight='bold', color='black')
        ax.text(hx, hy - 0.6, terrain, ha='center', va='center', fontsize=9, color='black', alpha=0.7)

    ax.scatter([hx for hx, hy in hex_centers], [hy for hx, hy in hex_centers],
               c=[terrain_types[t] for t in terrain_list], s=40, alpha=0)
    
    # --- Poziționează vertexurile ---
    vertex_positions = []
    # angle_offset = np.pi / 6  # 30° (colțurile hexagonului)
    angle_offset = - np.pi / 6 
    for hx, hy in hex_centers:
        for k in range(6):
            angle = angle_offset - k * np.pi / 3
            vx = hx + np.cos(angle) * hex_radius
            vy = hy + np.sin(angle) * hex_radius
            vertex_positions.append((vx, vy))
            ax.scatter(vx, vy, color='black', s=10)


    # --- Evidențiere top_pair ---
    if top_pair is not None:
        (i, j) = top_pair
        if i < len(vertex_positions) and j < len(vertex_positions):
            xi, yi = vertex_positions[j] 
            xj, yj = vertex_positions[i]
            ax.scatter(xi, yi, color='red', s=150, zorder=5)
            ax.scatter(xj, yj, color='red', s=150, zorder=5)
         
    plt.title("Quantum Catan Challenge — Random Terrain Map", fontsize=14)
    plt.show()

    return  board


# --- Run ---
terrain_list = ['Mountain', 'Hill', 'Forest', 'Mountain', 'Field', 'Pasture', 'Hill']
dice_numbers = [2, 4, 9, 11, 10, 8, 5]
terrain_list, dice_numbers = generate_lists()
board = draw_catan_terrain_map(terrain_list, dice_numbers)
# print(board)
edges = board.compute_edges()
# print("Edges:", edges)


Q = board.build_vertex_matrix(x=100)
print(Q)
