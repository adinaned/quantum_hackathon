import matplotlib.pyplot as plt
from matplotlib.patches import RegularPolygon
import networkx as nx
import numpy as np
import itertools
import random
from qiskit import QuantumCircuit, QuantumRegister, ClassicalRegister, transpile
from qiskit_aer import AerSimulator
from qiskit.circuit import ParameterVector


free_edges = []
occupied_edges = []

def draw_connected_catan_board_no_overlap():
    global free_edges, occupied_edges
    radius = 1.0
    hex_radius = radius
    axial_coords = [(0, 0), (1, 0), (1, -1), (0, -1), (-1, 0), (-1, 1), (0, 1)]

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
        if a2 != b2 and (a2, b2) not in merged_edges and (b2, a2) not in merged_edges:
            merged_edges.append((a2, b2))

    G = nx.Graph()
    for v in unique_vertices:
        G.add_node(v)
    for a, b in merged_edges:
        G.add_edge(a, b)

    occupied_edges = random.sample(merged_edges, 16)
    free_edges = [e for e in merged_edges if e not in occupied_edges]
    pos = {v: v for v in G.nodes()}
    fig, ax = plt.subplots(figsize=(8,8))
    ax.set_aspect('equal')
    ax.axis('off')

    for (hx, hy) in hex_centers:
        ax.add_patch(RegularPolygon(xy=(hx, hy), numVertices=6, radius=hex_radius,
                                    orientation=np.radians(0), facecolor='lightgray',
                                    alpha=0.2, edgecolor='k'))

    nx.draw_networkx_edges(G, pos, edgelist=free_edges, width=3, edge_color='lightgray', alpha=0.6)
    nx.draw_networkx_edges(G, pos, edgelist=occupied_edges, width=3, edge_color='red', alpha=0.9)
    nx.draw_networkx_nodes(G, pos, node_color='orange', node_size=120)

    plt.title("Catan 7-Hex Board: Red=Occupied, Gray=Free", fontsize=14)
    plt.show()

draw_connected_catan_board_no_overlap()
num_roads = len(free_edges)

def build_qubo_longest_road(num_roads):
    Q = np.zeros((num_roads, num_roads))
    for i in range(num_roads):
        Q[i,i] = -1 # reward for including road i
    for i,j in itertools.combinations(range(num_roads),2):
        Q[i,j] = 2 # penalty for including both roads i and j
        Q[j,i] = Q[i,j] # symmetric
    return Q

Q = build_qubo_longest_road(num_roads)


def qubo_to_ising(Q):
    n = len(Q)
    h = np.zeros(n)
    J = np.zeros((n,n))
    const = 0
    for i in range(n):
        const += Q[i,i]*0.25
        h[i] += -0.5*Q[i,i]
    for i in range(n):
        for j in range(i+1,n):
            const += 0.5*Q[i,j]
            J[i,j] += 0.5*Q[i,j]
            h[i] += -0.5*Q[i,j]
            h[j] += -0.5*Q[i,j]
    return h,J,const

h,J,const = qubo_to_ising(Q)

def build_qaoa_circuit(h,J,p=1):
    n = len(h)
    qr = QuantumRegister(n,'q')
    cr = ClassicalRegister(n,'c')
    qc = QuantumCircuit(qr,cr)
    gammas = ParameterVector('gamma',p)
    betas = ParameterVector('beta',p)

    for i in range(n):
        qc.h(qr[i])

    for layer in range(p):
        gamma = gammas[layer]
        beta = betas[layer]
        for i in range(n):
            qc.rz(2*gamma*h[i],qr[i])
        for i in range(n):
            for j in range(i+1,n):
                if abs(J[i,j])>1e-12:
                    qc.cx(qr[i],qr[j])
                    qc.rz(2*gamma*J[i,j],qr[j])
                    qc.cx(qr[i],qr[j])
        for i in range(n):
            qc.rx(2*beta,qr[i])

    qc.measure(qr,cr)
    print(qc)
    return qc,gammas,betas

qc,gammas,betas = build_qaoa_circuit(h,J,p=1)


def measure_using_aer_simulator(qc,shots=5000):
    sim = AerSimulator()
    qc_t = transpile(qc,sim)
    result = sim.run(qc_t,shots=shots).result()
    counts = result.get_counts(qc_t)
    return counts

bound_qc = qc.assign_parameters({gammas[0]:0.1, betas[0]:0.1})
counts = measure_using_aer_simulator(bound_qc,shots=5000)


def find_longest_paths(free_edges, max_length=6):
    G_free = nx.Graph()
    G_free.add_edges_from(free_edges)
    all_paths = []

    for u in G_free.nodes():
        for v in G_free.nodes():
            if u != v:
                for path in nx.all_simple_paths(G_free, source=u, target=v, cutoff=max_length):
                    edges_in_path = [(path[i], path[i+1]) for i in range(len(path)-1)]
                    if len(edges_in_path) <= max_length:
                        all_paths.append(edges_in_path)
    valid_paths = []
    for path in all_paths:
        subG = nx.Graph()
        subG.add_edges_from(path)
        if max(dict(subG.degree()).values()) <= 2:
            valid_paths.append(path)

    if not valid_paths:
        return []

    max_len = max(len(p) for p in valid_paths)
    longest_paths = [p for p in valid_paths if len(p) == max_len]
    return longest_paths


def draw_catan_with_qaoa_path_numbered_ordered(longest_paths):
    radius = 1.0
    hex_radius = radius
    axial_coords = [(0,0),(1,0),(1,-1),(0,-1),(-1,0),(-1,1),(0,1)]
    def axial_to_cart(q,r):
        x = hex_radius*(np.sqrt(3)*q + np.sqrt(3)/2*r)
        y = hex_radius*(1.5*r)
        return (x,y)
    hex_centers = [axial_to_cart(q,r) for q,r in axial_coords]

    G_total = nx.Graph()
    for e in free_edges + occupied_edges:
        G_total.add_edge(*e)

    pos = {v:v for e in free_edges+occupied_edges for v in e}

    fig,ax = plt.subplots(figsize=(8,8))
    ax.set_aspect('equal')
    ax.axis('off')

    for hx,hy in hex_centers:
        ax.add_patch(RegularPolygon(xy=(hx, hy), numVertices=6, radius=hex_radius,
                                    orientation=np.radians(0), facecolor='lightgray',
                                    alpha=0.2, edgecolor='k'))

    nx.draw_networkx_edges(G_total,pos,edgelist=free_edges,width=3,edge_color='lightgray',alpha=0.6)
    nx.draw_networkx_edges(G_total,pos,edgelist=occupied_edges,width=3,edge_color='red',alpha=0.9)

    for path_edges in longest_paths:
        nx.draw_networkx_edges(G_total,pos,edgelist=path_edges,width=4,edge_color='blue',alpha=0.9)

    nx.draw_networkx_nodes(G_total,pos,node_color='orange',node_size=120)
    for i,v in enumerate(G_total.nodes()):
        ax.text(v[0], v[1], str(i), color='black', fontsize=8, ha='center', va='center')

    unique_paths = set()  # folosit pentru a elimina duplicatele

    for path_edges in longest_paths:
        ordered_nodes = [path_edges[0][0], path_edges[0][1]]
        for a, b in path_edges[1:]:
            if b != ordered_nodes[-1]:
                ordered_nodes.append(b)
            elif a != ordered_nodes[-1]:
                ordered_nodes.append(a)

        path_tuple = tuple(ordered_nodes)
        path_tuple_rev = tuple(reversed(ordered_nodes))

        if path_tuple not in unique_paths and path_tuple_rev not in unique_paths:
            unique_paths.add(path_tuple)
            node_indices = [list(G_total.nodes()).index(v) for v in ordered_nodes]
            print(f"Valid path (length {len(path_edges)}): nodes in order={node_indices}")

    plt.title("Catan 7-Hex Board: Red=Occupied, Gray=Free, Blue=Longest Road(s)",fontsize=10)
    plt.show()


longest_paths = find_longest_paths(free_edges)
draw_catan_with_qaoa_path_numbered_ordered(longest_paths)
