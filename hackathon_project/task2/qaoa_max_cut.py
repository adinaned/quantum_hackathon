import matplotlib.pyplot as plt
from matplotlib.patches import RegularPolygon
import networkx as nx
import numpy as np
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


num_edges = len(free_edges)
Q = np.zeros((num_edges, num_edges))

for i in range(num_edges):
    for j in range(i+1, num_edges):
        if len(set(free_edges[i]) & set(free_edges[j])) == 0:
            Q[i,j] = 1
            Q[j,i] = 1


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
    return qc,gammas,betas

qc,gammas,betas = build_qaoa_circuit(h,J,p=1)


def measure_using_aer_simulator(qc, shots=5000):
    sim = AerSimulator()
    qc_t = transpile(qc, sim)
    result = sim.run(qc_t, shots=shots).result()
    counts = result.get_counts(qc_t)
    return counts

bound_qc = qc.assign_parameters({gammas[0]:0.1, betas[0]:0.1})
counts = measure_using_aer_simulator(bound_qc, shots=5000)


def longest_path_in_subgraph(edges):
    G = nx.Graph()
    G.add_edges_from(edges)
    max_len = 0
    best_path = []
    for start in G.nodes():
        path = [start]
        visited = set(path)
        def dfs(u):
            nonlocal max_len, best_path
            extended = False
            for v in G.neighbors(u):
                if v not in visited:
                    visited.add(v)
                    path.append(v)
                    dfs(v)
                    path.pop()
                    visited.remove(v)
                    extended = True
            if not extended and len(path)-1 > max_len:
                max_len = len(path)-1
                best_path = path.copy()
        dfs(start)
    return best_path


def find_best_path_from_counts(counts, free_edges):
    measured_sorted = sorted(counts.items(), key=lambda kv: kv[1], reverse=True)
    for bitstring, freq in measured_sorted:
        edges_included = [free_edges[i] for i,b in enumerate(bitstring) if b=='1']
        if edges_included:
            path_nodes = longest_path_in_subgraph(edges_included)
            path_edges = [(path_nodes[i], path_nodes[i+1]) for i in range(len(path_nodes)-1)]
            return path_edges
    return []

best_path_edges = find_best_path_from_counts(counts, free_edges)


def draw_catan_with_qaoa_path(path_edges):
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

    if path_edges:
        nx.draw_networkx_edges(G_total,pos,edgelist=path_edges,width=4,edge_color='blue',alpha=0.9)

    nx.draw_networkx_nodes(G_total,pos,node_color='orange',node_size=120)
    plt.title("Catan Board: Red=Occupied, Gray=Free, Blue=Longest Road (Max-Cut)",fontsize=10)
    plt.show()

draw_catan_with_qaoa_path(best_path_edges)
