# --- Cod combinat: harta + QAOA cu printuri și drumuri cele mai lungi ---
import matplotlib.pyplot as plt
from matplotlib.patches import RegularPolygon
import networkx as nx
import numpy as np
import itertools
import random
from qiskit import QuantumCircuit, QuantumRegister, ClassicalRegister, transpile
from qiskit_aer import AerSimulator
from qiskit.circuit import ParameterVector

# 1) Creăm harta și muchiile
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

    # ocupam random 15 muchii
    occupied_edges = random.sample(merged_edges, 15)
    free_edges = [e for e in merged_edges if e not in occupied_edges]

    print("=== Board Info ===")
    print(f"Total free edges: {len(free_edges)}")
    print(f"Total occupied edges: {len(occupied_edges)}")
    print("Free edges (first 5):", free_edges[:5])
    print("Occupied edges (first 5):", occupied_edges[:5])

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
    nx.draw_networkx_nodes(G, pos, node_color='orange', node_size=60)

    plt.title("Catan 7-Hex Board: Red=Occupied, Gray=Free", fontsize=14)
    plt.show()

draw_connected_catan_board_no_overlap()

# 2) Construim QUBO pentru drumuri libere
num_roads = len(free_edges)

def build_qubo_longest_road(num_roads, lam=10):
    Q = np.zeros((num_roads, num_roads))
    for i in range(num_roads):
        Q[i,i] = -1
    for i,j in itertools.combinations(range(num_roads),2):
        Q[i,j] = 2
        Q[j,i] = Q[i,j]
    print("\n=== QUBO Matrix ===")
    print(Q)
    return Q

Q = build_qubo_longest_road(num_roads)

# 3) QUBO -> Ising
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
    print("\n=== Ising h vector ===")
    print(h)
    print("=== Ising J matrix ===")
    print(J)
    return h,J,const

h,J,const = qubo_to_ising(Q)

# 4) Circuit QAOA
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

# 5) Simulare
def measure_using_aer_simulator(qc,shots=5000):
    sim = AerSimulator()
    qc_t = transpile(qc,sim)
    result = sim.run(qc_t,shots=shots).result()
    counts = result.get_counts(qc_t)
    print("\n=== QAOA Counts (top 10) ===")
    for bitstring, freq in list(counts.items())[:10]:
        print(f"{bitstring}: {freq}")
    return counts

bound_qc = qc.assign_parameters({gammas[0]:0.1, betas[0]:0.1})
counts = measure_using_aer_simulator(bound_qc,shots=5000)

# 6) Obținem energie și top soluții
def expectation_from_counts(counts,Q):
    n = len(Q)
    N = 2**n
    energies = np.zeros(N)
    for k in range(N):
        bits = format(k,'0'+str(n)+'b')
        bits_vec = np.array([int(b) for b in bits])
        energies[k] = bits_vec @ Q @ bits_vec
    exp_val = 0
    total = sum(counts.values())
    for bitstring,freq in counts.items():
        idx = int(bitstring,2)
        exp_val += (freq/total)*energies[idx]
    print("\n=== Energies (first 10 states) ===")
    for k in range(min(10,len(energies))):
        print(f"State {format(k,'0'+str(n)+'b')}: energy={energies[k]}")
    return exp_val,energies

exp_val, energies = expectation_from_counts(counts,Q)
print("Exp. value:",exp_val)

def show_top_solutions(counts,energies,top_k=5):
    n = int(np.log2(len(energies)))
    idx_sorted = np.argsort(energies)
    print(f"\nTop {top_k} configurations by energy:")
    for rank in range(min(top_k,len(idx_sorted))):
        idx = idx_sorted[rank]
        bits = format(idx,'0'+str(n)+'b')
        nodes = [i for i,b in enumerate(bits) if b=='1']
        print(f"#{rank+1}: bitstring={bits}, energy={energies[idx]}, roads={nodes}")

    measured_sorted = sorted(counts.items(), key=lambda kv: kv[1], reverse=True)
    print(f"\nTop measured outcomes:")
    for rank,(bitstring,freq) in enumerate(measured_sorted[:top_k]):
        nodes = [i for i,b in enumerate(bitstring) if b=='1']
        idx = int(bitstring,2)
        print(f"#{rank+1}: bitstring={bitstring}, freq={freq}, energy={energies[idx]}, roads={nodes}")

show_top_solutions(counts,energies)

# 7) Găsim drumurile cele mai lungi
def find_longest_paths(free_edges):
    G_free = nx.Graph()
    G_free.add_edges_from(free_edges)
    all_paths = []

    for u in G_free.nodes():
        for v in G_free.nodes():
            if u != v:
                for path in nx.all_simple_paths(G_free, source=u, target=v):
                    edges_in_path = [(path[i], path[i+1]) for i in range(len(path)-1)]
                    all_paths.append(edges_in_path)

    # validează: noduri nu mai mult de 2 conexiuni
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

# 8) Vizualizare drum QAOA pe hartă (numai cel mai lung)
def draw_catan_with_qaoa_path(longest_paths):
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

    # desenăm numai drumurile cele mai lungi
    for path_edges in longest_paths:
        nx.draw_networkx_edges(G_total,pos,edgelist=path_edges,width=4,edge_color='blue',alpha=0.9)

    nx.draw_networkx_nodes(G_total,pos,node_color='orange',node_size=60)
    plt.title("Catan 7-Hex Board: Red=Occupied, Gray=Free, Blue=Longest Road(s)",fontsize=14)
    plt.show()

longest_paths = find_longest_paths(free_edges)
print("\n=== Longest Path(s) Info ===")
for i, path in enumerate(longest_paths):
    print(f"Path {i+1}: {path}")

draw_catan_with_qaoa_path(longest_paths)
