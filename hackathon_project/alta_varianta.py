# Full Catan settlement optimization with QAOA
# Requires: qiskit, qiskit-aer, networkx
# Install with: pip install qiskit qiskit-aer networkx

import networkx as nx
from qiskit.quantum_info import SparsePauliOp, Statevector
from qiskit_algorithms import QAOA
from qiskit_algorithms.optimizers import COBYLA
from qiskit_aer.primitives import Estimator as AerEstimator
#from qiskit_aer.primitives import Sampler as AerSampler
from qiskit_aer import AerSimulator

# ----------------------
# Step 1: Define Catan Map
# ----------------------
graph = nx.Graph()
graph.add_nodes_from(range(7))  # 7 possible settlement spots
# Edges = conflicts (too close to place settlements)
graph.add_edges_from([
    (0, 1), (0, 2),
    (1, 2), (1, 3),
    (2, 3), (2, 4),
    (3, 4), (3, 5),
    (4, 5), (4, 6),
    (5, 6)
])

# Node weights = expected resource production
node_weights = {0: 8, 1: 6, 2: 5, 3: 9, 4: 4, 5: 3, 6: 10}

# ----------------------
# Step 2: Build Pauli Hamiltonian
# ----------------------
def build_catan_paulis(graph, node_weights):
    n = len(graph)
    pauli_list = []

    # Reward terms for each spot
    for node, weight in node_weights.items():
        paulis = ["I"] * n
        paulis[node] = "Z"
        pauli_list.append(("".join(paulis)[::-1], -weight))  # negative for minimization

    # Penalty terms for conflicts
    max_weight = max(node_weights.values())
    for edge in graph.edges():
        paulis = ["I"] * n
        paulis[edge[0]], paulis[edge[1]] = "Z", "Z"
        pauli_list.append(("".join(paulis)[::-1], 2 * max_weight))  # large penalty

    return pauli_list

catan_paulis = build_catan_paulis(graph, node_weights)
cost_hamiltonian = SparsePauliOp.from_list(catan_paulis)
pauli_sum_op = SparsePauliOp.from_list(catan_paulis)

# ----------------------
# Step 3: Set up QAOA
# ----------------------
'''quantum_instance = Sampler(AerSimulator())
optimizer = COBYLA(maxiter=100)
qaoa = QAOA(optimizer=optimizer, reps=2, quantum_instance=quantum_instance)'''

estimator = AerEstimator() 
#sampler = AerSampler()
optimizer = COBYLA(maxiter=100)
qaoa = QAOA(estimator, optimizer=optimizer, reps=2)
# Step 4: Run QAOA  
# ----------------------
result = qaoa.compute_minimum_eigenvalue(operator=pauli_sum_op)
print("Optimal Parameters:", result.optimal_point)
print("Minimum Cost:", result.eigenvalue.real)

# ----------------------
# Step 5: Extract best settlement placement
# ----------------------
state = Statevector(result.eigenstate)
probabilities = state.probabilities_dict()
sorted_probs = sorted(probabilities.items(), key=lambda x: x[1], reverse=True)
best_bitstring = sorted_probs[0][0]

print("\nBest settlement placement (bitstring):", best_bitstring)
for idx, bit in enumerate(best_bitstring[::-1]):  # reverse to match node order
    if bit == '1':
        print(f"Place settlement on spot {idx} (weight {node_weights[idx]})")
