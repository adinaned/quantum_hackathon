import numpy as np
import pandas as pd
from itertools import product
from qiskit import QuantumCircuit, QuantumRegister, ClassicalRegister, transpile
from qiskit_aer import AerSimulator
from base import resources, df_actions, resource_availability

max_repeat = 2  # max copies per action
action_names = list(df_actions.index)
n_actions = len(action_names)
n_qubits = n_actions * max_repeat
shots = 50000
gamma = 1.0
beta = np.pi / 4
lambda_penalty = 5.0  # penalty for infeasible actions

def measure_using_aer_simulator(qc, shots=50000, method='automatic'):
    simulator = AerSimulator(method=method)
    qc_transpiled = transpile(qc, simulator)
    result = simulator.run(qc_transpiled, shots=shots).result()
    counts = result.get_counts(qc_transpiled)
    return counts

def is_action_feasible(action_name, remaining_resources):
    return all(remaining_resources[r] >= df_actions.loc[action_name, r] for r in resources)

def apply_action(action_name, remaining_resources):
    for r in resources:
        remaining_resources[r] -= df_actions.loc[action_name, r]

qr = QuantumRegister(n_qubits)
cr = ClassicalRegister(n_qubits)
qc = QuantumCircuit(qr, cr)

qc.h(qr)  # initial superposition

# Apply cost unitary with dynamic resource check
for idx in range(n_qubits):
    a_idx = idx // max_repeat
    action_name = action_names[a_idx]
    points = df_actions.loc[action_name, "Points"]

    # simulate remaining resources if all previous qubits are applied
    remaining_resources = resource_availability.copy()
    for prev_idx in range(idx):
        prev_a_idx = prev_idx // max_repeat
        prev_name = action_names[prev_a_idx]
        if prev_idx % max_repeat < 1:  # first copy only
            apply_action(prev_name, remaining_resources)

    # Check feasibility
    penalty = 0 if is_action_feasible(action_name, remaining_resources) else lambda_penalty
    angle = 2 * gamma * (points - penalty)
    qc.rz(angle, qr[idx])

# Mixer unitary
for idx in range(n_qubits):
    qc.rx(2 * beta, qr[idx])

# Measurement
qc.measure(qr, cr)

counts = measure_using_aer_simulator(qc, shots=shots)

def collapse_counts(counts):
    best_bitstring = max(counts, key=counts.get)
    counts_dict = {a: 0 for a in action_names}
    remaining_resources = resource_availability.copy()
    
    # Iterăm bitstring-ul de la MSB la LSB (Qiskit LSB-first)
    for idx, b in enumerate(best_bitstring[::-1]):
        if b == '1':
            a_idx = idx // max_repeat
            a_name = action_names[a_idx]
            
            # Verificăm dacă acțiunea este fezabilă cu resursele curente
            feasible = all(remaining_resources[r] >= df_actions.loc[a_name, r] for r in resources)
            if feasible:
                counts_dict[a_name] += 1
                # Scădem resursele utilizate
                for r in resources:
                    remaining_resources[r] -= df_actions.loc[a_name, r]
    return counts_dict

best_counts = collapse_counts(counts)

def compute_usage_points(counts_dict):
    usage = {r: 0 for r in resources}
    points = 0
    for a, c in counts_dict.items():
        points += df_actions.loc[a, "Points"] * c
        for r in resources:
            usage[r] += df_actions.loc[a, r] * c
    return usage, points

usage, points = compute_usage_points(best_counts)

clean_usage = {r: float(v) for r, v in usage.items()}
print("\nQAOA suggested build counts:", best_counts)
print("Resource usage:")
for r, v in clean_usage.items():
    print(f"  {r}: {v}")
print(f"Total points: {points}")
