import numpy as np
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

def compute_penalty(action_name, remaining_resources, lambda_unit=5.0):
    missing = 0
    for r in resources:
        required = df_actions.loc[action_name, r]
        available = remaining_resources[r]
        if available < required:
            missing += required - available
    return lambda_unit * missing


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


def collapse_counts(counts, resource_state):
    best_bitstring = max(counts, key=counts.get)
    counts_dict = {a: 0 for a in action_names}
    remaining_resources = resource_state.copy()

    for idx, b in enumerate(best_bitstring[::-1]):  # Qiskit e LSB-first
        if b == '1':
            a_idx = idx // max_repeat
            a_name = action_names[a_idx]

            feasible = is_action_feasible(a_name, remaining_resources)
            if feasible:
                counts_dict[a_name] += 1
                apply_action(a_name, remaining_resources)
    return counts_dict


def compute_usage_points(counts_dict):
    usage = {r: 0 for r in resources}
    points = 0
    for a, c in counts_dict.items():
        points += df_actions.loc[a, "Points"] * c
        for r in resources:
            usage[r] += df_actions.loc[a, r] * c
    return usage, points


def run_qaoa_once(resource_state):
    qr = QuantumRegister(n_qubits)
    cr = ClassicalRegister(n_qubits)
    qc = QuantumCircuit(qr, cr)

    qc.h(qr)

    for idx in range(n_qubits):
        a_idx = idx // max_repeat
        action_name = action_names[a_idx]
        points = df_actions.loc[action_name, "Points"]

        remaining_resources = resource_state.copy()
        for prev_idx in range(idx):
            prev_a_idx = prev_idx // max_repeat
            prev_name = action_names[prev_a_idx]
            if prev_idx % max_repeat < 1:
                apply_action(prev_name, remaining_resources)

        penalty = compute_penalty(action_name, remaining_resources, lambda_unit=6.0)
        feasibility_bonus = 2.0 if is_action_feasible(action_name, remaining_resources) else 0.0
        redundancy_penalty = 0.5 * (idx % max_repeat)
        angle = 2 * gamma * (points + feasibility_bonus - penalty - redundancy_penalty)
        qc.rz(angle, qr[idx])

    # Mixer
    for idx in range(n_qubits):
        qc.rx(2 * beta, qr[idx])

    qc.measure(qr, cr)
    counts = measure_using_aer_simulator(qc, shots=shots)
    best_counts = collapse_counts(counts, resource_state)
    usage, points = compute_usage_points(best_counts)
    return best_counts, usage, points


def qaoa_turn_loop(resource_availability):
    total_points = 0
    total_usage = {r: 0 for r in resources}
    turn_actions = []

    iteration = 1
    current_resources = resource_availability.copy()

    while True:
        print(f"\n--- QAOA iteration {iteration} ---")
        best_counts, usage, points = run_qaoa_once(current_resources)

        if all(c == 0 for c in best_counts.values()):
            print("No feasible actions left. Turn ends.")
            break

        feasible_move = False
        for a, c in best_counts.items():
            if c > 0 and is_action_feasible(a, current_resources):
                feasible_move = True
                for r in resources:
                    current_resources[r] -= df_actions.loc[a, r] * c
                    total_usage[r] += df_actions.loc[a, r] * c

        if not feasible_move:
            print("No valid move this round — stopping.")
            break

        total_points += points
        turn_actions.append(best_counts)

        print("Chosen actions:", best_counts)
        print("Remaining resources:", current_resources)
        print(f"Turn points so far: {total_points}")
        iteration += 1

        if not any(is_action_feasible(a, current_resources) for a in action_names):
            print("All resources depleted. Turn ends.")
            break

    return turn_actions, total_usage, total_points


turn_actions, total_usage, total_points = qaoa_turn_loop(resource_availability)

print("\n=== FINAL TURN SUMMARY ===")
for i, acts in enumerate(turn_actions, 1):
    print(f"Step {i}: {acts}")

print("\nTotal resource usage:")
for r, v in total_usage.items():
    print(f"  {r}: {v}")
print(f"Total points gained: {total_points}")
