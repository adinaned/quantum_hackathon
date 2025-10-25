from qiskit import QuantumCircuit, QuantumRegister, ClassicalRegister, transpile
from qiskit_aer import AerSimulator
from qiskit.circuit import Parameter, ParameterVector
from qiskit.quantum_info import Statevector, Pauli
import matplotlib.pyplot as plt
import numpy as np
import itertools
from Algorithm import Q

OUTER_POINTS = 12

def measure_using_aer_simulator(qc, shots=500000, method='automatic'):
    simulator = AerSimulator(method=method)
    qc_transpiled = transpile(qc, simulator)
    result = simulator.run(qc_transpiled, shots=shots).result()
    counts = result.get_counts(qc_transpiled)

    # print(qc.draw(output='text'))
    # print(bound_qc.draw(output='text'))

    # fig = bound_qc.draw(output='mpl', fold=100)
    # fig.savefig('qaoa_circuit.png', dpi=200)
    # plt.show()
    return counts

# ----------------------------
# 2) Convertire QUBO -> Ising (h, J, offset)
# ----------------------------

def qubo_to_ising(Q):
    m = Q.shape[0] # number of variables
    h = np.zeros(m) # h vector for ising model
    J = np.zeros((m,m)) # matrix J for ising model
    const = 0.0
    for i in range(m):
        const += Q[i,i] * 1/4
        h[i] += -Q[i,i] * 1/2
    for i in range(m):
        for j in range(i+1, m):
            # off-diagonal terms
            const += 2*Q[i,j] * 1/4
            J[i,j] += 2*Q[i,j] * 1/4
            h[i] += -2*Q[i,j] * 1/2
            h[j] += -2*Q[i,j] * 1/2

    return h, J, const

# ----------------------------
# 3) Build circuit QAOA p-layers
# ----------------------------

def build_qaoa_circuit(h, J, p=1):
    n = len(h) # number of qubits
    qr = QuantumRegister(n, 'q') # quantum register with n qubits
    cr = ClassicalRegister(n, 'c') # classic register with n bits for measurement
    qc = QuantumCircuit(qr, cr) # create whole quantum circuit

    gammas = ParameterVector('gamma', length=p) # hamiltonian cost
    betas = ParameterVector('beta', length=p) # hamiltonian mixer

    for i in range(n):
        qc.h(qr[i]) # initial layer of Hadamard gates for each qubit

    for layer in range(p):
        # repeat for p layers (unitary cost + unitary mixer)
        gamma = gammas[layer]
        beta = betas[layer]

        for i in range(n):
            # single qubit Z-rotation for h terms
            # 2 - from QAOA derivation (e^{−iγZ} -> RZ)
            angle = 2 * gamma * h[i]
            qc.rz(angle, qr[i])

        for i in range(n):
            for j in range(i + 1, n):
                # J_{ij} * Z_{i} * Z_{j} terms implemented with CNOTs and RZ
                if abs(J[i, j]) > 1e-12:
                    angle = 2 * gamma * J[i, j]
                    qc.cx(qr[i], qr[j])
                    qc.rz(angle, qr[j])
                    qc.cx(qr[i], qr[j])

        for i in range(n):
            # mixer Hamiltonian (e^{−iβ∑X_i}
            # implemented with RX rotations - explores the solution space by mixing amplitudes
            qc.rx(2 * beta, qr[i])

    qc.measure(qr, cr) # measurement
    return qc, gammas, betas


def expectation_from_counts(counts, energies):
    shots = sum(counts.values())
    exp_val = 0.0
    for bitstring, count in counts.items():
        idx = int(bitstring, 2) # bitstring to index
        # probability of bitstring * energy
        # energies[idx] - QUOA energy for that bitstring
        exp_val += (count / shots) * energies[idx]
    return exp_val


def show_best_nodes(counts, E_z, top_k=5):
    n = int(np.log2(len(E_z)))
    # Top solutions by (lowest) energy from full state space
    idx_sorted = np.argsort(E_z)  # ascending: best (lowest) first
    print(f"Top {min(top_k, len(idx_sorted))} solutions by energy (lowest first):")
    for rank in range(min(top_k, len(idx_sorted))):
        idx = int(idx_sorted[rank])
        bits = format(idx, f'0{n}b')
        nodes = [i for i, b in enumerate(bits) if b == '1']
        print(f"  #{rank+1}: bitstring={bits} idx={idx} energy={E_z[idx]:.6f} nodes={nodes}")

    # Top measured outcomes from counts
    if counts:
        print(f"\nTop {top_k} measured outcomes (by frequency):")
        measured_sorted = sorted(counts.items(), key=lambda kv: kv[1], reverse=True)
        for rank, (bitstring, freq) in enumerate(measured_sorted[:top_k]):
            idx = int(bitstring, 2)
            nodes = [i for i, b in enumerate(bitstring) if b == '1']
            print(f"  #{rank+1}: bitstring={bitstring} freq={freq} energy={E_z[idx]:.6f} nodes={nodes}")
    else:
        print("No measurement counts available.")


if __name__ == '__main__':
    h, J, const = qubo_to_ising(Q) # convert QUBO to Ising model

    p = 1 # number of QAOA layers
    qc, gammas, betas = build_qaoa_circuit(h, J, p=p)

    # symbolic parameters assignment
    parameter_values = {
        gammas[0]: 0.1,
        betas[0]: 0.1
    }

    bound_qc = qc.assign_parameters(parameter_values)
    counts = measure_using_aer_simulator(bound_qc, shots=1000)

    n = len(h)
    N = 2 ** n # bitstring (possible solution) space size
    E_z = np.zeros(N)
    energies_pairs = {}
    E_z_filtered = np.full(N, np.nan)

    for i, j in itertools.combinations(range(n), 2):
        bitchars = ['0'] * n
        bitchars[i] = '1'
        bitchars[j] = '1'
        bits = np.array(list(map(int, bitchars)))
        idx_int = int(''.join(bitchars), 2)
        energy = bits @ Q @ bits
        energies_pairs[(i, j)] = (energy, ''.join(bitchars))
        E_z_filtered[idx_int] = energy

    for (i, j), (energy, bstr) in sorted(energies_pairs.items(), key=lambda kv: kv[1][0]):
        print(f"pair {(i, j)} bitstring={bstr} energy={energy:.6f}")

    if 'neighbors' in globals():
        print("\nAdjacent pairs:")
        for (i, j), (energy, bstr) in sorted(energies_pairs.items(), key=lambda kv: kv[1][0]):
            if j in neighbors[i]:
                print(f"pair {(i, j)} bitstring={bstr} energy={energy:.6f}")