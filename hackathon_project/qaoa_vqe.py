# QAOA implementation (Qiskit) pentru "Quantum Catan" (mini hartă cu 7 hexagoane)
#
# - Construiește QUBO pentru problema: alege exact 2 intersecții care maximizează probabilitatea
#   totală de a primi resurse (pornind de la numerotarea hexagoanelor și adiacențe).
# - Convertește QUBO -> Hamiltonian Ising H = sum h_i Z_i + sum_{i<j} J_ij Z_i Z_j + const
# - Construiește circuit QAOA parametrizat pentru p nivele: U_C(gamma) = exp(-i gamma H_C)
#   și U_M(beta) = exp(-i beta sum X_i) (mixer = Rx(2*beta) pe fiecare qubit)
# - Exemplu de rulare pe simulator (StatevectorSimulator / Aer) și calculare a valorii așteptate

# Necesită: qiskit (pip install qiskit)

from qiskit import QuantumCircuit, QuantumRegister, ClassicalRegister, transpile
from qiskit_aer import AerSimulator
from qiskit.circuit import Parameter, ParameterVector
from qiskit.quantum_info import Statevector, Pauli
import numpy as np
import itertools

def measure_using_aer_simulator(qc, shots=500000, method='automatic'):
    """
    Measure the quantum circuit using the AerSimulator
    Needed to reconstruct the image
    Args:
        qc: quantum circuit to measure
        shots: number of shots for the simulation
        method: method for the AerSimulator (default is 'automatic')
    Returns:
        counts: counts obtained from the quantum circuit
    """
    simulator = AerSimulator(method=method)
    qc_transpiled = transpile(qc, simulator)
    result = simulator.run(qc_transpiled, shots=shots).result()
    counts = result.get_counts(qc_transpiled)
    return counts


# ----------------------------
# 1) Construire QUBO
# ----------------------------

def build_qubo(hex_numbers, adjacency, lam=None):
    """
    hex_numbers: list of 7 numbers on hexes (values 2..12)
    adjacency: list of length m (num intersecții), each element e un list cu indexurile hex-urilor pe care le atinge
    lam: penalizare pentru constrângerea sum x = 2. Daca None, se alege heuristica 5 * max(w)

    Returneaza: Q (numpy array m x m), w (vector scoruri pe intersecții)
    """
    dice_probs = {2:1/36, 3:2/36, 4:3/36, 5:4/36, 6:5/36, 7:6/36, 8:5/36, 9:4/36, 10:3/36, 11:2/36, 12:1/36}
    hex_probs = [dice_probs[n] for n in hex_numbers]
    m = len(adjacency)
    w = np.array([sum(hex_probs[h] for h in adj) for adj in adjacency])
    print(f"w: {w}")
    if lam is None:
        lam = max(w) * 5.0
    print(f"lam: {lam}")
    Q = np.zeros((m,m))
    for i in range(m):
        Q[i,i] = -w[i] - 3*lam
    print(f"Q: {Q}")
    for i,j in itertools.combinations(range(m), 2):
        Q[i,j] = lam
        Q[j,i] = Q[i,j]
    print(f"Q: {Q}")
    return Q, w, lam

# ----------------------------
# 2) Convertire QUBO -> Ising (h, J, offset)
# ----------------------------

def qubo_to_ising(Q):
    """
    Q: numpy matrix (m x m)
    folosim transformarea x_i = (1 - Z_i)/2
    Rezultatul va fi H = const + sum_i h_i Z_i + sum_{i<j} J_ij Z_i Z_j

    Returneaza: h (length m), J (m x m symmetric, J_ij nonzero for i<j), const
    """
    m = Q.shape[0]
    # Expand: x^T Q x = sum_{i} Q_ii x_i + sum_{i<j} 2 Q_ij x_i x_j
    # Substituie x_i = (1 - Z_i)/2
    # x_i -> (1 - Z_i)/2
    # Produse si constante -> se determina coeficienti pentru Z_i si Z_i Z_j
    h = np.zeros(m)
    J = np.zeros((m,m))
    const = 0.0
    for i in range(m):
        const += Q[i,i] * 1/4
        h[i] += -Q[i,i] * 1/2  # coef pentru Z_i din termenii diagonali
    for i in range(m):
        for j in range(i+1, m):
            # coef din termenii 2*Q_ij x_i x_j
            const += 2*Q[i,j] * 1/4
            # Z_i Z_j coef
            J[i,j] += 2*Q[i,j] * 1/4
            # Z_i, Z_j coef (cross terms): each receives -2*Q_ij * 1/4
            h[i] += -2*Q[i,j] * 1/2
            h[j] += -2*Q[i,j] * 1/2
    # Ai grija la semne: verificare numerica recomandata
    # De fapt o implementare directă (expand symbolic) ar da:
    # E = sum_i Q_ii x_i + sum_{i<j} 2 Q_ij x_i x_j
    # Substitutie: x_i = (1 - Z_i)/2
    # Vom recomputa mai direct prin enumerare pentru consistenta:
    # recalculam h,J,const prin potrivire pe baza valorilor pe toate bazele computational-basis (optional)
    return h, J, const

# ----------------------------
# 3) Construire circuit QAOA p-straturi
# ----------------------------

def expectation_from_counts(counts, energies):
    """
    Calculate expectation value from measurement counts and energy values

    Args:
        counts: dictionary of measurement results and their frequencies
        energies: array of energy values for each possible state
    Returns:
        float: expected energy value
    """
    shots = sum(counts.values())
    exp_val = 0.0
    for bitstring, count in counts.items():
        idx = int(bitstring, 2)  # Convert bitstring to integer
        exp_val += (count / shots) * energies[idx]
    return exp_val


def build_qaoa_circuit(h, J, p=1):
    """
    h: vector length n
    J: matrix n x n (symmetric) with J[i,j] for i<j
    p: numarul de straturi QAOA
    """
    n = len(h)
    qr = QuantumRegister(n, 'q')
    cr = ClassicalRegister(n, 'c')
    qc = QuantumCircuit(qr, cr)

    # Parametri
    gammas = ParameterVector('gamma', length=p)
    betas = ParameterVector('beta', length=p)

    # Starea initiala |+>^n
    for i in range(n):
        qc.h(qr[i])

    # Construim p straturi
    for layer in range(p):
        gamma = gammas[layer]
        beta = betas[layer]

        # Cost unitary
        for i in range(n):
            angle = 2 * gamma * h[i]
            qc.rz(angle, qr[i])

        # ZZ terms
        for i in range(n):
            for j in range(i + 1, n):
                if abs(J[i, j]) > 1e-12:
                    angle = 2 * gamma * J[i, j]
                    qc.cx(qr[i], qr[j])
                    qc.rz(angle, qr[j])
                    qc.cx(qr[i], qr[j])

        # Mixer unitary
        for i in range(n):
            qc.rx(2 * beta, qr[i])

    qc.measure(qr, cr)
    return qc, gammas, betas


# Main execution part
if __name__ == '__main__':
    hex_numbers = [6, 4, 8, 5, 9, 10, 3]
    adjacency = [
        [0, 1, 2], [0, 2, 3], [0, 3, 4], [0, 4, 5], [0, 5, 6], [0, 6, 1],
        [1, 2], [2, 3], [3, 4], [4, 5], [5, 6], [6, 1]
    ]

    # Build QUBO and convert to Ising
    Q, w, lam = build_qubo(hex_numbers, adjacency)
    h, J, const = qubo_to_ising(Q)

    # Build QAOA circuit
    p = 1
    qc, gammas, betas = build_qaoa_circuit(h, J, p=p)

    # Bind parameters to some initial values
    parameter_values = {
        gammas[0]: 0.1,  # Initial gamma value
        betas[0]: 0.1  # Initial beta value
    }

    # Bind parameters and run simulation
    bound_qc = qc.assign_parameters(parameter_values)
    counts = measure_using_aer_simulator(bound_qc, shots=1000)

    # Print results
    print("Measurement counts:", counts)

    # Calculate expectation value
    n = len(h)
    N = 2 ** n
    E_z = np.zeros(N)
    for idx in range(N):
        bits = np.array(list(map(int, format(idx, f'0{n}b'))))
        E_z[idx] = bits @ Q @ bits

    exp_val = expectation_from_counts(counts, E_z)
    print("Expected energy:", exp_val)