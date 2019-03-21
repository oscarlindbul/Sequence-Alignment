import warnings
warnings.filterwarnings("ignore")

from qiskit import IBMQ
from qiskit.tools.monitor import job_monitor

import numpy as np
from qiskit_aqua import QuantumInstance
from qiskit_aqua.algorithms import QAOA, VQE, ExactEigensolver
from qiskit_aqua.components.optimizers import SPSA, COBYLA
from qiskit_aqua.components.variational_forms import RY
from qiskit_aqua.components.initial_states import Custom as CustomState
from qiskit import BasicAer as Aer
import MSA_column

# load IBM credentials
IBMQ.load_accounts()

backend = "ibmq_qasm_simulator"
shots = 1024
method = input("Quantum method? ")

sequences = ["CTA", "T"]
costs = [-10, 10, 10]

matchings = MSA_column.get_match_matrix(sequences, costs)

sizes = [len(sequences[i]) for i in range(len(sequences))]
inserts = 0
Hamiltonian_penalty = 100
Hamilt, shift, rev_inds = MSA_column.get_MSA_qubitops(sizes, matchings, gap_pen=costs[2], extra_inserts=inserts, coeffs=[1, Hamiltonian_penalty])
seq_inds = {}
for i in range(len(rev_inds)):
    seq_inds[rev_inds[i]] = i

print("Solving problem with {} qubits".format(Hamilt.num_qubits))

exact_solver = ExactEigensolver(Hamilt)
solution = exact_solver.run()
# exact solution
positions = MSA_column.sample_most_likely(solution["wavefunction"][0], rev_inds)
alignment = MSA_column.get_alignment_string(sequences, inserts, positions)
print("Exact solution: Energy=", solution["eigvals"][0] + shift)
for s in alignment:
    print(s)

# QAOA
if method == "QAOA":
    # initial_state = np.zeros(2**Hamilt.num_qubits)
    # state_bit_arr = np.zeros(Hamilt.num_qubits)
    # for s in range(len(sequences)):
    #     for n in range(len(sequences[s])):
    #         state_bit_arr[seq_inds[(s,n,n)]] = 1
    # print(state_bit_arr)
    # initial_state[int(np.sum(state_bit_arr*np.power(2, np.arange(Hamilt.num_qubits,dtype=np.int32))))] = 1
    # initial_state = CustomState(Hamilt.num_qubits, state_vector=initial_state)

    opt = COBYLA()
    p = 10
    solver = QAOA(Hamilt, opt, p)
    simulator = IBMQ.get_backend(backend)
    instance = QuantumInstance(backend=simulator, shots=shots)
    result = solver.run(instance)

    state = result["eigvecs"][0]
    energy = result["eigvals"][0] + shift

    positions = MSA_column.sample_most_likely(state, rev_inds)
    for (key, value) in positions.items():
        print(key, value)

    alignment = MSA_column.get_alignment_string(sequences, inserts, positions)
    print("QAOA solution: Energy=", energy)
    for s in alignment:
        print(s)

# VQE
if method == "VQE":
    opt = SPSA(max_trials=100)
    trial = RY(Hamilt.num_qubits, depth=10, entanglement='linear')
    solver = VQE(Hamilt, trial, opt, "paulis")
    simulator = IBMQ.get_backend(backend)
    instance = QuantumInstance(backend=simulator, shots=1024)
    print("Starting job")
    result = solver.run(instance)
    print("Job finished?")

    state = result["eigvecs"][0]
    energy = result["eigvals"][0] + shift

    positions = MSA_column.sample_most_likely(state, rev_inds)
    for (key, value) in positions.items():
        print(key, value)

    alignment = MSA_column.get_alignment_string(sequences, inserts, positions)
    print("VQE solution: Energy=", energy)
    for s in alignment:
        print(s)
