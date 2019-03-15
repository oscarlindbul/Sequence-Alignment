import numpy as np
from qiskit_aqua import QuantumInstance
from qiskit_aqua.algorithms import QAOA, ExactEigensolver
from qiskit_aqua.components.optimizers import COBYLA
from qiskit import BasicAer as Aer
import MSA_column

import warnings
warnings.filterwarnings("ignore")

sequences = ["CT", "T"]
costs = [-10, 10, 1]

matchings = MSA_column.get_match_matrix(sequences, costs)

sizes = [len(sequences[i]) for i in range(len(sequences))]
inserts = 0
Hamilt, shift, rev_inds = MSA_column.get_MSA_qubitops(sizes, matchings, gap_pen=costs[2], extra_inserts=inserts)

print("Solving problem with {} qubits".format(Hamilt.num_qubits))

exact_solver = ExactEigensolver(Hamilt)
solution = exact_solver.run()
# exact solution
positions = MSA_column.sample_most_likely(solution["wavefunction"][0], rev_inds)
alignment = MSA_column.get_alignment_string(sequences, inserts, positions)
print("Exact solution")
for s in alignment:
    print(s)

# QAOA

opt = COBYLA()
solver = QAOA(Hamilt, opt, p=2)
simulator = Aer.get_backend("statevector_simulator")
instance = QuantumInstance(backend=simulator)
result = solver.run(instance)

state = result["eigvecs"][0]

positions = MSA_column.sample_most_likely(state, rev_inds)
for (key, value) in positions.items():
    print(key, value)

alignment = MSA_column.get_alignment_string(sequences, inserts, positions)
print("QAOA solution")
for s in alignment:
    print(s)
