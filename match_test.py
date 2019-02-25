import MSA_classical
import MSA_column
from qiskit_aqua.algorithms import ExactEigensolver
import numpy as np

sequences = ["CT", "TA", "AT"]
costs = [1, -1, 0]

# classical version
MSA_classical.solve_MSA(sequences, costs)

# quantum spin column version
sizes = [len(sequences[i]) for i in range(len(sequences))]
# calculate weights for matching
matchings = np.zeros((len(sequences), max(sizes), len(sequences), max(sizes)))
for s1 in range(len(sequences)):
    for s2 in range(len(sequences)):
        for n1 in range(sizes[s1]):
            for n2 in range(sizes[s2]):
                if sequences[s1][n1] == sequences[s2][n2]:
                    matchings[s1,n1,s2,n2] = -1
                else:
                    matchings[s1,n1,s2,n2] = 1

Hamilt, shift, rev_inds = MSA_column.get_MSA_qubitops(sizes, matchings, gap_pen=1, extra_inserts=0)
print("Written")
ee = ExactEigensolver(Hamilt)
result = ee.run()
print("Solved")
state = result['wavefunction'][0]
ind = 0
for i in range(len(state)):
    if state[len(state) - i - 1] != 0:
        ind = i
        break
print(Hamilt)
qubits = Hamilt.num_qubits
bin_string = format(ind, "0" + str(qubits) + "b")[::-1]
print(bin_string)
bin_arr = [s == '1' for s in bin_string]
included = rev_inds[bin_arr]
align_string = np.chararray((len(sequences), max(sizes)), unicode=True)
align_string[:] = '-'
for (s,n,i) in included:
    align_string[s,i] = sequences[s][n]
for i in range(align_string.shape[0]):
    for j in range(align_string.shape[1]):
        print(align_string[i,j],end="")
    print()
