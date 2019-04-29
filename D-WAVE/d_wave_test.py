import MSA_column
from qiskit_aqua.algorithms import ExactEigensolver
import numpy as np
import dimod
import neal
from timeit import default_timer as timer
import pickle

sequences = ["CT", "T"]
#sequences = ["CTC", "TA", "AT"]

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

inserts = 0
mat, shift, rev_inds = MSA_column.get_MSA_qubitmat(sizes, matchings, gap_pen=1, extra_inserts=inserts)

def mat_to_dimod_format(matrix):
    n = matrix.shape[0]
    linear = {}
    interaction = {}
    for i in range(n):
        linear[i] = matrix[i,i]
        for j in range(i+1,n):
            interaction[(i,j)] = matrix[i,j]
    return linear, interaction

h, J = mat_to_dimod_format(mat)

bqm = dimod.BinaryQuadraticModel(h,J, shift, dimod.BINARY)
solver = neal.SimulatedAnnealingSampler()
response = solver.sample(bqm, num_reads=50)
for sample, energy in response.data(['sample', 'energy']):
    print(sample, energy)

# add pickling of results
