import MSA_column
from data_formats import SeqQuery
from qiskit_aqua.algorithms import ExactEigensolver
import numpy as np
import dimod
import neal
from timeit import default_timer as timer
import pickle

from dwave.system.samplers import DWaveSampler
from dwave.system.composites import EmbeddingComposite

samples = 10000
save_file = "d_wave_temp_data.dat"
# sequences = ["CT", "T"]
sequences = ["ATGC", "T", "AC", "GC"]
simulation = True
match_cost = -1
mismatch_cost = 1

# quantum spin column version
sizes = [len(sequences[i]) for i in range(len(sequences))]
# calculate weights for matching
matchings = np.zeros((len(sequences), max(sizes), len(sequences), max(sizes)))
for s1 in range(len(sequences)):
    for s2 in range(len(sequences)):
        for n1 in range(sizes[s1]):
            for n2 in range(sizes[s2]):
                if sequences[s1][n1] == sequences[s2][n2]:
                    matchings[s1,n1,s2,n2] = match_cost
                else:
                    matchings[s1,n1,s2,n2] = mismatch_cost

inserts = 0
gap_penalty = 0
params = {"gap_pen": gap_penalty, "extra_inserts": inserts}
mat, shift, rev_inds = MSA_column.get_MSA_qubitmat(sizes, matchings, gap_pen=gap_penalty, extra_inserts=inserts)

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
print("h", min(list(h.values())), max(list(h.values())))
print("J", min(list(J.values())), max(list(J.values())))
if not simulation:
    cont = input("Continue? y/n ")
    if cont != "y":
        exit()

bqm = dimod.BinaryQuadraticModel(h,J, shift, dimod.BINARY)
if simulation:
    solver = neal.SimulatedAnnealingSampler()
else:
    solver = EmbeddingComposite(DWaveSampler())
response = solver.sample(bqm, num_reads=samples)
result_dic = {}
for energy, occur in response.data(['energy', 'num_occurrences']):
    if energy not in result_dic:
        result_dic[energy] = occur
    else:
        result_dic[energy] += occur
print(result_dic)

data_query = SeqQuery()
data_query.sequences = sequences
data_query.params = params
data_query.costs = [match_cost, mismatch_cost, gap_penalty]
data_query.spin_mat = mat
data_query.spin_shift = shift
data_query.rev_inds = rev_inds
data_query.h = h
data_query.J = J
data_query.response = response

# add pickling of results
file = open(save_file, "wb")
pickle.dump(data_query, file, pickle.HIGHEST_PROTOCOL)
file.close()

result_dic_2 = {}
file = open(save_file, "rb")
data = pickle.load(file)
response = data.response
file.close()
for energy, occur in response.data(['energy', 'num_occurrences']):
    if energy not in result_dic_2:
        result_dic_2[energy] = occur
    else:
        result_dic_2[energy] += occur
print(result_dic_2)
