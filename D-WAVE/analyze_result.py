from data_formats import SeqQuery
from qiskit_aqua.algorithms import ExactEigensolver
import numpy as np
import matplotlib.pyplot as plt
import dimod
import neal
from timeit import default_timer as timer
import pickle

load_file = "d_wave_temp_data.dat"
gap_char = "n"

file = open(load_file, "rb")
data = pickle.load(file)
file.close()
seqs = data.sequences
sizes = [len(s) for s in seqs]
inserts = data.params["extra_inserts"]
ind_map = data.rev_inds
h = np.array(list(data.h.values()))
J = np.array(list(data.J.values()))

print("h span =", "[{},{}]".format(np.min(np.abs(h[h != 0])), np.max(np.abs(h[h != 0]))))
print("J span =", "[{},{}]".format(np.min(np.abs(J[J != 0])), np.max(np.abs(J[J != 0]))))



result_dic = {}
response = data.response
align_string = np.chararray((len(seqs), max(sizes)+inserts), unicode=True)
for samples, energy, occurrences in data.response.data(["sample", "energy", "num_occurrences"]):
    key_string = ""
    align_string[:] = gap_char
    included_bases = 0
    for key, value in samples.items():
        if value == 1:
            included_bases += 1
            (s, n, i) = ind_map[int(key)]
            if align_string[s,i] != gap_char:
                key_string = "invalid"
            else:
                align_string[s,i] = seqs[s][n]
    if included_bases != sum(sizes):
        key_string = "invalid"
    elif key_string != "invalid":
        for i in range(align_string.shape[0]):
            for j in range(align_string.shape[1]):
                key_string += align_string[i,j]
            if i < align_string.shape[0]-1:
                key_string += "\n"
    if key_string in result_dic:
        if key_string == "invalid":
            result_dic[key_string]["occurrences"] = max([result_dic[key_string]["occurrences"], occurrences])
            if occurrences > 5000:
                print(samples, occurrences)
        else:
            result_dic[key_string]["occurrences"] += occurrences
    else:
        result_dic[key_string] = {}
        result_dic[key_string]["occurrences"] = occurrences
        result_dic[key_string]["energy"] = energy


# for (s,n,i) in included:
#     align_string[s,i] = sequences[s][n]
# for i in range(align_string.shape[0]):
#     for j in range(align_string.shape[1]):
#         print(align_string[i,j],end="")
#     print()
# for energy, occur in response.data(['energy', 'num_occurrences']):
#     if energy not in result_dic:
#         result_dic[energy] = occur
#     else:
#         result_dic[energy] += occur
val_length = min([20, len(result_dic.keys())])
bar_inds = np.arange(val_length)
bar_labels = (list(result_dic.keys()))[:val_length]
bar_values = np.zeros(val_length)
bar_energies = np.zeros(val_length)
for i in range(val_length):
    vals = result_dic[bar_labels[i]]
    bar_values[i] = vals["occurrences"]
    bar_energies[i] = vals["energy"]

def align_yaxis(ax1, v1, ax2, v2):
    """adjust ax2 ylimit so that v2 in ax2 is aligned to v1 in ax1"""
    _, y1 = ax1.transData.transform((0, v1))
    _, y2 = ax2.transData.transform((0, v2))
    inv = ax2.transData.inverted()
    _, dy = inv.transform((0, 0)) - inv.transform((0, y1-y2))
    miny, maxy = ax2.get_ylim()
    ax2.set_ylim(miny+dy, maxy+dy)

energy_color="red"
count_color="blue"
fig, ax1 = plt.subplots()
width=0.35
ax1.bar(bar_inds-width/2, bar_values, width=width, color=count_color)
ax1.set_xlabel("Matchings")
ax1.set_ylabel("Counts", color=count_color)
ax1.set_xticks(bar_inds)
ax1.set_xticklabels(bar_labels, fontsize=7)
ax1.tick_params(axis="y", labelcolor=count_color)

ax2 = ax1.twinx()
ax2.bar(bar_inds+width/2, bar_energies, width=width, color=energy_color)
ax2.set_ylabel("Energy/Cost", color=energy_color)
ax2.tick_params(axis="y", labelcolor=energy_color)

y_lim = ax1.get_ylim()
if max(bar_energies) != 0:
    scale = min(bar_energies)/max(bar_energies)
else:
    scale = min(bar_energies)
y_min = scale*max(bar_values)
if y_min < 0:
    ax1.set_ylim([y_min, ax1.get_ylim()[1]])
align_yaxis(ax1, 0, ax2, 0)

title = "Matches for sequences: " + ", ".join(seqs)
fig.tight_layout()
plt.title(title)
plt.show()
# for align_string, val_dic in result_dic.items():
#     print()
#     print(align_string, val_dic["energy"], val_dic["occurrences"])
