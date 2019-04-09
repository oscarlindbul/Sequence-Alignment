from Ventura_QuAM import *

memory = np.array(["010", "100"])
pattern = np.array(["011"])
n = len(pattern[0])
sol=[2,4]

# memory = np.array(["011", "100"])
# pattern = np.array(["011"])
# n = len(pattern[0])
# sol=[3,4]

searcher = QuAM()
searcher.set_mem(memory)
print("mem state", searcher.memory, "prob", searcher.memory.norm())
searcher.set_query(pattern)
print("query state", searcher.query, "prob", searcher.query.norm())
searcher.set_oracle()
# print("query oracle", searcher.oracle)
searcher.set_diffusion()
# print("query oracle", searcher.diffusion)

result_ezhov, hist1 = searcher.match_ezhov()
success_prob = sum([abs(result_ezhov.full()[i])**2 for i in sol])
print("result_ezhov:", success_prob)
result_ezhov_exact, hist1_exact = searcher.match_ezhov(iteration="exact")
print("iterations: ", len(hist1_exact))
success_prob_exact = sum([abs(result_ezhov_exact.full()[i])**2 for i in sol])
print("result_ezhov_exact:", success_prob_exact)
success_probs_ezhov = np.zeros(len(hist1_exact))
for i in range(len(hist1_exact)):
    success_probs_ezhov[i] = np.abs(searcher.query.overlap(hist1_exact[i]))
    #success_probs_ezhov[i] = sum([abs(hist1_exact[i].full()[j])**2 for j in sol])
plt.plot(success_probs_ezhov)
plt.show()

result_C1, hist2 = searcher.match_C1()
print("C1:", result_C1)
print("iterations: ", len(hist2))
success_probs_C1 = np.zeros(len(hist2))
for i in range(len(hist2)):
    success_probs_C1[i] = np.abs(searcher.query.overlap(hist2[i]))
    #success_probs_C1[i] = sum([abs(hist2[i].full()[j])**2 for j in sol])
plt.plot(success_probs_C1)
plt.show()

# print("result_C1:", success_prob_C1)
result_C1_exact, hist2_exact = searcher.match_C1(iteration="exact")
print("iterations: ", len(hist2_exact))
print("C1 exact:", result_C1_exact)
success_prob_C1_exact = sum([abs(result_C1_exact.full()[i])**2 for i in sol])
print("result_C1_exact:", success_prob_C1_exact)

# a_prime = 0.1
# result_C2 = searcher.match_C2(a_prime)
# success_prob_C2 = sum([abs(result_C2.full()[i])**2 for i in sol])
# print("result_C2:", success_prob_C2)
# result_C2_exact = searcher.match_C2(a_prime, iteration="exact")
# success_prob_C2_exact = sum([abs(result_C2_exact.full()[i])**2 for i in sol])
# print("result_C2_exact:", success_prob_C2_exact)
