from QuAM import *

memory = np.array(["010", "100"])
pattern = np.array(["011"])
n = len(pattern[0])
sol=[2,4]

searcher = QuAM()
searcher.set_mem(memory)
print("mem state", searcher.memory, "prob", searcher.memory.norm())
searcher.set_query(pattern)
print("query state", searcher.query, "prob", searcher.query.norm())
searcher.set_oracle()
# print("query oracle", searcher.oracle)
searcher.set_diffusion()
# print("query oracle", searcher.diffusion)

result_ezhov = searcher.match_ezhov()
success_prob = sum([abs(result_ezhov.full()[i])**2 for i in sol])
print("result_ezhov:", success_prob)
result_ezhov_exact = searcher.match_ezhov(iteration="exact")
success_prob_exact = sum([abs(result_ezhov_exact.full()[i])**2 for i in sol])
print("result_ezhov_exact:", success_prob_exact)

result_C1 = searcher.match_C1(iteration=25)
print("C1:", result_C1)
success_prob_C1 = sum([abs(result_C1.full()[i])**2 for i in sol])
print("result_C1:", success_prob_C1)
result_C1_exact = searcher.match_C1(iteration="exact")
print("C1 exact:", result_C1_exact)
success_prob_C1_exact = sum([abs(result_C1_exact.full()[i])**2 for i in sol])
print("result_C1_exact:", success_prob_C1_exact)

a_prime = 0.1
result_C2 = searcher.match_C2(a_prime)
success_prob_C2 = sum([abs(result_C2.full()[i])**2 for i in sol])
print("result_C2:", success_prob_C2)
result_C2_exact = searcher.match_C2(a_prime, iteration="exact")
success_prob_C2_exact = sum([abs(result_C2_exact.full()[i])**2 for i in sol])
print("result_C2_exact:", success_prob_C2_exact)
