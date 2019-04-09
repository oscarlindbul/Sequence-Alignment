import sys
sys.path.append("../Ventura")
sys.path.append("../Trugenberger")

import Ventura_QuAM
import Trugenberg_QuAM

import numpy as np
import qutip as qu

"""Design a performance test for measuring maximum propability

    Params:
    1. One marked target, database filling ratio, size of database
    2. One distributed target, database filling ratio, size of database
    3. Several marked target (all hits in database), database filling ratio, size of database
    4. Several marked targets, ratio of marked hits, fixed database size
    5. Several distributed targets (center all hits), database filling ratio, fixed database size
    6. distributed target without center, Hamming distance from center to match, database size

    Output:
    1. Probs for hit
    2. Probs for hit
    3. Probs for hits / Prob for one hit
    4. Prob for hits
    5. Probs for centers / Probs for one center
    6. Probs for hit
"""

def int_to_bitstr(integer, bits=None):
    if bits is None:
        return format(integer, "b")
    else:
        return format(integer, "0" + bits + "b")

def Test1():
    max_bits = 15
    n_bits = np.arange(2, max_bits)
    number_of_marks = 1000
    ratio_vals = 100
    ratio_range = np.linspace(0.1, 0.8, ratio_vals)

    ventura_probs = np.zeros((max_bits, ratio_vals, number_of_marks))
    trugenberg_probs = np.zeros((max_bits, ratio_vals, number_of_marks))
    for i1 in range(len(n_bits)):
        n = n_bits[i1]
        N = 2**n
        mark_num = min(N, number_of_marks)
        for i2 in range(mark_num):
            mark = numpy.random.randint(0, mark_num)

            mem_states = set()
            for i3 in range(len(ratio_range)):
                ratio = ratio_range[i3]
                mem_states.clear()
                possible_states = [i for i in range(N)]
                N_mem = ceil(ratio*N)
                for j in range(N_mem):
                    num_ind = numpy.random.randint(0, len(possible_states))
                    mem_states.add(possible_states.pop(num_ind))

                query = [int_to_bitstr(mark)]
                memory = [int_to_bitstr(mem) for mem in mem_states]
                ventura = Ventura_QuAM.QuAM()
                ventura.set_mem(memory)
                ventura.set_query(query)
                ventura.set_oracle()
                ventura.set_diffusion()
                ventura.set_state()

                _, ventura_states = ventura.match_ezhov()

                max_prob = max([abs(state.full())**2 for state in ventura_states[:, mark]])
                ventura_probs[i1,i2,i3] += max_prob
        ventura_probs /= mark_num

    # plot result
