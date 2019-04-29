import sys
sys.path.append("../Ventura")
sys.path.append("../Trugenberger")

import Ventura_QuAM
import Trugenberg_QuAM

import numpy as np
import qutip as qu
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

"""Design a performance test for measuring maximum propability

    Params:
    1. One localized marked target, database filling ratio, size of database
    2. One distributed target, database filling ratio, size of database
    3. Several localized marked target (all hits in database), database filling ratio, size of database
    4. Several localized marked targets, ratio of marked hits, fixed database size
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
        return format(integer, "0" + str(bits) + "b")

def hamm_dist(state1, state2):
    n = int(np.ceil(max(np.log2(state1+1), np.log2(state2+1))))
    s_state1 = format(state1, "0" + str(n) + "b")
    s_state2 = format(state2, "0" + str(n) + "b")

    return str_dist(s_state1, s_state2)

def str_dist(s1, s2):
    dist = 0
    for (c1,c2) in zip(s1, s2):
        if c1 != c2:
            dist += 1
    return dist

def Test1(data_file=None, method="ezhov"):
    """Test1

    Match with a spiked query, iterating over number of qubits, filling of database, as well as averaging over random choices of marked state.

    """
    max_bits = 7 # maximum number of qubits
    n_bits = np.arange(3, max_bits+1)
    n_nums = len(n_bits)
    number_of_marks = 20 # number of random marks to test for
    ratio_vals = 20 # number of ratio parameters to test for
    ratio_range = np.linspace(0.1, 0.9, ratio_vals)
    peak_factor = 0.9 # 0 is a delta peak at mark

    data_shape = (n_nums, number_of_marks, ratio_vals)
    ventura_probs = np.zeros(data_shape)
    ventura_prob_inds = np.zeros(data_shape)
    # trugenberg_probs = np.zeros((n_nums, number_of_marks, ratio_vals))
    for i1 in range(len(n_bits)): # iterate number of bits
        n = n_bits[i1]
        N = 2**n
        mark_num = min(N, number_of_marks) # number of marks to test
        for i2 in range(mark_num): # iterate number of marks
            mark = np.random.randint(0, mark_num) # random mark

            mem_states = set()
            for i3 in range(len(ratio_range)): # iterate ratios
                ratio = ratio_range[i3]

                # create memory list
                mem_states.clear()
                possible_states = [i for i in range(N)]
                # number of memory states
                N_mem = int(np.ceil(ratio*N))-1
                if N_mem >= N-2:
                    continue
                # add mark to memory
                mem_states.add(possible_states.pop(mark))
                for j in range(N_mem): # add random memories
                    num_ind = np.random.randint(0, len(possible_states))
                    add_state = possible_states.pop(num_ind)
                    mem_states.add(add_state)

                # make query and memory list
                query = [int_to_bitstr(i, n) for i in range(N)]
                query_weights = [(1-peak_factor)/(N-1) if i != mark else peak_factor for i in range(N)]
                memory = [int_to_bitstr(mem, n) for mem in mem_states]

                # initialize QuAM
                ventura = Ventura_QuAM.QuAM()
                ventura.set_mem(memory)
                ventura.set_query(query, weights=query_weights) #bin_param=peaking_factor)
                ventura.set_oracle()
                ventura.set_diffusion()
                ventura.set_max_iterations(3000)
                print("memory:", memory)
                # print("query:", query)

                # perform matching
                repeat_case = True
                iteration_limit = int(N)
                while repeat_case:
                    repeat_case = False
                    if method == "ezhov":
                        _, ventura_states = ventura.match_ezhov(iteration=iteration_limit)
                    if method == "improved":
                        _, ventura_states = ventura.match_C1(iteration=iteration_limit)

                    # gather probabilites
                    success_probs = [abs(state.full()[mark])**2 for state in ventura_states]
                    max_prob_ind = np.argmax(np.array(success_probs))
                    max_prob = success_probs[max_prob_ind]
                    print("n  ratio    prob     ind")
                    print(n, format(ratio, "0.3f"), max_prob, max_prob_ind)
                    # if max_prob < 0.9:
                    #     plt.plot(success_probs)
                    #     plt.show()
                    ventura_probs[i1,i2,i3] = max_prob
                    ventura_prob_inds[i1,i2,i3] = max_prob_ind
                    if max_prob_ind >= iteration_limit or max_prob_ind <= 1:
                        print("repeating test case")
                        iteration_limit += 500
                        repeat_case = True

    # save results
    if data_file is None:
        data_file = "data"

    np.savez(data_file, bits=n_bits, ratios=ratio_range, peak_factor=np.array([peak_factor]), values=ventura_probs, iters=ventura_prob_inds)

def Test2(data_file=None, method="ezhov"):
    max_bits = 7 # maximum number of qubits
    n_bits = np.arange(3, max_bits+1)
    n_nums = len(n_bits)
    number_of_marks = 20 # number of random marks to test for
    ratio_vals = 20 # number of ratio parameters to test for
    ratio_range = np.linspace(0.1, 0.9, ratio_vals)
    peak_factor = 0.25 # 0 is a delta peak at mark

    data_shape = (n_nums, number_of_marks, ratio_vals)
    ventura_probs = np.zeros(data_shape)
    ventura_prob_inds = np.zeros(data_shape)
    # trugenberg_probs = np.zeros((n_nums, number_of_marks, ratio_vals))
    for i1 in range(len(n_bits)): # iterate number of bits
        n = n_bits[i1]
        N = 2**n
        mark_num = min(N, number_of_marks) # number of marks to test
        for i2 in range(mark_num): # iterate number of marks
            mark = np.random.randint(0, mark_num) # random mark

            mem_states = set()
            for i3 in range(len(ratio_range)): # iterate ratios
                ratio = ratio_range[i3]

                # create memory list
                mem_states.clear()
                possible_states = [i for i in range(N)]
                # number of memory states
                N_mem = int(np.ceil(ratio*N))-1
                if N_mem >= N-2:
                    continue
                # add mark to memory
                mem_states.add(possible_states.pop(mark))
                for j in range(N_mem): # add random memories
                    num_ind = np.random.randint(0, len(possible_states))
                    add_state = possible_states.pop(num_ind)
                    mem_states.add(add_state)

                # make query and memory list
                query = [int_to_bitstr(mark, n) for i in range(N)]
                # query_weights = [(1-peak_factor)/(N-1) if i != mark else peak_factor for i in range(N)]
                memory = [int_to_bitstr(mem, n) for mem in mem_states]

                # initialize QuAM
                ventura = Ventura_QuAM.QuAM()
                ventura.set_mem(memory)
                ventura.set_query(query, bin_param = peak_factor)
                ventura.set_oracle()
                ventura.set_diffusion()
                ventura.set_max_iterations(3000)
                print("memory:", memory)
                # print("query:", query)

                # perform matching
                max_prob_ind = np.inf
                iteration_limit = int(N)
                repeat_case = True
                while repeat_case:
                    repeat_case = False
                    if method=="ezhov":
                        _, ventura_states = ventura.match_ezhov(iteration=iteration_limit)
                    elif method == "improved":
                        _, ventura_states = ventura.match_C1(iteration=iteration_limit)

                    # gather probabilites
                    success_probs = [abs(state.full()[mark])**2 for state in ventura_states]
                    max_prob_ind = np.argmax(np.array(success_probs))
                    max_prob = success_probs[max_prob_ind]
                    print("n  ratio    prob      ind")
                    print(n, format(ratio, "0.3f"), max_prob, max_prob_ind)
                    # if max_prob < 0.9:
                    #     plt.plot(success_probs)
                    #     plt.show()
                    ventura_probs[i1,i2,i3] = max_prob
                    ventura_prob_inds[i1, i2, i3] = max_prob_ind
                    if max_prob_ind >= iteration_limit or max_prob_ind <= 1:
                        print("repeating test case")
                        iteration_limit += 500
                        repeat_case = True

    # save results
    if data_file is None:
        data_file = "data"

    np.savez(data_file, bits=n_bits, ratios=ratio_range, peak_factor=np.array([peak_factor]), values=ventura_probs, value_inds=ventura_prob_inds)

def Test3(data_file=None, method="ezhov"):
    """Test3

    Match with a spiked query of multiple marks, iterating over number of qubits, filling of database, as well as averaging over random choices of marked state. Several numbers of targets are tested

    """
    max_bits = 7 # maximum number of qubits
    n_bits = np.arange(3, max_bits+1)
    n_nums = len(n_bits)
    number_of_marks = 20 # number of random marks to test for
    target_num = np.arange(1, 6) # number of targets in query
    ratio_vals = 20 # number of ratio parameters to test for
    ratio_range = np.linspace(0.1, 0.9, ratio_vals)
    peak_factor = 0.5 # 0 is a delta peak at mark

    data_shape = (n_nums, number_of_marks, ratio_vals, len(target_num))
    ventura_probs = np.zeros(data_shape)
    ventura_prob_inds = np.zeros(data_shape)
    # trugenberg_probs = np.zeros((n_nums, number_of_marks, ratio_vals))
    for i1 in range(len(n_bits)): # iterate number of bits
        n = n_bits[i1]
        N = 2**n
        mark_num = min(N, number_of_marks) # number of marks to test
        for i2 in range(mark_num): # iterate number of marks
            for mark_ind in range(len(target_num)):
                q_N = target_num[mark_ind]
                marks = set()
                possible_states = [i for i in range(N)]
                for i in range(q_N):
                    rand_ind = np.random.randint(0, len(possible_states))
                    marks.add(possible_states.pop(rand_ind))
                mark_projector = qu.Qobj()
                for mark in marks:
                    mark_projector += qu.projection(N, mark, mark)
                mark_projector = mark_projector.unit()

                mem_states = set()
                for i3 in range(len(ratio_range)): # iterate ratios
                    ratio = ratio_range[i3]

                    # create memory list
                    mem_states.clear()
                    # number of memory states
                    N_mem = int(np.ceil(ratio*N))-q_N
                    if N_mem >= N-q_N-1:
                        continue
                    # add mark to memory
                    for mark in marks:
                        mem_states.add(mark)
                    available_states = possible_states.copy()
                    for j in range(N_mem): # add random memories
                        num_ind = np.random.randint(0, len(available_states))
                        add_state = available_states.pop(num_ind)
                        mem_states.add(add_state)

                    # make query and memory list
                    query = [int_to_bitstr(i, n) for i in range(N)]
                    query_weights = [peak_factor if i in marks else (1-peak_factor)/(N-q_N)/q_N for i in range(N)]
                    memory = [int_to_bitstr(mem, n) for mem in mem_states]
                    mark_list = [int_to_bitstr(mark, n) for mark in marks]

                    # initialize QuAM
                    ventura = Ventura_QuAM.QuAM()
                    ventura.set_mem(memory)
                    ventura.set_query(query, weights=query_weights) #bin_param=peaking_factor)
                    ventura.set_oracle()
                    ventura.set_diffusion()
                    ventura.set_max_iterations(3000)
                    print("memory:", memory)
                    print("query:", mark_list)

                    # perform matching
                    iteration_limit = int(N)
                    repeat_case = True
                    while repeat_case:
                        repeat_case = False
                        if method == "ezhov":
                            _, ventura_states = ventura.match_ezhov(iteration=iteration_limit)
                        elif method == "improved":
                            _, ventura_states = ventura.match_C1(iteration=iteration_limit)


                        # gather probabilites
                        success_probs = [(mark_projector*state).norm()**2 for state in ventura_states]

                        max_prob_ind = np.argmax(success_probs)
                        max_prob = success_probs[max_prob_ind]
                        print("n  q_N  ratio    prob    ind")
                        print(n, q_N, format(ratio, "0.3f"), max_prob, max_prob_ind)
                        # if max_prob < 0.9:
                        #     plt.plot(success_probs)
                        #     plt.show()
                        ventura_probs[i1,i2,i3, mark_ind] = max_prob
                        ventura_prob_inds[i1,i2,i3,mark_ind] = max_prob_ind
                        if max_prob_ind >= iteration_limit:
                            print("repeating test case")
                            repeat_case = True
                            iteration_limit += 500


    # save results
    if data_file is None:
        data_file = "data"

    np.savez(data_file, bits=n_bits, ratios=ratio_range, peak_factor=np.array([peak_factor]), target_nums=target_num, values=ventura_probs, iters=ventura_prob_inds)

def Test4(data_file=None):
    """Test4

    Match with a spiked query of multiple marks, iterating over number of qubits, filling of database, as well as averaging over random choices of marked state. Several numbers of targets are tested, but here, only one target is in memory

    """
    max_bits = 7 # maximum number of qubits
    n_bits = np.arange(3, max_bits+1)
    n_nums = len(n_bits)
    number_of_marks = 20 # number of random marks to test for
    target_num = np.arange(1, 6) # number of targets in query
    ratio_vals = 20 # number of ratio parameters to test for
    ratio_range = np.linspace(0.1, 0.9, ratio_vals)
    peak_factor = 0.9 # 0 is a delta peak at mark

    ventura_probs = np.zeros((n_nums, number_of_marks, ratio_vals, len(target_num)))
    # trugenberg_probs = np.zeros((n_nums, number_of_marks, ratio_vals))
    for i1 in range(len(n_bits)): # iterate number of bits
        n = n_bits[i1]
        N = 2**n
        mark_num = min(N, number_of_marks) # number of marks to test
        for i2 in range(mark_num): # iterate number of marks
            for mark_ind in range(len(target_num)):
                q_N = target_num[mark_ind]
                marks = set()
                possible_states = [i for i in range(N)]
                for i in range(q_N):
                    rand_ind = np.random.randint(0, len(possible_states))
                    marks.add(possible_states.pop(rand_ind))
                # mark_projector = qu.Qobj()
                # for mark in marks:
                #     mark_projector += qu.projection(N, mark, mark)
                # mark_projector = mark_projector.unit()

                mem_states = set()
                for i3 in range(len(ratio_range)): # iterate ratios
                    ratio = ratio_range[i3]

                    # create memory list
                    mem_states.clear()
                    # number of memory states
                    N_mem = int(np.ceil(ratio*N))-q_N
                    if N_mem >= N-q_N-1:
                        continue
                    # add mark to memory
                    mem_mark = marks.pop()
                    mem_states.add(mem_mark)
                    marks.add(mem_mark)
                    mark_projector = qu.projection(N, mem_mark, mem_mark)
                    available_states = possible_states.copy()
                    for j in range(N_mem): # add random memories
                        num_ind = np.random.randint(0, len(available_states))
                        add_state = available_states.pop(num_ind)
                        mem_states.add(add_state)

                    # make query and memory list
                    query = [int_to_bitstr(i, n) for i in range(N)]
                    query_weights = [peak_factor/q_N if i in marks else (1-peak_factor)/(N-q_N) for i in range(N)]
                    memory = [int_to_bitstr(mem, n) for mem in mem_states]
                    mark_list = [int_to_bitstr(mark, n) for mark in marks]

                    # initialize QuAM
                    ventura = Ventura_QuAM.QuAM()
                    ventura.set_mem(memory)
                    ventura.set_query(query, weights=query_weights) #bin_param=peaking_factor)
                    ventura.set_oracle()
                    ventura.set_diffusion()
                    ventura.set_max_iterations(3000)
                    print("memory:", memory)
                    print("query:", mark_list)

                    # perform matching
                    _, ventura_states = ventura.match_ezhov(iteration=1000)

                    # gather probabilites
                    success_probs = [(mark_projector*state).norm()**2 for state in ventura_states]

                    max_prob = max(success_probs)
                    print("n  q_N  ratio    prob")
                    print(n, q_N, format(ratio, "0.3f"), max_prob)
                    # if max_prob < 0.9:
                    #     plt.plot(success_probs)
                    #     plt.show()
                    ventura_probs[i1,i2,i3, mark_ind] = max_prob

    # save results
    if data_file is None:
        data_file = "data"

    np.savez(data_file, bits=n_bits, ratios=ratio_range, peak_factor=np.array([peak_factor]), target_nums=target_num, values=ventura_probs)

def Test5(data_file=None):
    """Test5

    Match with a distributed query of multiple marks, iterating over number of qubits, filling of database, as well as averaging over random choices of marked state. Several numbers of targets are tested, but here, only center targets are in memory

    """
    max_bits = 7 # maximum number of qubits
    n_bits = np.arange(3, max_bits+1)
    n_nums = len(n_bits)
    number_of_marks = 20 # number of random marks to test for
    target_num = np.arange(1, 5) # number of targets in query
    ratio_vals = 20 # number of ratio parameters to test for
    ratio_range = np.linspace(0.1, 0.9, ratio_vals)
    peak_factor = 0.1 # 0 is a delta peak at mark

    ventura_probs = np.zeros((n_nums, number_of_marks, ratio_vals, len(target_num)))
    # trugenberg_probs = np.zeros((n_nums, number_of_marks, ratio_vals))
    for i1 in range(len(n_bits)): # iterate number of bits
        n = n_bits[i1]
        N = 2**n
        mark_num = min(N, number_of_marks) # number of marks to test
        for i2 in range(mark_num): # iterate number of marks
            for mark_ind in range(len(target_num)):
                q_N = target_num[mark_ind]
                marks = set()
                possible_states = [i for i in range(N)]
                for i in range(q_N):
                    rand_ind = np.random.randint(0, len(possible_states))
                    marks.add(possible_states.pop(rand_ind))
                mark_projector = qu.Qobj()
                for mark in marks:
                    mark_projector += qu.projection(N, mark, mark)
                mark_projector = mark_projector.unit()

                mem_states = set()
                for i3 in range(len(ratio_range)): # iterate ratios
                    ratio = ratio_range[i3]

                    # create memory list
                    mem_states.clear()
                    # number of memory states
                    N_mem = int(np.ceil(ratio*N))-q_N
                    if N_mem >= N-q_N-1:
                        continue
                    # add mark to memory
                    for mark in marks:
                        mem_states.add(mark)
                    available_states = possible_states.copy()
                    for j in range(N_mem): # add random memories
                        num_ind = np.random.randint(0, len(available_states))
                        add_state = available_states.pop(num_ind)
                        mem_states.add(add_state)

                    # make query and memory list
                    query = [int_to_bitstr(mark, n) for mark in marks]
                    # query_weights = [peak_factor/q_N if i in marks else (1-peak_factor)/(N-q_N) for i in range(N)]
                    memory = [int_to_bitstr(mem, n) for mem in mem_states]
                    # mark_list = [int_to_bitstr(mark, n) for mark in marks]

                    # initialize QuAM
                    ventura = Ventura_QuAM.QuAM()
                    ventura.set_mem(memory)
                    ventura.set_query(query, bin_param=peak_factor) #bin_param=peaking_factor)
                    ventura.set_oracle()
                    ventura.set_diffusion()
                    ventura.set_max_iterations(1000)
                    print("memory:", memory)
                    print("query:", query)

                    # perform matching
                    _, ventura_states = ventura.match_ezhov(iteration=1000)

                    # gather probabilites
                    success_probs = [(mark_projector*state).norm()**2 for state in ventura_states]

                    max_prob = max(success_probs)
                    print("n  q_N  ratio    prob")
                    print(n, q_N, format(ratio, "0.3f"), max_prob)
                    # if max_prob < 0.9:
                    #     plt.plot(success_probs)
                    #     plt.show()
                    ventura_probs[i1,i2,i3, mark_ind] = max_prob

    # save results
    if data_file is None:
        data_file = "data"

    np.savez(data_file, bits=n_bits, ratios=ratio_range, peak_factor=np.array([peak_factor]), target_nums=target_num, values=ventura_probs)

def Test6(data_file=None):
    """Test6

    Match with a distributed query of one target, iterating over number of qubits, filling of database, as well as averaging over random choices of marked state. Center is not in memory, but one pattern a distance away is present.

    """
    max_bits = 7 # maximum number of qubits
    n_bits = np.arange(4, max_bits+1)
    n_nums = len(n_bits)
    number_of_marks = 20 # number of random marks to test for
    distances = np.arange(0, 4) # number of targets in query
    ratio_vals = 20 # number of ratio parameters to test for
    ratio_range = np.linspace(0.1, 0.9, ratio_vals)
    peak_factor = 0.1 # 0 is a delta peak at mark

    ventura_probs = np.zeros((n_nums, number_of_marks, ratio_vals, len(distances)))
    # trugenberg_probs = np.zeros((n_nums, number_of_marks, ratio_vals))
    for i1 in range(len(n_bits)): # iterate number of bits
        n = n_bits[i1]
        N = 2**n
        mark_num = min(N, number_of_marks) # number of marks to test
        for i2 in range(mark_num): # iterate number of marks
            mark = np.random.randint(0, N)
            for dist_ind in range(len(distances)):
                dist = distances[dist_ind]
                possible_states = [i for i in range(N)]
                possible_states.pop(mark)
                first = True
                dist_states = set()
                mark_projector = qu.Qobj()
                i = 0
                while i < len(possible_states):
                    if hamm_dist(possible_states[i], mark) == dist:
                        dist_states.add(possible_states.pop(i))
                    else:
                        i += 1

                mem_states = set()
                target_state = dist_states.pop()
                for i3 in range(len(ratio_range)): # iterate ratios
                    ratio = ratio_range[i3]

                    # create memory list
                    mem_states.clear()
                    # number of memory states
                    N_mem = int(np.ceil(ratio*N))-1
                    if N_mem >= N-len(dist_states)-1:
                        continue
                    # add mark to memory
                    mem_states.add(target_state)
                    mark_projector = qu.projection(N, target_state, target_state)
                    available_states = possible_states.copy()
                    for j in range(N_mem): # add random memories
                        num_ind = np.random.randint(0, len(available_states))
                        add_state = available_states.pop(num_ind)
                        mem_states.add(add_state)

                    # make query and memory list
                    query = [int_to_bitstr(mark, n)]
                    # query_weights = [peak_factor/q_N if i in marks else (1-peak_factor)/(N-q_N) for i in range(N)]
                    memory = [int_to_bitstr(mem, n) for mem in mem_states]
                    # mark_list = [int_to_bitstr(mark, n) for mark in marks]

                    # initialize QuAM
                    ventura = Ventura_QuAM.QuAM()
                    ventura.set_mem(memory)
                    ventura.set_query(query, bin_param=peak_factor) #bin_param=peaking_factor)
                    ventura.set_oracle()
                    ventura.set_diffusion()
                    ventura.set_max_iterations(1000)
                    print("memory:", memory)
                    print("query:", query)

                    # perform matching
                    _, ventura_states = ventura.match_ezhov(iteration=1000)

                    # gather probabilites
                    success_probs = [(mark_projector*state).norm()**2 for state in ventura_states]

                    max_prob = max(success_probs)
                    print("n  ratio    prob")
                    print(n, format(ratio, "0.3f"), max_prob)
                    # if max_prob < 0.9:
                    #     plt.plot(success_probs)
                    #     plt.show()
                    ventura_probs[i1,i2,i3, dist_ind] = max_prob

    # save results
    if data_file is None:
        data_file = "data"

    np.savez(data_file, bits=n_bits, ratios=ratio_range, peak_factor=np.array([peak_factor]), distances=distances, values=ventura_probs)

def Test7(data_file=None, method="ezhov"):
    """Test7

    Match with a distributed query of one target, iterating over number of qubits, filling of database, as well as averaging over random choices of marked state. only include states in memory, which are at least a hamming distance 3 away from mark. Include only one other state, with a given distance.

    """
    max_bits = 8 # maximum number of qubits
    n_bits = np.arange(4, max_bits+1)
    n_nums = len(n_bits)
    number_of_marks = 10 # number of random marks to test for
    max_dist = 3
    distances = np.arange(1, 4) # the distance at which to put the target memory state
    ratio_vals = 30 # number of ratio parameters to test for
    ratio_range = np.linspace(0.1, 0.9, ratio_vals)
    peak_factor = 0.45 # 0 is a delta peak at mark
    amp_ratio = 0.2

    data_struct = (n_nums, number_of_marks, ratio_vals, len(distances))
    ventura_probs = np.zeros(data_struct)
    ventura_prob_inds = np.zeros(data_struct)
    # trugenberg_probs = np.zeros((n_nums, number_of_marks, ratio_vals))
    for i1 in range(len(n_bits)): # iterate number of bits
        n = n_bits[i1]
        N = 2**n
        mark_num = min(N, number_of_marks) # number of marks to test
        for i2 in range(mark_num): # iterate number of marks
            mark = np.random.randint(0, N)
            for dist_ind in range(len(distances)):
                target_dist = distances[dist_ind]
                if target_dist >= N:
                    continue
                max_dist = target_dist
                possible_states = [i for i in range(N)]
                dist_states = set()
                i = 0
                while i < len(possible_states):
                    if hamm_dist(possible_states[i], mark) == target_dist:
                        dist_states.add(possible_states.pop(i))
                    elif hamm_dist(possible_states[i], mark) <= max_dist:
                        possible_states.pop(i)
                    else:
                        i += 1

                mem_states = set()
                target_state = dist_states.pop()
                for i3 in range(len(ratio_range)): # iterate ratios
                    ratio = ratio_range[i3]

                    # create memory list
                    mem_states.clear()
                    # number of memory states
                    available_states = possible_states.copy()
                    N_mem = int(np.ceil(ratio*len(available_states)))
                    # add mark to memory
                    mem_states.add(target_state)
                    mark_projector = qu.projection(N, target_state, target_state)
                    for j in range(N_mem): # add random memories
                        num_ind = np.random.randint(0, len(available_states))
                        add_state = available_states.pop(num_ind)
                        mem_states.add(add_state)

                    # make query and memory list
                    query = [int_to_bitstr(i, n) for i in range(N)]
                    weights = np.zeros(N)
                    for i in range(N):
                        dist = hamm_dist(i, mark)
                        if dist == max_dist:
                            weights[i] = np.sqrt((peak_factor)**dist * (1-peak_factor)**(n-dist))
                    weights[mark] = 0
                    n_zero = len(weights[weights == 0])
                    weights[weights == 0] = np.sqrt(np.sum(weights**2)/n_zero)*amp_ratio/(1-amp_ratio)

                    print(weights)

                    # query_weights = [peak_factor/q_N if i in marks else (1-peak_factor)/(N-q_N) for i in range(N)]
                    memory = [int_to_bitstr(mem, n) for mem in mem_states]
                    # mark_list = [int_to_bitstr(mark, n) for mark in marks]

                    # initialize QuAM

                    ventura = Ventura_QuAM.QuAM()
                    ventura.set_mem(memory)
                    ventura.set_query(query, weights=weights) #bin_param=peaking_factor)
                    ventura.set_oracle()
                    ventura.set_diffusion()
                    ventura.set_max_iterations(1000)
                    print("memory:", memory)
                    print("query:", format(mark, "0" + str(n) + "b"))

                    # perform matching
                    repeat_case = True
                    iteration_limit = int(N)
                    while repeat_case:
                        repeat_case = False
                        if method == "ezhov":
                            _, ventura_states = ventura.match_ezhov(iteration=iteration_limit)
                        if method == "improved":
                            _, ventura_states = ventura.match_C1(iteration=iteration_limit)

                        # gather probabilites
                        success_probs = [(mark_projector*state).norm()**2 for state in ventura_states]
                        max_prob_ind = np.argmax(np.array(success_probs))
                        max_prob = success_probs[max_prob_ind]
                        print("n  ratio   dist    prob     ind")
                        print(n, format(ratio, "0.3f"), target_dist, max_prob, max_prob_ind)
                        # if max_prob < 0.9:
                        #     plt.plot(success_probs)
                        #     plt.show()
                        ventura_probs[i1,i2,i3, dist_ind] = max_prob
                        ventura_prob_inds[i1,i2,i3, dist_ind] = max_prob_ind
                        if max_prob_ind >= iteration_limit or max_prob_ind <= 1:
                            iteration_limit += int(N)
                            print("repeating test case with limit", iteration_limit, len(success_probs))
                            repeat_case = True

    # save results
    if data_file is None:
        data_file = "data"

    np.savez(data_file, bits=n_bits, ratios=ratio_range, params=np.array([peak_factor, amp_ratio]), distances=distances, values=ventura_probs, inds=ventura_prob_inds)

def Test8(data_file=None, method="ezhov"):
    """Test8

    Match with a distributed query of one target, iterating over number of qubits, filling of database, as well as averaging over random choices of marked state. only include states in memory, which are at least a hamming distance 3 away from mark. Include only one other state, with a given distance.

    """
    max_bits = 7 # maximum number of qubits
    n_bits = np.arange(4, max_bits+1)
    n_nums = len(n_bits)
    number_of_marks = 10 # number of random marks to test for
    max_dist = 3
    distances = np.arange(1, max_dist+1) # the distance at which to put the target memory state
    ratio_vals = 10 # number of ratio parameters to test for
    ratio_range = np.linspace(0.1, 0.9, ratio_vals)
    peak_factor = 0.4 # 0 is a delta peak at mark
    amp_ratio = 0.1

    data_struct = (n_nums, number_of_marks, ratio_vals, len(distances))
    ventura_probs = np.zeros(data_struct)
    ventura_prob_inds = np.zeros(data_struct)
    # trugenberg_probs = np.zeros((n_nums, number_of_marks, ratio_vals))
    for i1 in range(len(n_bits)): # iterate number of bits
        n = n_bits[i1]
        N = 2**n
        mark_num = min(N, number_of_marks) # number of marks to test
        for i2 in range(mark_num): # iterate number of marks
            mark = np.random.randint(0, N)
            for dist_ind in range(len(distances)):
                target_dist = distances[dist_ind]
                if target_dist >= N:
                    continue
                possible_states = [i for i in range(N)]
                dist_states = set()
                mark_projector = qu.Qobj()
                i = 0
                while i < len(possible_states):
                    if hamm_dist(possible_states[i], mark) == target_dist:
                        dist_states.add(possible_states.pop(i))
                    elif hamm_dist(possible_states[i], mark) <= max_dist:
                        possible_states.pop(i)
                    else:
                        i += 1

                mem_states = set()
                target_state = dist_states.pop()
                mark_projector = qu.projection(N, target_state, target_state)
                for i3 in range(len(ratio_range)): # iterate ratios
                    ratio = ratio_range[i3]

                    # create memory list
                    mem_states.clear()
                    # number of memory states
                    available_states = possible_states.copy()
                    N_mem = int(np.ceil(ratio*len(available_states)))
                    # add mark to memory
                    mem_states.add(target_state)
                    for j in range(N_mem): # add random memories
                        num_ind = np.random.randint(0, len(available_states))
                        add_state = available_states.pop(num_ind)
                        mem_states.add(add_state)

                    # make query and memory list
                    query = [int_to_bitstr(i, n) for i in range(N)]
                    weights = np.zeros(N)
                    for i in range(N):
                        dist = hamm_dist(i, mark)
                        if dist <= max_dist:
                            weights[i] = np.sqrt((peak_factor)**dist * (1-peak_factor)**(n-dist))
                    weights[mark] = 0
                    n_zero = len(weights[weights == 0])
                    weights[weights == 0] = np.sqrt(np.sum(weights**2)*amp_ratio/(1-amp_ratio))/n_zero

                    print(weights)

                    # query_weights = [peak_factor/q_N if i in marks else (1-peak_factor)/(N-q_N) for i in range(N)]
                    memory = [int_to_bitstr(mem, n) for mem in mem_states]
                    # mark_list = [int_to_bitstr(mark, n) for mark in marks]

                    # initialize QuAM

                    ventura = Ventura_QuAM.QuAM()
                    ventura.set_mem(memory)
                    ventura.set_query(query, weights=weights) #bin_param=peaking_factor)
                    ventura.set_oracle()
                    ventura.set_diffusion()
                    ventura.set_max_iterations(1000)
                    print("memory:", memory)
                    print("query:", format(mark, "0" + str(n) + "b"))

                    # perform matching
                    repeat_case = True
                    iteration_limit = int(N)
                    while repeat_case:
                        repeat_case = False
                        if method == "ezhov":
                            _, ventura_states = ventura.match_ezhov(iteration=iteration_limit)
                        if method == "improved":
                            _, ventura_states = ventura.match_C1(iteration=iteration_limit)

                        # gather probabilites
                        success_probs = [(mark_projector*state).norm()**2 for state in ventura_states]
                        max_prob_ind = np.argmax(np.array(success_probs))
                        max_prob = success_probs[max_prob_ind]
                        print("n  ratio   dist    prob     ind")
                        print(n, format(ratio, "0.3f"), target_dist, max_prob, max_prob_ind)
                        # if max_prob < 0.9:
                        #     plt.plot(success_probs)
                        #     plt.show()
                        ventura_probs[i1,i2,i3, dist_ind] = max_prob
                        ventura_prob_inds[i1,i2,i3, dist_ind] = max_prob_ind
                        if max_prob_ind >= iteration_limit or max_prob_ind <= 0:
                            print("repeating test case")
                            iteration_limit += int(N)
                            repeat_case = True

    # save results
    if data_file is None:
        data_file = "data"

    np.savez(data_file, bits=n_bits, ratios=ratio_range, peak_factor=np.array([peak_factor]), amp_ratio=np.array([amp_ratio]), distances=distances, values=ventura_probs, inds=ventura_prob_inds)

def Test9(data_file=None, method="ezhov"):
    """Test9

    Match with a distributed query of one target, iterating over number of qubits, filling of database, as well as averaging over random choices of marked state. Include states in memory, which are at least a hamming distance 3 away from mark and also a certain percentage of states a specified distance from target.

    """
    max_bits = 9 # maximum number of qubits
    n_bits = np.arange(4, max_bits+1)
    n_nums = len(n_bits)
    number_of_marks = 10 # number of random marks to test for
    max_dist = 3
    distances = np.arange(1, max_dist+1) # the distance at which to put the target memory state
    ratio_vals = 4 # number of ratio parameters to test for
    mark_ratio_vals = 10
    ratio_range = np.linspace(0.1, 0.9, ratio_vals)
    mark_ratios = np.linspace(0.1, 0.9, mark_ratio_vals)
    peak_factor = 0.4 # 0 is a delta peak at mark
    amp_ratio = 0.1

    data_struct = (n_nums, number_of_marks, len(mark_ratios), ratio_vals, len(distances))
    ventura_probs = np.zeros(data_struct)
    ventura_prob_inds = np.zeros(data_struct)
    # trugenberg_probs = np.zeros((n_nums, number_of_marks, ratio_vals))
    for i1 in range(len(n_bits)): # iterate number of bits
        n = n_bits[i1]
        N = 2**n
        mark_num = min(N, number_of_marks) # number of marks to test
        for i2 in range(mark_num): # iterate number of marks
            mark = np.random.randint(0, N)
            for dist_ind in range(len(distances)):
                target_dist = distances[dist_ind]
                if target_dist >= N:
                    continue
                possible_states = [i for i in range(N)]
                dist_states = set()
                mark_projector = qu.Qobj()
                i = 0
                while i < len(possible_states):
                    if hamm_dist(possible_states[i], mark) == target_dist:
                        dist_states.add(possible_states.pop(i))
                    elif hamm_dist(possible_states[i], mark) <= max_dist:
                        possible_states.pop(i)
                    else:
                        i += 1

                mem_states = set()
                for mark_ratio_ind in range(len(mark_ratios)):
                    # prepare target states and projector
                    mark_ratio = mark_ratios[mark_ratio_ind]
                    target_states = set()
                    possible_targets = dist_states.copy()
                    target_num = int(len(possible_targets)*mark_ratio)
                    if target_num < 1:
                        target_num = 1
                    mark_projector = qu.Qobj()
                    for i in range(target_num):
                        state = possible_targets.pop()
                        target_states.add(state)
                        mark_projector += qu.projection(N, state, state)

                    for i3 in range(len(ratio_range)): # iterate ratios
                        ratio = ratio_range[i3]

                        # create memory list
                        mem_states = target_states.copy()
                        # number of memory states
                        available_states = possible_states.copy()
                        N_mem = int(np.ceil(ratio*len(available_states)))

                        for j in range(N_mem): # add random memories
                            num_ind = np.random.randint(0, len(available_states))
                            add_state = available_states.pop(num_ind)
                            mem_states.add(add_state)

                        # make query and memory list
                        query = [int_to_bitstr(i, n) for i in range(N)]
                        weights = np.zeros(N)
                        for i in range(N):
                            dist = hamm_dist(i, mark)
                            if dist <= max_dist:
                                weights[i] = np.sqrt((peak_factor)**dist * (1-peak_factor)**(n-dist))
                        weights[mark] = 0
                        n_zero = len(weights[weights == 0])
                        weights[weights == 0] = np.sqrt(np.sum(weights**2)*amp_ratio/(1-amp_ratio))/n_zero

                        # print(weights)

                        # query_weights = [peak_factor/q_N if i in marks else (1-peak_factor)/(N-q_N) for i in range(N)]
                        memory = [int_to_bitstr(mem, n) for mem in mem_states]
                        # mark_list = [int_to_bitstr(mark, n) for mark in marks]

                        # initialize QuAM

                        ventura = Ventura_QuAM.QuAM()
                        ventura.set_mem(memory)
                        ventura.set_query(query, weights=weights) #bin_param=peaking_factor)
                        ventura.set_oracle()
                        ventura.set_diffusion()
                        ventura.set_max_iterations(1000)
                        # print("memory:", memory)
                        print("query:", format(mark, "0" + str(n) + "b"))

                        # perform matching
                        repeat_case = True
                        iteration_limit = int(N)
                        while repeat_case:
                            repeat_case = False
                            if method == "ezhov":
                                _, ventura_states = ventura.match_ezhov(iteration=iteration_limit)
                            if method == "improved":
                                _, ventura_states = ventura.match_C1(iteration=iteration_limit)

                            # gather probabilites
                            success_probs = [(mark_projector*state).norm()**2 for state in ventura_states]
                            max_prob_ind = np.argmax(np.array(success_probs))
                            max_prob = success_probs[max_prob_ind]
                            print("n ratio mark_ratio dist prob ind")
                            print(n, format(ratio, "0.3f"), format(mark_ratio, "0.3f"), target_dist, format(max_prob, "0.4f"), max_prob_ind)
                            # if max_prob < 0.9:
                            #     plt.plot(success_probs)
                            #     plt.show()
                            ventura_probs[i1,i2,mark_ratio_ind,i3, dist_ind] = max_prob
                            ventura_prob_inds[i1,i2,mark_ratio_ind,i3, dist_ind] = max_prob_ind
                            if max_prob_ind >= iteration_limit or max_prob_ind <= 0:
                                print("repeating test case")
                                iteration_limit += int(N)
                                repeat_case = True

    # save results
    if data_file is None:
        data_file = "data"

    np.savez(data_file, bits=n_bits, ratios=ratio_range, mark_ratios=mark_ratios, peak_factor=np.array([peak_factor]), amp_ratio=np.array([amp_ratio]), distances=distances, values=ventura_probs, inds=ventura_prob_inds)

def Test10(data_file=None, method="ezhov"):
    """Test10

    Match with a distributed query of one target, iterating over number of qubits, filling of database, as well as averaging over random choices of marked state. Include states in memory, which are at least a hamming distance 3 away from mark and also a certain percentage of states within a specified distance from target.

    """
    max_bits = 9 # maximum number of qubits
    n_bits = np.arange(4, max_bits+1)
    n_nums = len(n_bits)
    number_of_marks = 10 # number of random marks to test for
    max_dist = 3
    distances = np.arange(1, max_dist+1) # the distance at which to put the target memory state
    ratio_vals = 4 # number of ratio parameters to test for
    mark_ratio_vals = 10
    ratio_range = np.linspace(0.1, 0.9, ratio_vals)
    mark_ratios = np.linspace(0.1, 0.9, mark_ratio_vals)
    peak_factor = 0.4 # 0 is a delta peak at mark
    amp_ratio = 0.1

    data_struct = (n_nums, number_of_marks, len(mark_ratios), ratio_vals, len(distances))
    ventura_probs = np.zeros(data_struct)
    ventura_prob_inds = np.zeros(data_struct)
    # trugenberg_probs = np.zeros((n_nums, number_of_marks, ratio_vals))
    for i1 in range(len(n_bits)): # iterate number of bits
        n = n_bits[i1]
        N = 2**n
        mark_num = min(N, number_of_marks) # number of marks to test
        for i2 in range(number_of_marks): # iterate number of marks
            mark = np.random.randint(0, N)
            for dist_ind in range(len(distances)):
                target_dist = distances[dist_ind]
                max_dist = target_dist
                if target_dist >= N:
                    continue
                possible_states = [i for i in range(N)]
                dist_states = set()
                mark_projector = qu.Qobj()
                i = 0
                while i < len(possible_states):
                    if hamm_dist(possible_states[i], mark) <= target_dist:
                        dist_states.add(possible_states.pop(i))
                    elif hamm_dist(possible_states[i], mark) <= max_dist:
                        possible_states.pop(i)
                    else:
                        i += 1

                mem_states = set()
                for mark_ratio_ind in range(len(mark_ratios)):
                    # prepare target states and projector
                    mark_ratio = mark_ratios[mark_ratio_ind]
                    target_states = set()
                    possible_targets = dist_states.copy()
                    target_num = int(len(possible_targets)*mark_ratio)
                    if target_num < 1:
                        target_num = 1
                    mark_projector = qu.Qobj()
                    for i in range(target_num):
                        state = possible_targets.pop()
                        target_states.add(state)
                        mark_projector += qu.projection(N, state, state)

                    for i3 in range(len(ratio_range)): # iterate ratios
                        ratio = ratio_range[i3]

                        # create memory list
                        mem_states = target_states.copy()
                        # number of memory states
                        available_states = possible_states.copy()
                        N_mem = int(np.ceil(ratio*len(available_states)))

                        for j in range(N_mem): # add random memories
                            num_ind = np.random.randint(0, len(available_states))
                            add_state = available_states.pop(num_ind)
                            mem_states.add(add_state)

                        # make query and memory list
                        query = [int_to_bitstr(i, n) for i in range(N)]
                        weights = np.zeros(N)
                        for i in range(N):
                            dist = hamm_dist(i, mark)
                            if dist <= max_dist:
                                weights[i] = np.sqrt((peak_factor)**dist * (1-peak_factor)**(n-dist))
                        weights[mark] = 0
                        n_zero = len(weights[weights == 0])
                        weights[weights == 0] = np.sqrt(np.sum(weights**2)*amp_ratio/(1-amp_ratio))/n_zero

                        # print(weights)

                        # query_weights = [peak_factor/q_N if i in marks else (1-peak_factor)/(N-q_N) for i in range(N)]
                        memory = [int_to_bitstr(mem, n) for mem in mem_states]
                        # mark_list = [int_to_bitstr(mark, n) for mark in marks]

                        # initialize QuAM

                        ventura = Ventura_QuAM.QuAM()
                        ventura.set_mem(memory)
                        ventura.set_query(query, weights=weights) #bin_param=peaking_factor)
                        ventura.set_oracle()
                        ventura.set_diffusion()
                        ventura.set_max_iterations(1000)
                        # print("memory:", memory)
                        print("query:", format(mark, "0" + str(n) + "b"))

                        # perform matching
                        repeat_case = True
                        iteration_cap = int(np.ceil(np.sqrt(N)))
                        iteration_limit = iteration_cap
                        while repeat_case:
                            repeat_case = False
                            if method == "ezhov":
                                _, ventura_states = ventura.match_ezhov(iteration=iteration_limit)
                            if method == "improved":
                                _, ventura_states = ventura.match_C1(iteration=iteration_limit)

                            # gather probabilites
                            success_probs = [(mark_projector*state).norm()**2 for state in ventura_states]
                            max_prob_ind = np.argmax(np.array(success_probs))
                            max_prob = success_probs[max_prob_ind]
                            print("n ratio mark_ratio dist prob ind")
                            print(n, format(ratio, "0.3f"), format(mark_ratio, "0.3f"), target_dist, format(max_prob, "0.4f"), max_prob_ind)
                            # if max_prob < 0.9:
                            #     plt.plot(success_probs)
                            #     plt.show()
                            ventura_probs[i1,i2,mark_ratio_ind,i3, dist_ind] = max_prob
                            ventura_prob_inds[i1,i2,mark_ratio_ind,i3, dist_ind] = max_prob_ind
                            if max_prob_ind >= iteration_limit or max_prob_ind <= 0:
                                print("repeating test case")
                                iteration_limit += iteration_cap
                                repeat_case = True

    # save results
    if data_file is None:
        data_file = "data"

    np.savez(data_file, bits=n_bits, ratios=ratio_range, mark_ratios=mark_ratios, peak_factor=np.array([peak_factor]), amp_ratio=np.array([amp_ratio]), distances=distances, values=ventura_probs, inds=ventura_prob_inds)

def plot_test1_data(data_file):
    data = np.load(data_file)

    bits = data["bits"]
    ratios = data["ratios"]
    param = data["peak_factor"]
    vals = data["values"]

    vals[vals == 0] = np.nan

    mean_vals = np.nanmean(vals, axis=1)
    std_vals = np.nanstd(vals, axis=1)

    fig = plt.figure()
    ax = fig.gca(projection="3d")
    X,Y = np.meshgrid(ratios, bits)
    Z = mean_vals

    ax.plot_surface(X, Y, Z)
    ax.set_zlim([0, 1])
    plt.ylabel("Bits (n)")
    plt.xlabel("Database Filling ratio (m/N)")
    plt.title("Probability of finding one marked (existing) item")
    plt.show()

def plot_test2_data(data_file):
    data = np.load(data_file)

    bits = data["bits"]
    ratios = data["ratios"]
    param = data["peak_factor"]
    vals = data["values"]

    vals[vals == 0] = np.nan

    mean_vals = np.nanmean(vals, axis=1)
    std_vals = np.nanstd(vals, axis=1)

    fig = plt.figure()
    ax = fig.gca(projection="3d")
    X,Y = np.meshgrid(ratios, bits)
    Z = mean_vals

    ax.plot_surface(X, Y, Z)
    ax.set_zlim([0, 1])
    plt.ylabel("Bits (n)")
    plt.xlabel("Database Filling ratio (m/N)")
    plt.title("Probability of finding one marked (existing) item")
    plt.show()

def plot_test3_data(data_file):
    data = np.load(data_file)

    bits = data["bits"]
    ratios = data["ratios"]
    param = data["peak_factor"]
    vals = data["values"]
    number_of_targets = data["target_nums"]

    vals[vals == 0] = np.nan

    mean_vals = np.nanmean(vals, axis=1)
    std_vals = np.nanstd(vals, axis=1)

    fig = plt.figure()
    ax = fig.gca(projection="3d")
    X,Y = np.meshgrid(ratios, bits)
    for i in reversed(range(len(number_of_targets))):
        Z = mean_vals[:,:,i]

        surf = ax.plot_surface(X, Y, Z, label="{} targets".format(number_of_targets[i]))
        surf._facecolors2d = surf._facecolors3d
        surf._edgecolors2d = surf._edgecolors3d
    ax.set_zlim([0, 1])
    ax.legend()
    plt.ylabel("Bits (n)")
    plt.xlabel("Database Filling ratio (m/N)")
    plt.title("Probability of finding several marked (existing) item")
    plt.show()

def plot_test4_data(data_file):
    data = np.load(data_file)

    bits = data["bits"]
    ratios = data["ratios"]
    param = data["peak_factor"]
    vals = data["values"]
    number_of_targets = data["target_nums"]

    vals[vals == 0] = np.nan

    mean_vals = np.nanmean(vals, axis=1)
    std_vals = np.nanstd(vals, axis=1)

    fig = plt.figure()
    ax = fig.gca(projection="3d")
    X,Y = np.meshgrid(ratios, bits)
    for i in reversed(range(len(number_of_targets))):
        Z = mean_vals[:,:,i]

        surf = ax.plot_surface(X, Y, Z, label="{} targets".format(number_of_targets[i]))
        surf._facecolors2d = surf._facecolors3d
        surf._edgecolors2d = surf._edgecolors3d
    ax.set_zlim([0, 1])
    ax.legend()
    plt.ylabel("Bits (n)")
    plt.xlabel("Database Filling ratio (m/N)")
    plt.title("Probability of finding several marked (but only one existing) item")
    plt.show()

def plot_test8_data(data_file):
    data = np.load(data_file)

    bits = data["bits"]
    ratios = data["ratios"]
    param = data["peak_factor"]
    vals = data["values"]
    inds = data["inds"]
    distances = data["distances"]

    vals[vals == 0] = np.nan

    mean_vals = np.nanmean(vals, axis=1)
    std_vals = np.nanstd(vals, axis=1)

    fig = plt.figure()
    ax = fig.gca(projection="3d")
    X,Y = np.meshgrid(ratios, bits)
    for i in reversed(range(len(distances))):
        Z = mean_vals[:,:,i]

        surf = ax.plot_surface(X, Y, Z, label="distance {} targets".format(distances[i]))
        surf._facecolors2d = surf._facecolors3d
        surf._edgecolors2d = surf._edgecolors3d
    ax.set_zlim([0, 1])
    ax.legend()
    plt.ylabel("Bits (n)")
    plt.xlabel("Database Filling ratio (m/N)")
    plt.title("Probability of item at distance from target (only one such item in memory)")
    plt.show()

def plot_test9_data(data_file):
    data = np.load(data_file)

    bits = data["bits"]
    ratios = data["ratios"]
    mark_ratios = data["mark_ratios"]
    param = data["peak_factor"]
    vals = data["values"]
    inds = data["inds"]
    distances = data["distances"]

    vals[vals == 0] = np.nan

    mean_vals = np.nanmean(vals, axis=1)
    std_vals = np.nanstd(vals, axis=1)

    # project out the ratio parameter
    mean_vals = np.mean(mean_vals, axis=2)
    std_vals = np.mean(std_vals, axis=2)

    print("vals shape", mean_vals.shape)

    for i in reversed(range(len(distances))):
        fig = plt.figure()
        ax = fig.gca(projection="3d")
        X,Y = np.meshgrid(mark_ratios, bits)
        Z = mean_vals[:,:,i]

        surf = ax.plot_surface(X, Y, Z, label="distance {} targets".format(distances[i]))
        surf._facecolors2d = surf._facecolors3d
        surf._edgecolors2d = surf._edgecolors3d
        ax.set_zlim([0, 1])
        ax.legend()
        plt.ylabel("Bits (n)")
        plt.xlabel("Target filling ratio")
        plt.title("Probability of items a distance from target")
    plt.show()

def plot_test10_data(data_file):
    data = np.load(data_file)

    bits = data["bits"]
    ratios = data["ratios"]
    mark_ratios = data["mark_ratios"]
    param = data["peak_factor"]
    vals = data["values"]
    inds = data["inds"]
    distances = data["distances"]

    vals[vals == 0] = np.nan

    mean_vals = np.nanmean(vals, axis=1)
    std_vals = np.nanstd(vals, axis=1)

    # project out the ratio parameter
    mean_vals = np.mean(mean_vals, axis=2)
    std_vals = np.mean(std_vals, axis=2)

    print("vals shape", mean_vals.shape)

    for i in reversed(range(len(distances))):
        fig = plt.figure()
        ax = fig.gca(projection="3d")
        X,Y = np.meshgrid(mark_ratios, bits)
        Z = mean_vals[:,:,i]

        surf = ax.plot_surface(X, Y, Z, label="distance {} targets".format(distances[i]))
        surf._facecolors2d = surf._facecolors3d
        surf._edgecolors2d = surf._edgecolors3d
        ax.set_zlim([0, 1])
        ax.legend()
        plt.ylabel("Bits (n)")
        plt.xlabel("Target filling ratio")
        plt.title("Probability of items a distance from target")
    plt.show()

# Test8("test8_data", method="improved")

plot_test10_data("test10_data_C1.npz")
