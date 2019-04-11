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

def Test1(data_file=None):
    """Test1

    Match with a spiked query, iterating over number of qubits, filling of database, as well as averaging over random choices of marked state.

    """
    max_bits = 13 # maximum number of qubits
    n_bits = np.arange(3, max_bits+1)
    n_nums = len(n_bits)
    number_of_marks = 20 # number of random marks to test for
    ratio_vals = 20 # number of ratio parameters to test for
    ratio_range = np.linspace(0.1, 0.9, ratio_vals)
    peak_factor = 0.9 # 0 is a delta peak at mark

    ventura_probs = np.zeros((n_nums, number_of_marks, ratio_vals))
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
                _, ventura_states = ventura.match_ezhov(iteration=1500)

                # gather probabilites
                success_probs = [abs(state.full()[mark])**2 for state in ventura_states]
                max_prob = max(success_probs)
                print("n  ratio    prob")
                print(n, format(ratio, "0.3f"), max_prob)
                # if max_prob < 0.9:
                #     plt.plot(success_probs)
                #     plt.show()
                ventura_probs[i1,i2,i3] = max_prob

    # save results
    if data_file is None:
        data_file = "data"

    np.savez(data_file, bits=n_bits, ratios=ratio_range, peak_factor=np.array([peak_factor]), values=ventura_probs)

def Test2(data_file=None):
    max_bits = 7 # maximum number of qubits
    n_bits = np.arange(3, max_bits+1)
    n_nums = len(n_bits)
    number_of_marks = 20 # number of random marks to test for
    ratio_vals = 20 # number of ratio parameters to test for
    ratio_range = np.linspace(0.1, 0.9, ratio_vals)
    peak_factor = 0.25 # 0 is a delta peak at mark

    ventura_probs = np.zeros((n_nums, number_of_marks, ratio_vals))
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
                _, ventura_states = ventura.match_ezhov(iteration=1000)

                # gather probabilites
                success_probs = [abs(state.full()[mark])**2 for state in ventura_states]
                max_prob = max(success_probs)
                print("n  ratio    prob")
                print(n, format(ratio, "0.3f"), max_prob)
                # if max_prob < 0.9:
                #     plt.plot(success_probs)
                #     plt.show()
                ventura_probs[i1,i2,i3] = max_prob

    # save results
    if data_file is None:
        data_file = "data"

    np.savez(data_file, bits=n_bits, ratios=ratio_range, peak_factor=np.array([peak_factor]), values=ventura_probs)

def Test3(data_file=None):
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

# Test3("test3_data")

plot_test3_data("test3_data.npz")
