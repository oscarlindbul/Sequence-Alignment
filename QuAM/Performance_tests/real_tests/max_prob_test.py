import sys
sys.path.append("../../Ventura")

import Ventura_QuAM

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

    8-10: vicinity tests checking different ratios of matches at specific hamming distances from target.

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

def vicinity_ideal_test(data_file=None, method="ezhov"):
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
    target_failure_probs = np.zeros(data_struct)
    memory_failure_probs = np.zeros(data_struct)

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

                        # create memory projector
                        mem_projector = qu.Qobj()
                        for mem in mem_states:
                            mem_projector += qu.projection(N, mem, mem)
                        # Create complement (spurious) projector to memory
                        non_mem_projector = qu.identity(N) - mem_projector
                        non_target_mem_projector = mem_projector - mark_projector

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
                        iteration_cap = int(np.ceil(np.sqrt(len(mem_states))))
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
                            good_failure_prob = (non_target_mem_projector*ventura_states[max_prob_ind]).norm()**2
                            bad_failure_prob = (non_mem_projector*ventura_states[max_prob_ind]).norm()**2
                            print("n ratio mark_ratio dist prob fail_prob bad_fail ind")
                            print(n, format(ratio, "0.3f"), format(mark_ratio, "0.3f"), target_dist, format(max_prob, "0.4f"), format(good_failure_prob, "0.4f"), format(bad_failure_prob, "0.4f"), max_prob_ind)
                            # if max_prob < 0.9:
                            #     plt.plot(success_probs)
                            #     plt.show()
                            ventura_probs[i1,i2,mark_ratio_ind,i3, dist_ind] = max_prob
                            target_failure_probs[i1,i2,mark_ratio_ind,i3,dist_ind] = good_failure_prob
                            memory_failure_probs[i1,i2,mark_ratio_ind,i3,dist_ind] = bad_failure_prob
                            ventura_prob_inds[i1,i2,mark_ratio_ind,i3, dist_ind] = max_prob_ind

                            if max_prob_ind >= iteration_limit or max_prob_ind <= 0:
                                print("repeating test case")
                                iteration_limit += iteration_cap
                                repeat_case = True

    # save results
    if data_file is None:
        data_file = "data"

    np.savez(data_file, bits=n_bits, ratios=ratio_range, mark_ratios=mark_ratios, peak_factor=np.array([peak_factor]), amp_ratio=np.array([amp_ratio]), distances=distances, values=ventura_probs, good_failure=target_failure_probs,bad_failure=memory_failure_probs, inds=ventura_prob_inds)

def plot_vicinity_data(data_file):
    data = np.load(data_file)

    bits = data["bits"]
    ratios = data["ratios"]
    mark_ratios = data["mark_ratios"]
    param = data["peak_factor"]
    vals = data["values"]
    inds = data["inds"]
    distances = data["distances"]
    failure_vals = data["bad_failure"]

    vals[vals == 0] = np.nan

    mean_vals = np.nanmean(vals, axis=1)
    std_vals = np.nanstd(vals, axis=1)

    failure_vals[failure_vals == 0] = np.nan

    failure_vals_means = np.nanmean(failure_vals, axis=1)

    # project out the ratio parameter
    mean_vals = np.mean(mean_vals, axis=2)
    failure_vals_means = np.mean(failure_vals_means, axis=2)

    print("vals shape", mean_vals.shape)

    for i in range(len(distances)):
        fig = plt.figure()
        ax = fig.gca(projection="3d")
        X,Y = np.meshgrid(mark_ratios, bits)
        Z = mean_vals[:,:,i]

        surf = ax.plot_surface(X, Y, Z, label="distance {} targets".format(distances[i]))
        surf._facecolors2d = surf._facecolors3d
        surf._edgecolors2d = surf._edgecolors3d
        ax.set_zlim([0, 1])
        # ax.legend()
        plt.ylabel("Bits (n)")
        plt.xlabel("Target filling ratio")
        plt.title("Probability of items up to distance {} from target".format(distances[i]))

    for i in range(len(distances)):
        fig = plt.figure()
        ax = fig.gca(projection="3d")
        X,Y = np.meshgrid(mark_ratios, bits)
        Z = failure_vals_means[:,:,i]

        surf = ax.plot_surface(X, Y, Z, label="distance {} targets".format(distances[i]))
        surf._facecolors2d = surf._facecolors3d
        surf._edgecolors2d = surf._edgecolors3d
        ax.set_zlim([0, 1])
        # ax.legend()
        plt.ylabel("Bits (n)")
        plt.xlabel("Target filling ratio")
        plt.title("Bad failure probability at vicinity from target")
    plt.show()

name = "vicinity_test_improved"

# vicinity_ideal_test(name, method="improved")

plot_vicinity_data(name + ".npz")
