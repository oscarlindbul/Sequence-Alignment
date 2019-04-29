import numpy as np

from qiskit.quantum_info import Pauli
from qiskit_aqua import Operator

from collections import OrderedDict

def get_MSA_qubitmat(sizes, weights, gap_pen=0, extra_inserts=0, allow_delete=False, coeffs=None):
    """Generate Hamiltonian for Multiple Sequence Alignment (MSA) column formulation, as minimization problem.

    Args:
        sizes - size of each sequence : 1D np.array
        weights - weight/cost matrix between elements in sequences (s1,n1,s2,n2) : 4D np.array
        gap_pen - gap penaly : float
        extra_inserts - number of extra inserts to allow : int
        allow_delete - Flag for allowing deletion of elements in problem : Bool

    Returns:
        operator.Operator : operator for the Hamiltonian and a constant float : shift for the obj function

    Goals:
        1 Minimize cost function
        2 Place every element somewhere
        3 Respect element order
    """

    L = len(sizes) # number of sequences
    n_tot = np.sum(sizes)
    N = np.max(sizes) # max number of elements in sequences
    num_pos = N + extra_inserts + 1*allow_delete # number of positions

    """Number of spins x_{s,n,i}
    s in [1, L]
    n in [1, sizes(s)]
    i in [1, N + extra + {1 if deletion}]
    number of spins = n_tot*positions
    """
    num_spins = n_tot*num_pos
    pauli_list = []
    shift = 0

    spin_mat = np.zeros((num_spins, num_spins))

    if coeffs is None:
        match_max = np.max(np.max(np.max(np.max(weights))))
        match_max = max(match_max, gap_pen)
        """
        maximum penalty given by (maximum row size)*sum_0_to_n(i*(i-1))*max_cost
        sum0_to_n(i*(i-1)) = [n(n+1)(2n+1)/6] - [n(n+1)/2] = [n(n+1)/2][(2n+1)/3 - 1] = n(n+1)(n-1)/3 = n(n^2-1)/3
        """
        sum_0_to_n = lambda x: x*(x**2-1)/3
        max_penalty = (np.max(sizes) + extra_inserts)*sum_0_to_n(len(sizes))*match_max
        coeffs = [1, max_penalty + 1]

    A = coeffs[0]    # cost function coefficient
    B = coeffs[1]    # placement coefficient
    C = coeffs[1]    # order coefficient

    def pos2ind(s, n, i):
        """Return spin index from sequence, element and position indices
        Index scheme: first N*L spins are 'removal' spins
                      the following L*N*num_pos spins are location spins
        """
        return int((np.sum(sizes)*L)*allow_delete \
                + (np.sum(sizes[:s]) + n)*num_pos + i)

    rev_ind_scheme = np.empty(num_spins, dtype=tuple)
    for s in range(L):
        for n in range(sizes[s]):
            for i in range(num_pos):
                rev_ind_scheme[pos2ind(s,n,i)] = (s, n, i)

    def to_bool_arr(arr):
        length = int(np.ceil(np.log2(np.max(arr))))
        bool_arr = [0]*len(arr)
        for i in range(len(arr)):
            bin_string = format(arr[i], "0" + str(length) + "b")
            bool_arr[i] = np.array([x == '1' for x in bin_string], dtype=np.bool)
        return bool_arr

    def add_pauli_bool(coeff, *inds):
        nonlocal shift
        nonlocal spin_mat
        inds_arr = np.zeros(len(inds), dtype=np.int32)
        if len(inds) > 2 or len(inds) <= 0:
            raise ValueError("Invalid number of indices, must be on QUBO form")

        for i in range(len(inds)):
            s,n,j = inds[i]
            inds_arr[i] = pos2ind(s, n, j)

        if len(inds_arr) == 1:
            spin_mat[inds_arr[0], inds_arr[0]] += coeff
        elif len(inds_arr) >= 2:
            inds_arr = np.sort(inds_arr)
            spin_mat[tuple(inds_arr)] += coeff

    """Cost function (Goal 1)
    Matching at same position
    H_matching = A*sum_{s1,s2} sum_{n1,n2} sum_i w_{s1,s2,n1,n2} * x_{s1,n1,i}*x_{s2,n2,i}
    """
    for s1 in range(L):
        for s2 in range(s1+1,L):
            for n1 in range(sizes[s1]):
                for n2 in range(sizes[s2]):
                    for i in range(num_pos):
                        # matching cost
                        w = weights[s1,n1,s2,n2]
                        add_pauli_bool(A*w, (s1,n1,i), (s2,n2,i))

    print("after matchings")
    print(spin_mat)
    """Penalties version 1 (penalty for number of gaps/deletions)
    Deletion for element
    H_del = A*sum_{s,n} x_{s,n,0}
    Insertion of gap
    H_insert = A*sum_s sum_{n1>n2} sum_{i>j} (i-j-(n1-n2))x_{s,n1,j}x_{s,n2,i}
    """
    # for s in range(L):
    #     for n1 in range(sizes[s]):
    #         for n2 in range(n1):
    #             for i in range(num_pos):
    #                 for j in range(i):
    #                     # insertion penalty
    #                     distance = i-j - (n1-n2)
    #                     w = A*gap_pen*distance
    #                     add_pauli_bool(w, (s,n1,i), (s,n2,j))
    #         # deletion penalty
    #         ind = np.sum(sizes[:s])) + n1
    #         w = del_pen
    #         add_pauli_bool(A*w, ind)

    """Penalties version 2 (pair of sum penalties)
    Pairing with gaps
    H_gap = A*sum_{s1,n1}sum_{s2}sum_i g*x_{s1,n1,i}(1 - sum_n2 x_{s2,n2,i})
    Represents pairing of (s1,n1) at i to nothing in s2
    """
    if gap_pen != 0:
        for s1 in range(L):
            for n1 in range(sizes[s1]):
                for s2 in range(s1+1, L):
                    for i in range(num_pos):
                        w = A*gap_pen

                        add_pauli_bool(w, (s1, n1, i))
                        for n2 in range(sizes[s2]):
                            add_pauli_bool(-w, (s1, n1, i), (s2, n2, i))
    print("After gap pentaly")
    print(spin_mat)

    """Placement terms
    Place at only one position
    H_placement = B*sum_{s,n} (1-sum_i x_{s,n,i})^2
    = B*n_tot - 2*B*sum_{s,n}sum_i x_{s,n,i} \
    + sum_{s,n} sum_{i,j} x_{s,n,i}x_{s,n,j}
    = B*n_tot - B*sum_{s,n}sum_i x_{s,n,i} + B*sum_{s,n}sum_{i!=j} x_{s,n,i}x_{s,n,j}
    """
    shift += B*n_tot
    for s in range(L):
        for n in range(sizes[s]):
            for i in range(num_pos):
                for j in range(num_pos):
                    if i != j:
                        w = B
                        add_pauli_bool(w, (s,n,i), (s,n,j))
                    else:
                        w = -B
                        print((s,n,i), pos2ind(s, n, i), w)
                        add_pauli_bool(w,(s,n,i))

    print("After placement constrictment")
    print(spin_mat)

    """Order terms
    no deletions
    H_order_no_del = C*sum_{s,n} sum_{i<=j} x_{s,n,j}x_{s,n+1,i}
    with deletions
    H_order = C*sum_s sum_{n1<n2}sum_{i<=j} x_{s,n1,j}x_{s,n2,i}
    """
    for s in range(L):
        if allow_delete:
            for n1 in range(sizes[s]):
                for n2 in range(n1):
                    for i in range(num_pos):
                        for j in range(i+1):
                            w = C
                            add_pauli_bool(w,(s,n1,i),(s,n2,j))
        else:
            for n in range(sizes[s]-1):
                for i in range(num_pos):
                    for j in range(i+1):
                        w = C
                        add_pauli_bool(w, (s,n,i), (s,n+1,j))
                        print((s,n,i), (s,n+1,j))
    # exceptions = [(0,0,0), (0, 0, 1)]
    # add_pauli_bool(-50000, *exceptions)
    # for s in range(L):
    #     for n in range(sizes[s]):
    #         for i in range(num_pos):
    #             # if not (s,n,i) in exceptions:
    #             add_pauli_bool(1000, (s,n,i))
    #             print("not", (s,n,i))
    print("Number of paulis:", len(pauli_list))
    return spin_mat, shift, rev_ind_scheme

def sample_most_likely(state_vector, rev_inds):
    """Compute the most likely DNA matching"""

    if isinstance(state_vector, dict) or isinstance(state_vector, OrderedDict):
        binary_string = sorted(state_vector.items(), key=lambda kv: kv[1])[-1][0]
        x = np.asarray([int(y)==1 for y in reversed(list(binary_string))])
    else:
        n = int(np.log2(state_vector.shape[0]))
        k = np.argmax(np.flip(np.abs(state_vector)))
        x = np.zeros(n, dtype=np.bool)
        for i in range(n):
            x[i] = (k % 2) == 1
            k >>= 1

    # parse output binary string
    positions = {}
    included_comps = rev_inds[x]
    for (s, n, i) in included_comps:
        positions[(s, n)] = i
    return positions

def get_alignment_string(sequences, gaps, positions):
    string_size = max([len(s) for s in sequences]) + gaps
    align_strings = [["-" for i in range(string_size)] for i in range(len(sequences))]
    for (key, value) in positions.items():
        align_strings[key[0]][value] = sequences[key[0]][key[1]]
    return align_strings

def get_match_matrix(sequences, costs):
    """Constructs the matching matrix for later construction of Hamiltonian

    Constructs matching matrix for each (sequence, element) pair with given match reward and penalty

    Input:
        sequences - list of sequence strings : string list
        costs     - reward and penalty for element match/mismatch [reward, penalty] : float list
    Output:
        matchings - list on the format (s1, n1, s2, n2) of match scores : numpy array
    """
    reward = costs[0]
    penalty = costs[1]

    sizes = [len(sequences[i]) for i in range(len(sequences))]
    # calculate weights for matching
    matchings = np.zeros((len(sequences), max(sizes), len(sequences), max(sizes)))
    for s1 in range(len(sequences)):
        for s2 in range(len(sequences)):
            if s1 != s2:
                for n1 in range(sizes[s1]):
                    for n2 in range(sizes[s2]):
                        if sequences[s1][n1] == sequences[s2][n2]:
                            matchings[s1,n1,s2,n2] = reward
                        else:
                            matchings[s1,n1,s2,n2] = penalty
    return matchings
