import numpy as np

from qiskit.quantum_info import Pauli
from qiskit_aqua import Operator

def get_MSA_qubitops(sizes, weights, gap_pen=0, extra_inserts=0, allow_delete=False, coeffs=[1, 1000]):
    """Generate Hamiltonian for Multiple Sequence Alignment (MSA) column formulation

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
        nonlocal pauli_list
        inds_arr = np.zeros(len(inds), dtype=np.int32)
        for i in range(len(inds)):
            s,n,j = inds[i]
            inds_arr[i] = pos2ind(s, n, j)
        bool_list = to_bool_arr(np.arange(1, 2**len(inds)))
        factor = (1/2)**len(inds)
        for bool_arr in bool_list:
            xp = np.zeros(num_spins, dtype=np.bool)
            zp = np.zeros(num_spins, dtype=np.bool)
            zp[inds_arr[bool_arr]] = True
            pauli_list.append([coeff*factor, Pauli(zp, xp)])
        shift += coeff*factor

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
    H_gap = A*sum_{s1,n1}sum_{s2}sum_i g*x_{s1,n1,i}(sum_n2 x_{s2,n2,i} - 1)
    Represents pairing of (s1,n1) at i to nothing in s2
    """
    if gap_pen != 0:
        for s1 in range(L):
            for n1 in range(sizes[s1]):
                for s2 in range(s1+1, L):
                    for i in range(num_pos):
                        w = -A*gap_pen

                        add_pauli_bool(w, (s1, n1, i))
                        for n2 in range(sizes[s2]):
                            add_pauli_bool(w, (s1, n1, i), (s2, n2, i))

    """Placement terms
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
                        add_pauli_bool(w,(s,n,i))

    """Order terms
    no deletions
    H_order_no_del = C*sum_{s,n} sum_{i>j} x_{s,n,j}x_{s,n+1,i}
    with deletions
    H_order = C*sum_s sum_{n1>n2}sum_{i>j} x_{s,n1,j}x_{s,n2,i}
    """
    for s in range(L):
        if allow_delete:
            for n1 in range(sizes[s]):
                for n2 in range(n1):
                    for i in range(num_pos):
                        for j in range(i):
                            w = C
                            add_pauli_bool(w,(s,n1,j),(s,n2,i))
        else:
            for n in range(sizes[s]-1):
                for i in range(num_pos):
                    for j in range(i+1):
                        w = C
                        add_pauli_bool(w, (s,n,i), (s,n+1,j))
    # exceptions = [(0,0,0), (0, 0, 1)]
    # add_pauli_bool(-50000, *exceptions)
    # for s in range(L):
    #     for n in range(sizes[s]):
    #         for i in range(num_pos):
    #             # if not (s,n,i) in exceptions:
    #             add_pauli_bool(1000, (s,n,i))
    #             print("not", (s,n,i))
    print("Number of paulis:", len(pauli_list))
    return Operator(paulis=pauli_list), shift, rev_ind_scheme
