import numpy as np
import qutip as qu

def bit_arr_to_int(bit_arr):
    return np.sum(pattern*np.pow(2, np.arange(len(pattern)-1, -1, -1)))

def find_closest_int(func, limit=1000):
    vals = func(np.arange(limit))
    vals = np.abs(np.mod(vals, 1) - 0.5)
    return np.argmax(vals)

class QuAM:

    def __init__(self):
        pass

    def hamm_bin(center, params):
        a = params[0]
        def hamm_dist(pattern, ref):
            return np.sum(np.logical_not(np.logical_xor(pattern, ref)))
        N = len(center)
        perm_numbers = np.arange(0, 2**N)
        perms = [format(s, "0" + str(N) + "b") for s in range(len(perm_numbers))]
        perms = np.array([np.array([x == "1" for x in s]) for s in perms])
        weights = np.zeros(len(perms))
        for i in range(len(perms)):
            dist = hamm_dist(perms[i], center)
            weights[i] = (a**dist)*(1-a)**(N-dist)
        for i in range(len(perms)):
            if i == 0:
                state = weights[i]*qu.basis(perm_numbers[i], len(perm_numbers))
            else:
                state += weights[i]*qu.basis(perm_numbers[i], len(perm_numbers))
        return state.unit()

    def set_query(self, patterns, weights=None, scheme=None):
        """Sets the pattern of the search

        Unless weights or different scheme is given, uses a Hamming distance binomial distribution for the distributed pattern query

        patterns - list of binary array for the patterns sought after : list of bool array
        weights - (relative) probability weights for the patterns in the query : numpy array of floats
        scheme - Function for making query distribution : function handle with 1 argument, the pattern
        """

        if not weights:
            if not scheme:
                scheme = lambda pattern: hamm_bin(pattern, 0.5)
            state = qu.Qobj()
            for pattern in patterns:
                state += scheme(pattern)
        else:
            state = qu.Qobj()
            max_ID = 2**len(patterns[0])
            for (pattern, weight) in zip(patterns, weights):
                state_id = bit_arr_to_int(pattern)
                state += weight*qu.basis(state_id, max_ID)
        state = state.unit()
        self.query = state

    def excl_mem(pattern, state):
        N = 2**len(pattern)
        if not state:
            state = sum([qu.basis(i, N) for i in range(N)])
            state = state.unit()
        pattern_state = qu.basis(bit_arr_to_int(pattern), N)
        state -= pattern_state.overlap(state)*pattern_state
        return state

    def incl_mem(pattern, state):
        N = 2**len(pattern)
        if not state:
            state = Qobj()
        state += qu.basis(bit_arr_to_int(pattern), N)
        return state

    def set_mem(self, memories, scheme=None):
        """Sets the memory of the search

        Takes a list of states to remember, and a scheme of how to input them into the memory (standards: 'inclusion', 'exclusion', 'vary_phase')

        memories - list of binary arrays to memorize : list of bool numpy arrays
        scheme - the memorization scheme (standard: exclusion) : string or function handle taking a pattern and current memory state
        """
        if not scheme:
            scheme = "exclusion"
        if isinstance(scheme, str):
            if scheme == "inclusion":
                scheme = incl_mem
            elif scheme == "exclusion":
                scheme = excl_mem
            elif scheme == "vary_phase":
                # TODO
                pass
        elif not isinstance(scheme, types.FunctionType):
            raise Exception("Memorization scheme is not valid")

        state = None
        for mem in memories:
            state = scheme(meme, state)
        self.memory = state.unit()

    def find_match(self, iteration="approx"):
        n = len(self.pattern.dims[0])
        N = 2**N
        # prepare operators
        oracle = qu.identity(N) - 2*self.pattern.ket2dm()
        diffuse = 2*self.memory.ket2dm() - qu.identity(N)
        # figure out
        if iteration == "approx":
            B = np.sum(self.pattern.data)

        else:
            B = self.pattern.overlap(self.memory)
        w = 2*np.asin(B)
        T = 2*np.pi/w
        M = find_closest_int(lambda x : T*(1/4 + x))
        
