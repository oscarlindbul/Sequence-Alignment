import unittest as test
import numpy as np
import qutip as qu
from QuAM import *


class Test_QuAM(test.TestCase):

    def iterate_equals(self, arg1, arg2):
        if isinstance(arg1, np.ndarray) and isinstance(arg2, np.ndarray):
            self.assertTupleEqual(arg1.shape, arg2.shape)

            if len(arg1.shape) == 1:
                for i in range(arg1.shape[0]):
                    self.assertAlmostEqual(arg1[i], arg2[i])
            elif len(arg1.shape) == 2:
                for i in range(arg1.shape[0]):
                    for j in range(arg1.shape[1]):
                        self.assertAlmostEqual(arg1[i,j], arg2[i,j])

    def test_distribution_construction(self):
        n = 3
        a = 0.1
        c = 0.9**n
        center = np.array([False, True, True])
        numbers = np.arange(0, 2**n)
        center_num = np.sum(np.power(2, np.arange(n))*center)
        dists = np.array([2, 1, 1, 0, 3, 2, 2, 1])
        amps = np.sqrt(c/9**dists)

        distribution = np.reshape(hamm_bin(center, a).full(), (2**n))
        self.iterate_equals(amps, distribution)

    def test_query_construction(self):
        n = 3
        a1 = 0.25
        c1 = (3/4)**n
        a2 = 0.1
        c2 = (9/10)**n
        numbers = np.arange(0, 2**n)

        pattern1 = "011"
        center1 = np.array([s == "1" for s in pattern1])
        center_num1 = np.sum(np.power(2, np.arange(n))*center1)
        dists1 = np.array([2, 1, 1, 0, 3, 2, 2, 1])
        amps1 = np.sqrt(c1/3**dists1)

        pattern2 = "101"
        center2 = np.array([s == "1" for s in pattern2])
        center_num2 = np.sum(np.power(2, np.arange(n))*center2)
        dists2 = np.array([2, 1, 3, 2, 1, 0, 2, 1])
        amps2 = np.sqrt(c2/9**dists2)

        quam = QuAM()
        quam.set_query([pattern1], bin_param=a1)
        dist = np.reshape(quam.query.full(), (2**n))
        self.iterate_equals(amps1, dist)
        quam.set_query([pattern2], bin_param=a2)
        dist = np.reshape(quam.query.full(), (2**n))
        self.iterate_equals(amps2, dist)

        number_strings = [format(m, "0" + str(n) + "b") for m in numbers]
        quam.set_query(number_strings, amps1)
        dist = np.reshape(quam.query.full(), (2**n))
        self.iterate_equals(dist, amps1)
        new_strings = [number_strings[0], number_strings[3]]
        new_amps = np.array([amps1[0], amps2[3]])
        quam.set_query(new_strings, new_amps)
        new_amps = np.array([amps1[0], 0, 0, amps2[3], 0, 0, 0, 0])
        new_amps /= np.linalg.norm(new_amps)
        dist = np.reshape(quam.query.full(), (2**n))
        self.iterate_equals(new_amps, dist)

        def test_scheme(pattern):
            number = 0
            N = 2**len(pattern)
            const = 2**(len(pattern)-1)
            for i in range(len(pattern)):
                number += const*pattern[i]
                const /= 2
            number = int(number)
            return qu.basis(N, number) if number % 2 == 0 else 0*qu.basis(N, number)
        new_amps = np.array([1, 0, 1, 0, 1, 0, 1, 0])
        new_amps = new_amps/np.linalg.norm(new_amps)
        quam.set_query(patterns=number_strings, scheme=test_scheme)
        dist = np.reshape(quam.query.full(), (2**n))
        self.iterate_equals(dist, new_amps)

    def test_oracle_construction(self):
        center = "011"
        n = len(center)
        quam = QuAM()
        number = sum([(center[i] == "1")*2**(n-i-1) for i in range(n)])
        quam.set_query([center])

        key = np.diag(np.ones(2**n))
        key[number,number] = -1

        quam.set_oracle(qu.Qobj(key))
        oracle = quam.oracle.full()
        self.iterate_equals(oracle, key)

        oracle_state = qu.basis(2**n, number)
        quam.set_oracle(oracle_state)
        oracle = quam.oracle.full()
        self.iterate_equals(oracle, key)

        key = hamm_bin(np.array([s == "1" for s in center]), 0.25)
        key = qu.identity(2**n) - 2*qu.ket2dm(key)
        quam.set_oracle()
        oracle = quam.oracle.full()
        self.iterate_equals(oracle, key)

    def test_diffusion_construction(self):
        pass

    def test_memory_construction(self):
        pass
    def test_memory_projection_construction(self):
        pass
    def test_iteration_number_calcs(self):
        pass
    def test_normal_search(self):
        pass
    def test_improved_search(self):
        pass

if __name__ == "__main__":
    test.main()
