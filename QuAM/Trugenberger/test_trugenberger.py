import warnings
warnings.filterwarnings("ignore")

import numpy as np
from Trugenberg_QuAM import QuAM
from qiskit import QuantumCircuit, BasicAer, execute
import qutip as qu
import unittest

def bit_str_to_int(bit_str):
    bit_arr = np.array([c == "1" for c in bit_str])
    nums = np.power(2, np.arange(len(bit_str)-1,-1,-1))
    return np.sum(bit_arr*nums)

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

class TrugenbergerTest(unittest.TestCase):

    def assert_iterative(self, a1, a2):
        for i in range(len(a1)):
            self.assertAlmostEqual(a1[i], a2[i])
    def assert_unitaries(self, m1, m2):
        for i in range(m1.shape[0]):
            for j in range(m1.shape[1]):
                self.assertAlmostEqual(m1[i,j], m2[i,j])

    def test_pattern_loading(self):
        patterns = ["100", "001"]
        quam = QuAM(3)
        quam.load_patterns(patterns)

        backend = BasicAer.get_backend("statevector_simulator")
        job = execute(quam.load_mem_circuit, backend)

        state_vec = job.result().get_statevector()
        key = np.array([0, 1, 0, 0, 1, 0, 0, 0])/np.sqrt(2)
        self.assert_iterative(key, state_vec)

    def test_hamm_exp_construction(self):
        bits = 2
        control_bits = 2
        c_bit = 1

        hamm_unitary = (qu.sigmaz() + 1)/2
        id = qu.identity([2]*(bits-1))
        hamm_unitary = qu.tensor(hamm_unitary, id)
        order = np.arange(bits)
        hamm_tot = qu.Qobj()
        for i in range(bits):
            hamm_temp = hamm_unitary.permute(order)
            hamm_tot += hamm_temp
            order = np.roll(order, 1)
        c_unitary = qu.sigmaz()
        if control_bits > 1:
            id = qu.identity([2]*(control_bits-1))
            order = np.roll(np.arange(control_bits), c_bit)
            c_unitary = qu.tensor(id, c_unitary)
            c_unitary = c_unitary.permute(order)
        hamm_tot = qu.tensor(c_unitary, hamm_tot)
        hamm_tot = (1j*np.pi*hamm_tot/(2*bits)).expm()
        # account for global shift
        hamm_tot *= np.exp(-1j*np.pi/2)
        key = hamm_tot.data

        quam = QuAM(bits, control_bits)
        hamm_op = quam.hamming_measure(quam.control_reg[c_bit])
        backend = BasicAer.get_backend("unitary_simulator")
        unitary = execute(hamm_op, backend).result().get_unitary()

        self.assert_unitaries(unitary, key)

    def test_retrieval_circuit(self):
        bits_comb = np.arange(1,6)
        control_bits_comb = np.arange(1,6)

        for bits in bits_comb:
            for control_bits in control_bits_comb:
                mem_size = np.random.randint(1, 2**bits)
                mem_set = set()
                for i in range(2**bits):
                    mem_set.add(i)
                memory = [0]*mem_size
                for i in range(mem_size):
                    state = mem_set.pop()
                    memory[i] = format(state, "0" + str(bits) + "b")
                query = format(np.random.randint(0, 2**bits), "0" + str(bits) + "b")

                quam = QuAM(bits, control_bits)
                quam.load_patterns(memory)
                quam.load_query({query : 1})
                quam.make_retrieval_circuit(controls=control_bits)

                key = qu.Qobj()
                for k in range(len(memory)):
                    mem_state = memory[k]
                    for s in range(control_bits+1):
                        query_memory_string = query + mem_state
                        control_state_set = set()
                        def create_perms(state_set, ref, start, depth, limit):
                            if depth >= limit:
                                state_set.add(ref)
                                return
                            for i in range(start, len(ref)):
                                if ref[i] == "0":
                                    create_perms(state_set, ref[:i] + "1" + ref[i+1:], i+1, depth+1, limit)
                        create_perms(control_state_set, "0"*control_bits, 0, 0, s)
                        phase = np.pi/(2*bits)*str_dist(query, memory[k])
                        factor = 1/np.sqrt(len(memory))*np.cos(phase)**(control_bits-s)*np.sin(phase)**s

                        state_strings = [query_memory_string + J for J in control_state_set]
                        contrib = qu.Qobj()
                        for state_str in state_strings:
                            state = qu.basis(2**bits, bit_str_to_int(state_str[:bits]))
                            state = qu.tensor(state, qu.basis(2**bits, bit_str_to_int(state_str[bits:2*bits])))
                            state = qu.tensor(state, qu.basis(2**control_bits, bit_str_to_int(state_str[2*bits:(2*bits+control_bits)])))
                            contrib += state
                        key += factor*contrib
                # account for global shift
                key *= np.exp(-1j*np.pi/2*control_bits)

                # # print(key)
                # print()
                # print("key content (i,p,b)")
                # key_data = key.full()[:, 0]
                # for i in range(len(key_data)):
                #     if key_data[i] != 0:
                #         bit_str = format(i, "0" + str(2*bits + control_bits) + "b")
                #         bit_str = "{},{},{}".format(bit_str[:bits], bit_str[bits:(2*bits)], bit_str[2*bits:(2*bits+control_bits)])
                #         print(bit_str, key_data[i])

                vector = quam.simulate_match_state()
                # vector_data = vector
                # print("result content (b,p,i)")
                # for i in range(len(vector_data)):
                #     if vector_data[i] != 0:
                #         bit_str = format(i, "0" + str(2*bits + control_bits) + "b")
                #         bit_str = "{},{},{}".format(bit_str[:control_bits], bit_str[control_bits:(control_bits + bits)], bit_str[(control_bits + bits):])
                #         print(bit_str, vector_data[i])

                # adhere to qiskit tensor structure
                order = np.flip(np.arange(3))
                key = key.permute(order)

                key = key.full()[:, 0]
                # print(key)
                # print(vector)

                self.assert_iterative(vector, key)

if __name__ == "__main__":
    unittest.main()
