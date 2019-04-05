import warnings
warnings.filterwarnings("ignore")

import numpy as np
from QuAM import QuAM
from qiskit import QuantumCircuit, BasicAer, execute
import qutip as qu
import unittest

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
        job = execute(quam.load_circuit, backend)

        state_vec = job.result().get_statevector()
        key = np.array([0, 1, 0, 0, 1, 0, 0, 0])/np.sqrt(2)
        self.assert_iterative(key, state_vec)

    def test_hamm_exp_construction(self):
        bits = 2
        control_bits = 1
        c_bit = 0

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


if __name__ == "__main__":
    unittest.main()
