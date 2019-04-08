import warnings
warnings.filterwarnings("ignore")

import numpy as np
from qiskit import ClassicalRegister, QuantumRegister, QuantumCircuit, execute

def bit_str_to_int(bit_str):
    bit_arr = np.array([c == "1" for c in bit_str])
    nums = np.power(2, np.arange(len(bit_str)-1,-1,-1))
    return np.sum(bit_arr*nums)

class QuAM:

    def __init__(self, pattern_length, control_length=1):
        self.bit_num = pattern_length
        self.load_bits = QuantumRegister(pattern_length, "load")
        self.mem_bits = QuantumRegister(pattern_length, "mem")
        self.control_reg = QuantumRegister(1, "control")
        self.load_mem_circuit = None
        self.load_query_circuit = None
        self.retrieval_circuit = None
        self.reset_circuit = None

    def load_patterns(self, patterns):
        amps = np.zeros(2**self.bit_num)
        for i in range(len(patterns)):
            ind = bit_str_to_int(patterns[i])
            amps[ind] = 1
        amps /= np.sqrt(len(patterns))

        self.load_mem_circuit = QuantumCircuit(self.mem_bits)
        self.load_mem_circuit.initialize(amps, [self.mem_bits[i] for i in range(self.bit_num)])

    def load_query(self, query):
        # initialize query state
        amps = np.zeros(2**self.bit_num)
        for (key, val) in query.items():
            ind = bit_str_to_int(key)
            amps[ind] = val
        if np.abs(np.sum(amps) - 1) > 1e-6:
            raise Exception("amplitudes not normalized")

        self.load_query_circuit = QuantumCircuit(self.load_bits)
        self.load_query_circuit.initialize(amps, self.load_bits)

    def make_retrieval_circuit(self):
        # create retrieval algorithm
        comparison_circuit = QuantumCircuit(self.load_bits, self.mem_bits, self.control_reg)
        comparison_circuit.h(self.control_reg[0])
        for i in range(self.bit_num):
            comparison_circuit.cx(self.load_bits[i], self.mem_bits[i])
            comparison_circuit.x(self.mem_bits[i])

        hamming_circuit = self.hamming_measure()

        # create inverse of comparison circuit (inversion of circuit apparently not supported)
        inverse_comparison = comparison_circuit.inverse()

        self.retrieval_circuit = comparison_circuit + hamming_circuit + inverse_comparison


    def hamming_measure(self, control_bit):
        """Constructs the operator differentiating between Hamming distances

        Constructed by Unitary decomposition Prod(CU^-2)*Prod(U) (with relative phase corrector)
        """
        # prepare necessary gates
        # PROD(U) part
        unitary = QuantumCircuit(self.mem_bits)
        param = np.pi/(2*self.bit_num)
        for i in range(self.bit_num):
            unitary.u1(-param, self.mem_bits[i])
        # PROD(CU^2) (with corrector)
        c_unitary = QuantumCircuit(self.mem_bits, self.control_reg)
        c_unitary.u1(-2*param*self.bit_num, control_bit) # corrector
        for i in range(self.bit_num):
            c_unitary.cu1(2*param, control_bit, self.mem_bits[i])
        # combine operations
        hamm_unitary = unitary + c_unitary
        return hamm_unitary
