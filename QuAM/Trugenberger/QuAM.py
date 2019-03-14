import numpy as np
from qiskit import ClassicalRegister, QuantumRegister, QuantumCircuit, execute

def bit_str_to_int(bit_str):
    bit_arr = np.array([c == "1" for c in bit_str])
    return np.sum(bit_arr*np.arange(2**(len(bit_str)-1, -1, -1)))

class QuAM:

    def __init__(self, pattern_length):
        self.bit_num = pattern_length
        self.load_bits = QuantumRegister(pattern_length, "load")
        self.mem_bits = QuantumRegister(pattern_length, "mem")
        self.control_reg = QuantumCircuit(2, "control")
        self.load_circuit = None
        self.retrieval_circuit = None
        self.reset_circuit = None

    def load_patterns(patterns):
        patterns = np.array([np.array([x == "1" for x in s]) for s in patterns])
        amps = np.zeros(2**self.bit_num)
        for i in range(patterns):
            ind = bit_str_to_int(patterns[i])
            amps[ind] = 1
        amps /= np.sqrt(length(patterns))

        self.load_circuit = QuantumCircuit(mem_bits)
        self.load_circuit.initialize(amps, [self.mem_bits[i] for i in range(self.bit_num)])

    def retrieve_pattern(query):
        # prepare necessary gates
        unitary = QuantumCircuit(self.mem_bits)
        param = np.pi/(2*self.bit_num)
        for i in range(self.bit_num):
            unitary.u1(-param, self.mem_bits[i])
        c_unitary = QuantumCircuit(self.self.mem_bits)
        for i in range(self.bit_num):
            c_unitary.cu1(2*param, self.control_reg[0])
        exp_unitary = unitary + c_unitary
        
        # initialize query state
        amps = np.zeros(2**self.bit_num)
        for (key, val) in query.items():
            ind = bit_str_to_int(key)
            amps[ind] = val
        if np.abs(np.sum(amps) - 1) > 1e-6:
            raise Exception("amplitudes not normalized")

        init_circ = QuantumCircuit(self.load_bits)
        init_circ.initialize(amps, self.load_bits)

        # create retrieval algorithm
        self.retrieval_circuit = QuantumCircuit(self.load_bits, self.mem_bits, self.control_reg)
        self.retrieval_circuit.h(self.control_reg[0])
        for i in range(self.bit_num):
            self.retrieval_circuit.cx(self.load_bits[i], self.mem_bits[i])
            self.retrieval_circuit.x(self.mem_bits[i])
