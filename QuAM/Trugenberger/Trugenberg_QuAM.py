import warnings
warnings.filterwarnings("ignore")

import numpy as np
from qiskit import ClassicalRegister, QuantumRegister, QuantumCircuit, execute
from qiskit import BasicAer

def bit_str_to_int(bit_str):
    bit_arr = np.array([c == "1" for c in bit_str])
    nums = np.power(2, np.arange(len(bit_str)-1,-1,-1))
    return np.sum(bit_arr*nums)

class QuAM:

    def __init__(self, pattern_length, control_length=1):
        self.bit_num = pattern_length
        self.control_reg = QuantumRegister(control_length, "control")
        self.mem_bits = QuantumRegister(pattern_length, "mem")
        self.load_bits = QuantumRegister(pattern_length, "load")
        self.aux_result_reg = ClassicalRegister(control_length, "aux_result")
        self.data_reg = ClassicalRegister(pattern_length, "data")
        self.load_mem_circuit = None
        self.load_query_circuit = None
        self.retrieval_circuit = None
        self.evaluation_circuit = None
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

    def make_retrieval_circuit(self, controls=1):
        # create retrieval algorithm
        for c in range(controls):
            comparison_circuit = QuantumCircuit(self.load_bits, self.mem_bits, self.control_reg)
            comparison_circuit.h(self.control_reg[c])
            for i in range(self.bit_num):
                comparison_circuit.cx(self.load_bits[i], self.mem_bits[i])
                comparison_circuit.x(self.mem_bits[i])

            hamming_circuit = self.hamming_measure(self.control_reg[c])

            # create inverse of comparison circuit (inversion of circuit apparently not supported)
            inverse_comparison = QuantumCircuit(self.load_bits, self.mem_bits, self.control_reg)
            for i in reversed(range(self.bit_num)):
                inverse_comparison.x(self.mem_bits[i])
                inverse_comparison.cx(self.load_bits[i], self.mem_bits[i])
            inverse_comparison.h(self.control_reg[c])
            # compensate for error in paper giving relative phase of -i
            inverse_comparison.u1(-np.pi/2, self.control_reg[c])

            if c == 0:
                self.retrieval_circuit = comparison_circuit + hamming_circuit + inverse_comparison
            else:
                self.retrieval_circuit += comparison_circuit + hamming_circuit + inverse_comparison


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

    def simulate_match_state(self):
        """Performs the matching between current query, memory and control register, returns simulated state, including the auxiliary bits.
        """
        match_circuit = QuantumCircuit(self.control_reg, self.mem_bits, self.load_bits)
        match_circuit = self.load_query_circuit + self.load_mem_circuit + self.retrieval_circuit

        # print(match_circuit)

        backend = BasicAer.get_backend("statevector_simulator")
        vector = execute(match_circuit, backend).result().get_statevector()

        return vector

    def create_evaluation_circuit(self, scheme="post_select"):
        if scheme == "post_select":
            # Perform match using the post-select procedure
            self.evaluation_circuit = QuantumCircuit(self.mem_bits, self.control_reg, self.aux_result_reg, self.data_reg)

            self.evaluation_circuit.measure(self.control_reg, self.aux_result_reg)
            self.evaluation_circuit.barrier(self.control_reg)

            fail_state = np.array([0]*self.bit_num)
            fail_state[2**self.bit_num-1] = 1
            # success scenario
            self.evaluation_circuit.measure(self.mem_bits).if_c(self.aux_result_reg, 0)
            # fail scenario
            for i in range(1, 2**self.bit_num):
                self.evaluation_circuit.initialize(fail_state, self.control_reg).if_c(self.aux_result_reg, i)
                self.evaluation_circuit.measure(self.control_reg, self.data_reg).if_c(self.aux_result_reg, i)

        elif scheme == "amplification":
            # Perform match using the amplitude amplification procedure on the auxiliary bits
            pass
        else:
            raise ValueError("Scheme not recognized among ['post_select', 'amplification']")

    def match(self):
        
