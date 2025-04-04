from qiskit import QuantumCircuit, transpile
from qiskit import QuantumCircuit, QuantumRegister, ClassicalRegister
from qiskit.circuit.library import QFT, UnitaryGate
from qiskit.quantum_info import Pauli, SparsePauliOp, Operator
from qiskit_aer import Aer
from qiskit.visualization import plot_histogram, circuit_drawer
from qiskit.quantum_info import Statevector
from typing import Optional, Union
import random
import numpy as np
from functools import reduce
import scipy.linalg
import pandas as pd

# Type aliases
coefficient = float
Hamiltonian = list[tuple[coefficient, Pauli]]

# Function to generate a random Hamiltonian
def generate_random_hamiltonian(num_qubits, num_terms) -> Hamiltonian:
    pauli_matrices = [Pauli('X'), Pauli('Z'), Pauli('I')]
    hamiltonian_terms = []
    for _ in range(num_terms):
        pauli_string = reduce(lambda x, y: x.tensor(y), random.choices(pauli_matrices, k=num_qubits))
        hamiltonian_terms.append((random.uniform(0, 1), pauli_string)) # The first element is the coefficient
    return hamiltonian_terms


def construct_hamiltonian(hamiltonian_terms: Hamiltonian) -> SparsePauliOp:
    """Constructs the full Hamiltonian as a SparsePauliOp."""
    return SparsePauliOp([term[1] for term in hamiltonian_terms], coeffs=[term[0] for term in hamiltonian_terms])

def get_eigenvalues_vectors(hamiltonian: SparsePauliOp):
    """Computes eigenvalues and eigenvectors of the Hamiltonian."""
    matrix = hamiltonian.to_matrix()  # Convert to a dense matrix
    eigenvalues, eigenvectors = scipy.linalg.eigh(matrix)
    return eigenvalues, eigenvectors


def eigenvector_to_circuit(eigenvector: np.ndarray) -> QuantumCircuit:
    """Creates a quantum circuit that prepares a given eigenvector."""
    num_qubits = int(np.log2(len(eigenvector)))  # Number of qubits
    statevector = Statevector(eigenvector)  # Convert to Qiskit state
    qc = QuantumCircuit(num_qubits)
    qc.initialize(statevector)  # Initialize circuit with eigenstate
    return qc

def exponentiate_hamiltonian(hamiltonian: SparsePauliOp, time: float) -> SparsePauliOp:
    """Exponentiates the Hamiltonian to obtain U = e^(-i H t)."""
    matrix = hamiltonian.to_matrix()
    unitary_matrix = scipy.linalg.expm(1j * time * matrix)
    return SparsePauliOp(unitary_matrix)

#--------------------- QPE implementation ---------------------#
def qpe(hamiltonian: SparsePauliOp, num_qubits: int, num_ancillas: int) -> QuantumCircuit:
    """Quantum Phase Estimation (QPE) circuit."""
    qc = QuantumCircuit(num_qubits + num_ancillas, num_ancillas)
    qc.h(range(num_ancillas))  # Apply Hadamard gates to ancilla qubits

    # Controlled unitary operations
    for i in range(num_ancillas):
        qc.append(UnitaryGate(hamiltonian, label='U'), [i] + list(range(num_ancillas, num_qubits + num_ancillas)))

    # Inverse QFT
    qc.append(QFT(num_ancillas, inverse=True), range(num_ancillas))

    # Measurement
    qc.measure(range(num_ancillas), range(num_ancillas))
    return qc

def standard_qpe(unitary: Operator, eigenstate: Union[QuantumCircuit, Statevector], num_ancilla: int, swap: Optional[bool] = True) -> QuantumCircuit:
    """Constructs a standard Quantum Phase Estimation (QPE) circuit using controlled-U operations."""
    
    num_qubits = unitary.num_qubits
    qc = QuantumCircuit(num_ancilla + num_qubits, num_ancilla)

    if isinstance(eigenstate, QuantumCircuit):
        # If eigenstate is a circuit, append it to the quantum circuit
        qc.append(eigenstate, range(num_ancilla, num_ancilla + num_qubits))
    elif isinstance(eigenstate, Statevector):
        # If eigenstate is a statevector, prepare it in the circuit
        qc.initialize(eigenstate, range(num_ancilla, num_ancilla + num_qubits))

    # Apply Hadamard gates to ancilla qubits
    qc.h(range(num_ancilla))

    # Apply controlled-U^(2^k) gates
    for k in range(num_ancilla):
        for _ in range(2**k):  # Apply U 2^k times
            controlled_U = unitary.control(1)
            qc.append(controlled_U, [k] + list(range(num_ancilla, num_ancilla + num_qubits)))


    # Apply inverse QFT on ancilla qubits
    qft_dg = QFT(num_ancilla, do_swaps=swap, inverse=True)
    qc.append(qft_dg, range(num_ancilla))

    # Measure ancilla qubits
    qc.measure(range(num_ancilla), range(num_ancilla))

    return qc


def standard_qpe_v2(unitary: Operator, eigenstate: Statevector, num_ancilla: int, num_target: int, swap: Optional[bool] = True) -> QuantumCircuit:
    ancilla_quantum_register = QuantumRegister(size=num_ancilla, name="ancilla")
    target_register = QuantumRegister(size=num_target, name="target")
    classical_register = ClassicalRegister(size=num_ancilla)
    qc = QuantumCircuit(ancilla_quantum_register, target_register, classical_register)

    # Step 1: initialize target register in the eigenstate
    if isinstance(eigenstate, Statevector):
        qc.initialize(eigenstate, target_register)
    else:
        # scream
        raise ValueError("Eigenstate must be a Statevector.")
    
    # Step 2: Apply QFT to ancilla register
    qc.append(QFT(num_ancilla, do_swaps=swap), ancilla_quantum_register)
    

    # Step 3: Apply controlled-U gates, under the convention that qubit 0 is the MSB
    for num, qubit in enumerate(ancilla_quantum_register):
        for _ in range(2**(num_ancilla - num - 1)):
            controlled_U = unitary.control(1)
            qc.append(controlled_U, [qubit] + target_register[:])

    
    # Step 4: Apply inverse QFT to ancilla register
    qc.append(QFT(num_ancilla, inverse=True, do_swaps=swap), ancilla_quantum_register)

    # Step 5: Measure ancilla register
    qc.measure(ancilla_quantum_register, classical_register)

    return qc

