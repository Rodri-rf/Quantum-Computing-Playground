
import sys
import os
import numpy as np
from qiskit.quantum_info import Operator
from qiskit.circuit.library import PhaseGate
from sane_applications.qft_qpe.algos import standard_qpe, generate_ising_hamiltonian, exponentiate_hamiltonian, prepare_eigenstate_state_circuit, calculate_ground_state_and_energy
from qiskit_aer import AerSimulator
from qiskit import transpile, QuantumCircuit
from qiskit.visualization import plot_histogram
import pytest
import math
from random_cool_trinkets.utils import hamiltonian_explicit, hamiltonian_simplified, format_hamiltonian_matrix_as_latex, plot_complex_unit_circle

# Setup for the test environment
LOG_FILE = "qpe_histograms_log.txt"
if os.path.exists(LOG_FILE): # Clear log file at start of test run
    os.remove(LOG_FILE)

# Sanity check: QPE on a simple phase gate and a single qubit, and three ancilla qubits
@pytest.mark.parametrize("phase, expected_bin", [
    (math.pi / 4, '001'),
    (math.pi / 2,  '010'),
    (3 * math.pi / 4, '011'),
    (math.pi,   '100'),
    (5 * math.pi / 4, '101'),
    (3 * math.pi / 2,  '110'),
    (7 * math.pi / 4, '111'),
])
def test_general_qpe_with_parametrized_phase(phase, expected_bin):
    unitary = Operator(PhaseGate(phase))
    
    eigenstate = QuantumCircuit(1)
    eigenstate.x(0)

    num_ancilla = 3
    shots = 1024
    qc = standard_qpe(unitary, eigenstate, num_ancilla)

    simulator = AerSimulator()
    job = simulator.run(transpile(qc, simulator), shots=shots)
    result = job.result()
    counts = result.get_counts(qc)
    most_probable = max(counts, key=counts.get)
    estimated_decimal = int(most_probable, 2) / (2 ** num_ancilla)
    estimated_phase = 2 * math.pi * estimated_decimal
    assert most_probable.startswith(expected_bin), f"Expected prefix {expected_bin}, got {most_probable}"

    plot_histogram(counts).savefig(f"output_general_phase_{round(phase, 3)}.png")
    # Generate filename and save histogram
    filename = f"output_general_phase_{round(phase, 3)}_a{num_ancilla}.png"
    plot_histogram(counts).savefig(filename)

    # save circuit diagram
    qc.draw("mpl").savefig(f"circuit_general_phase_{round(phase, 3)}_a{num_ancilla}.png")

    # Log data
    with open(LOG_FILE, "a") as log_file:
        log_file.write(f"{filename},{phase},{num_ancilla},{most_probable},{estimated_phase}\n")

# Sanity check but with varying number of ancilla qubits
ANCILLA_VALUES = [4]
@pytest.mark.parametrize("phase, expected_bin, num_ancilla", [(2 * math.pi * k / 2**n, bin(k)[2:].zfill(n), n) for n in ANCILLA_VALUES for k in range(1, 2**n) ])
def test_generat_qpe_with_parametrized_phase_and_ancilla(phase, expected_bin, num_ancilla):
    """Test QPE with parametrized phase and varying number of ancilla qubits."""
    unitary = Operator(PhaseGate(phase))

    # Prepare the eigenstate circuit
    eigenstate = QuantumCircuit(1)
    eigenstate.x(0)

    # Generate the QPE circuit
    qc = standard_qpe(unitary, eigenstate, num_ancilla)

    # Simulate the circuit
    simulator = AerSimulator()
    job = simulator.run(transpile(qc, simulator), shots=1024)
    result = job.result()
    counts = result.get_counts(qc)

    # Determine the most probable bitstring
    most_probable = max(counts, key=counts.get)
    
    # Assert that the most probable bitstring starts with the expected prefix
    assert most_probable.startswith(expected_bin), f"Expected prefix {expected_bin}, got {most_probable}"


# Real deal: QPE to estimate eigenvalues of Ising Hamiltonian

# Parameters for the Ising model
NUM_QUBITS = 2
J = 1.2
G = 1.0

# Parameter sweep for time
TIME_VALUES = [0.01, 0.05, 0.1, 0.5, 1.0, 5.0]
SHOTS_VALUES = [10000]
ANCILLA_VALUES = [5]

@pytest.mark.parametrize("time", TIME_VALUES)
@pytest.mark.parametrize("shots", SHOTS_VALUES)
@pytest.mark.parametrize("num_ancilla", ANCILLA_VALUES)
def test_qpe_ising_hamiltonian_general_case_positive_phase(time, shots, num_ancilla):
    """Test QPE with Ising Hamiltonian (General Case) and log Hamiltonian representations."""

    # Generate Ising Hamiltonian
    H = generate_ising_hamiltonian(NUM_QUBITS, J, G)
    matrix = H.to_matrix()
    eigenvalues, eigenvectors = np.linalg.eig(matrix)
    first_positive_eigenvalue = min(eigenvalues[eigenvalues > 0])
    eigenvector_index = np.where(eigenvalues == first_positive_eigenvalue)[0][0]
    first_positive_eigenvect = eigenvectors[:, eigenvector_index]

    # Hamiltonian representations to display in the log
    hamiltonian_tensor = hamiltonian_explicit(H)
    hamiltonian_simplified_repr = hamiltonian_simplified(H)
    hamiltonian_matrix = format_hamiltonian_matrix_as_latex(H.to_matrix())

    # Expected phase calculation
    expected_phase = (first_positive_eigenvalue.real * time) / (2 * np.pi) % 1
    expected_bitstring = bin(round(expected_phase * (2 ** num_ancilla)))[2:].zfill(num_ancilla)
    

    # Exponentiate the Hamiltonian
    U = exponentiate_hamiltonian(H, time)

    # Construct QPE circuit
    eigenstate_circuit = prepare_eigenstate_state_circuit(first_positive_eigenvect)
    qc = standard_qpe(U, eigenstate_circuit, num_ancilla)

    # Simulate the circuit
    simulator = AerSimulator()
    compiled_circuit = transpile(qc, simulator)
    result = simulator.run(compiled_circuit, shots=shots).result()
    counts = result.get_counts()

    # Determine the most probable bitstring
    most_probable = max(counts, key=counts.get)
    estimated_decimal = int(most_probable, 2) / (2 ** num_ancilla)
    estimated_phase = estimated_decimal
    estimated_energy = 2 * np.pi * estimated_phase / time

    # Save histogram
    filename = f"general_case_t{time}_shots{shots}_a{num_ancilla}.png"
    plot_histogram(counts).savefig(filename)

    # Save circuit diagram
    circuit_filename = f"circuit_general_case_t{time}_shots{shots}_a{num_ancilla}.png"
    qc.draw("mpl").savefig(circuit_filename)

    # Plot the complex unit circle
    unit_circle_filename = f"unit_circle_t{time}_shots{shots}_a{num_ancilla}.png"
    plot_complex_unit_circle(estimated_phase, expected_phase, num_ancilla, time, unit_circle_filename)

    # Log data with the Hamiltonian representation
    with open(LOG_FILE, "a") as log_file:
        log_file.write(f"general case,{filename},{time},{shots},{num_ancilla},{most_probable},{estimated_phase},{expected_phase},{estimated_energy},{first_positive_eigenvalue},{hamiltonian_tensor},{hamiltonian_simplified_repr},{hamiltonian_matrix},{expected_bitstring}\n")

    # Assert that the most probable bitstring starts with the expected prefix
    assert most_probable.startswith(expected_bitstring), f"Expected prefix {expected_bitstring}, got {most_probable}"

@pytest.mark.parametrize("time", TIME_VALUES)
@pytest.mark.parametrize("shots", SHOTS_VALUES)
@pytest.mark.parametrize("num_ancilla", ANCILLA_VALUES)

def test_qpe_ising_hamiltonian_general_case_negative_phase(time, shots, num_ancilla):
    """Test QPE with Ising Hamiltonian (General Case) and log Hamiltonian representations."""

    # Generate Ising Hamiltonian
    H = generate_ising_hamiltonian(NUM_QUBITS, J, G)
    matrix = H.to_matrix()

    # Hamiltonian representations to display in the log
    hamiltonian_tensor = hamiltonian_explicit(H)
    hamiltonian_simplified_repr = hamiltonian_simplified(H)
    hamiltonian_matrix = format_hamiltonian_matrix_as_latex(H.to_matrix())

    # Calculate the first negative eigenvalue and its corresponding eigenvector
    eigenvalues, eigenvectors = np.linalg.eig(matrix)
    first_negative_eigenvalue = min(eigenvalues)
    eigenvector_index = np.where(eigenvalues == first_negative_eigenvalue)[0][0]
    first_negative_eigenvect = eigenvectors[:, eigenvector_index]

    # Expected phase calculation
    expected_phase = (first_negative_eigenvalue.real * time) / (2 * np.pi) % 1
    expected_bitstring = bin(round(expected_phase * (2 ** num_ancilla)))[2:].zfill(num_ancilla)


    # Exponentiate the Hamiltonian
    U = exponentiate_hamiltonian(H, time)

    # Construct QPE circuit
    eigenstate_circuit = prepare_eigenstate_state_circuit(first_negative_eigenvect)
    qc = standard_qpe(U, eigenstate_circuit, num_ancilla)

    # Simulate the circuit
    simulator = AerSimulator()
    compiled_circuit = transpile(qc, simulator)
    result = simulator.run(compiled_circuit, shots=shots).result()
    counts = result.get_counts()

    # Determine the most probable bitstring
    most_probable = max(counts, key=counts.get)
    estimated_decimal = int(most_probable, 2) / (2 ** num_ancilla)
    estimated_phase = estimated_decimal
    is_eigenval_negative = estimated_phase >= 0.5
    if is_eigenval_negative:
        estimated_phase = estimated_phase - 1
    estimated_energy = 2 * np.pi * estimated_phase / time
    
    # Save histogram
    filename = f"general_case_t{time}_shots{shots}_a{num_ancilla}.png"
    plot_histogram(counts).savefig(filename)

    # Save circuit diagram
    circuit_filename = f"circuit_general_case_t{time}_shots{shots}_a{num_ancilla}.png"
    qc.draw("mpl").savefig(circuit_filename)

    # Log data with the Hamiltonian representation
    with open(LOG_FILE, "a") as log_file:
        log_file.write(f"general case,{filename},{time},{shots},{num_ancilla},{most_probable},{estimated_phase},{expected_phase},{estimated_energy},{first_negative_eigenvalue},{hamiltonian_tensor},{hamiltonian_simplified_repr},{hamiltonian_matrix},{expected_bitstring}\n")

    # Assert that the most probable bitstring starts with the expected prefix
    assert most_probable.startswith(expected_bitstring), f"Expected prefix {expected_bitstring}, got {most_probable}"

# Ancilla qubit values to sweep
ANCILLA_VALUES = [3]
@pytest.mark.parametrize("num_ancilla", ANCILLA_VALUES)
@pytest.mark.parametrize("k", range(1, 7))  # Testing only k values that fit well in binary
def test_qpe_ising_hamiltonian_exact_phases(num_ancilla, k):
    """
    Test QPE with Ising Hamiltonian (Exact Phases).
    Ensure the time parameter is set so the phase can be exactly represented.
    """

    # Generate Ising Hamiltonian
    H = generate_ising_hamiltonian(NUM_QUBITS, J, G)
    ground_state, ground_energy = calculate_ground_state_and_energy(H)
    eigenstate_circuit = prepare_eigenstate_state_circuit(ground_state)

    # Hamiltonian representations
    hamiltonian_tensor = hamiltonian_explicit(H)
    hamiltonian_simplified_repr = hamiltonian_simplified(H)
    hamiltonian_matrix = format_hamiltonian_matrix_as_latex(H.to_matrix())

    # Calculate the time parameter for exact phase estimation
    t = (2 * np.pi * k) / (ground_energy.real * (2 ** num_ancilla))

    # Expected binary string
    expected_bin = bin(k)[2:].zfill(num_ancilla)

    # Exponentiate the Hamiltonian
    U = exponentiate_hamiltonian(H, t)

    # Construct QPE circuit
    qc = standard_qpe(U, eigenstate_circuit, num_ancilla)

    # Simulate
    simulator = AerSimulator()
    result = simulator.run(transpile(qc, simulator), shots=1024).result()
    counts = result.get_counts()

    # Most probable bitstring
    most_probable = max(counts, key=counts.get)

    # Save histogram
    filename = f"exact_phase_t{round(t, 4)}_a{num_ancilla}_k{k}.png"
    plot_histogram(counts).savefig(filename)

    # Save circuit diagram
    circuit_filename = f"circuit_exact_phase_t{round(t, 4)}_a{num_ancilla}_k{k}.png"
    qc.draw("mpl").savefig(circuit_filename)

    # Log data
    with open(LOG_FILE, "a") as log_file:
        log_file.write(f"exact phase,{filename},{t},{num_ancilla},{most_probable},{expected_bin},{hamiltonian_tensor},{hamiltonian_simplified_repr},{hamiltonian_matrix}\n")
    
    assert most_probable.startswith(expected_bin), f"Expected prefix {expected_bin}, got {most_probable}"