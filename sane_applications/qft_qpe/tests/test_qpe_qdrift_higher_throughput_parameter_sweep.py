import os
import numpy as np
from qiskit.quantum_info import Operator, SparsePauliOp
from qiskit.circuit.library import PhaseGate
from sane_applications.qft_qpe.algos import generate_ising_hamiltonian, prepare_eigenstate_circuit, qdrift_qpe_extra_random, qdrift_qpe, qdrift_qpe_chat_gpts_take, generate_random_hamiltonian_with_pauli_tensor_structure, deterministic_qpe_qdrift_with_error_budget
from qiskit_aer import AerSimulator
from qiskit import transpile, QuantumCircuit
from qiskit.visualization import plot_histogram
import pytest
import pandas as pd
import csv
from line_profiler import profile
from typing import Callable, Tuple
import inspect
import datetime
# memory_guard
import psutil
import time
import os

def wait_if_memory_high(threshold_percent: float = 85.0, check_every_sec: int = 2, max_wait_sec: int = 120):
    """
    Pauses the current worker if memory usage exceeds threshold.
    Useful when running parallel tests (e.g., via pytest-xdist).
    """
    waited = 0
    while psutil.virtual_memory().percent >= threshold_percent:
        print(f"[Worker PID {os.getpid()}] High RAM usage ({psutil.virtual_memory().percent}%). Pausing...")
        time.sleep(check_every_sec)
        waited += check_every_sec
        if waited > max_wait_sec:
            print(f"[Worker PID {os.getpid()}] Waited {waited}s. Continuing anyway.")
            break


def deterministic_wrapper(
    H: SparsePauliOp,
    *,
    eigenstate: QuantumCircuit,
    time: float,
    num_qubits: int,
    num_ancilla: int
) -> QuantumCircuit:
    """
    Adapter to call deterministic_qpe_qdrift_with_error_budget with
    an error budget epsilon_total = 1 / (2^m).  That way, the diamond-norm
    error is roughly O(1/2^m), matching QPE's resolution 1/2^m.
    """
    m = num_ancilla
    # Set ε_tot = 1 / 2^m  (you can tune this if you wish)
    epsilon_total = 1.0 / (2**m)
    # deterministic_qpe_qdrift_with_error_budget takes (H, ε_tot, m, eigenstate, num_qubits)
    return deterministic_qpe_qdrift_with_error_budget(
        hamiltonian=H,
        epsilon_total=epsilon_total,
        m=m,
        eigenstate=eigenstate,
        num_qubits=num_qubits
    )

def process_qpe_results(time, num_ancilla, counts):
    most_probable = max(counts, key=counts.get)
    estimated_decimal = int(most_probable, 2) / (2 ** num_ancilla)
    estimated_phase = estimated_decimal
    eigenval_is_negative = estimated_phase >= 0.5
    if eigenval_is_negative:
        estimated_phase = estimated_phase - 1
    estimated_energy = 2 * np.pi * estimated_phase / time
    return most_probable,estimated_phase,estimated_energy


# Parameters for the Ising model
NUM_QUBITS = 2
J = 1.2
G = 1.0

#Log file for QDRIFT tests
CSV_FILE_QDRIFT_EXTRA_RANDOM = f"qdrift_ising_model_sweep_data_extra_random_{datetime.datetime.today()}.csv"  # New CSV file for QDRIFT tests
CSV_FILE_QDRIFT_QPE_ALL = f"qdrift_ising_model_sweep_data_all_{datetime.datetime.today().strftime('%Y-%m-%d')}.csv"  # New CSV file for QDRIFT tests

QDRIFT_IMPLEMENTATIONS = [(qdrift_qpe, "exponential invocations of qdrift channel"), (qdrift_qpe_chat_gpts_take, "linear invocations of qdrift channel")]
rand_pauli = generate_random_hamiltonian_with_pauli_tensor_structure(NUM_QUBITS, num_terms=9)
matrix = rand_pauli.to_matrix()
eigenvalues, eigenvectors = np.linalg.eig(matrix)
assert np.all(np.isreal(eigenvalues)), "Eigenvalues are not all real!"

# HAMILTONIANS = [("Ising", generate_ising_hamiltonian(NUM_QUBITS, J, G)), ("Random Pauli", rand_pauli), ("Simple Z",     SparsePauliOp(["Z"*NUM_QUBITS], coeffs=[1.0]))]
HAMILTONIANS = [("Ising", generate_ising_hamiltonian(NUM_QUBITS, J, G))]
RANDOMNESS = [(1, 1024),  (64, 16), (1024, 16), (2048, 16)]  # (num_random_circuits, num_shots_per_circuit)
ANCILLA_VALUES = [4, 6, 8, 10, 12]  # Number of ancilla qubits
TIME_VALUES = TIME_VALUES = list(np.logspace(-3, -1, num=100)) 

@pytest.mark.parametrize("qdrift_impl", QDRIFT_IMPLEMENTATIONS)
@pytest.mark.parametrize("calculate_ground_state", [False])
@pytest.mark.parametrize("H", HAMILTONIANS)
@pytest.mark.parametrize("num_random_circuits, num_shots_per_circuit", RANDOMNESS)
@pytest.mark.parametrize("num_ancilla", ANCILLA_VALUES)
@pytest.mark.parametrize("total_simulation_time", TIME_VALUES)


def test_qdrift_qpe_general_case(total_simulation_time, num_ancilla, qdrift_impl: Tuple[Callable, str], num_random_circuits, num_shots_per_circuit, calculate_ground_state: bool, H: SparsePauliOp):
    """Test QPE with Ising Hamiltonian (General Case) and log Hamiltonian representations."""
    # Generate Ising Hamiltonian
    type_of_hamiltonian, H = H
    matrix = H.to_matrix()
    eigenvalues, eigenvectors = np.linalg.eig(matrix)
    alpha = sum(abs(H.coeffs))

    # compute the exact target eigenvalue and eigenvector
    if calculate_ground_state:
        # Let's do the ground state
        first_positive_eigenvalue = min(eigenvalues[eigenvalues > 0])
        eigenvector_index = np.where(eigenvalues == first_positive_eigenvalue)[0][0]
        eigenstate = eigenvectors[:, eigenvector_index]
    else:
        # Let's do the largest eigenvalue
        first_positive_eigenvalue = max(eigenvalues)
        eigenvector_index = np.where(eigenvalues == first_positive_eigenvalue)[0][0]
        eigenstate = eigenvectors[:, eigenvector_index]
    
    # Prepare the eigenstate circuit
    eigenstate_circuit = prepare_eigenstate_circuit(eigenstate)

    # Expected phase calculation
    expected_phase = (first_positive_eigenvalue.real * total_simulation_time) / (2 * np.pi) % 1
    expected_bitstring = bin(round(expected_phase * (2 ** num_ancilla)))[2:].zfill(num_ancilla)

    # build the actual quantum circuits
    wait_if_memory_high(threshold_percent=95.0, check_every_sec=2, max_wait_sec=120)
    generate_qdrift_circuit, impl_name = qdrift_impl
    results = []
    simulator = AerSimulator()
    for rand_circuit in range(num_random_circuits):
        qc = generate_qdrift_circuit(H, eigenstate=eigenstate_circuit, time=total_simulation_time, num_qubits=NUM_QUBITS, num_ancilla=num_ancilla)
        # Simulate the circuit
        compiled_circuit = transpile(qc, simulator)
        result = simulator.run(compiled_circuit, shots=num_shots_per_circuit).result()
        results.append(result)
    # Aggregate counts across all random circuits
    counts = {}
    for result in results:
        result_counts = result.get_counts()
        for bitstring, count in result_counts.items():
            if bitstring in counts:
                counts[bitstring] += count
            else:
                counts[bitstring] = count

    # Determine the most probable bitstring
    most_probable, estimated_phase, estimated_energy = process_qpe_results(total_simulation_time, num_ancilla, counts)

    # Calculate error
    eigenvalue_error = np.abs(estimated_energy - first_positive_eigenvalue)

    # Append directly to CSV
    header = [
        "Num Qubits", "Time", "Shots", "Num Ancilla",
        "Exact Eigenvalue", "Expected Phase",
        "Most Probable Bitstring", "Estimated Phase",
        "Estimated Eigenvalue", "Eigenvalue Error", "Alpha", "QDRIFT Implementation", 
        "type of Hamiltonian", "Num Random Circuits", "Num Shots per Circuit", "Circuit Depth",
        "Raw results" 
    ]
    row = [
        NUM_QUBITS, total_simulation_time, num_random_circuits * num_shots_per_circuit, num_ancilla,
        first_positive_eigenvalue, expected_phase,
        most_probable, estimated_phase,
        estimated_energy, eigenvalue_error, alpha, impl_name, type_of_hamiltonian,
        num_random_circuits, num_shots_per_circuit, qc.depth() if isinstance(qc, QuantumCircuit) else "N/A",
        str(counts)  # Store raw results as a string
    ]

    file_exists = os.path.isfile(CSV_FILE_QDRIFT_QPE_ALL)
    with open(CSV_FILE_QDRIFT_QPE_ALL, mode='a', newline='') as file:
        writer = csv.writer(file)
        if not file_exists:
            writer.writerow(header)
        writer.writerow(row)

    # Assert that the most probable bitstring starts with the expected prefix
    assert most_probable.startswith(expected_bitstring), f"Expected prefix {expected_bitstring}, got {most_probable}"



    
'''
def test_qdrift_qpe_extra_random(tot_simulation_time, num_ancilla, num_samples):
    H = generate_ising_hamiltonian(NUM_QUBITS, J, G)
    matrix = H.to_matrix()
    eigenvalues, eigenvectors = np.linalg.eig(matrix)
    alpha = sum(abs(H.coeffs))

    # let's do the largest eigenvalue
    largest_eigenvalue = max(eigenvalues)
    eigenvector_index = np.where(eigenvalues == largest_eigenvalue)[0][0]
    largest_eigenvect = eigenvectors[:, eigenvector_index]
    eigenstate_circuit = prepare_eigenstate_circuit(largest_eigenvect)

    # Expected phase calculation
    expected_phase = (largest_eigenvalue.real * tot_simulation_time) / (2 * np.pi) % 1
    expected_bitstring = bin(round(expected_phase * (2 ** num_ancilla)))[2:].zfill(num_ancilla)
    # build the actual quantum circuits

    curr_samples = 0
    circuits = []
    simulator = AerSimulator()
    while curr_samples < num_samples:
        qc = qdrift_qpe_extra_random(hamiltonian=H, eigenstate=eigenstate_circuit, num_qubits=NUM_QUBITS, num_ancilla=num_ancilla, total_simulation_time=tot_simulation_time, num_samples=num_samples)
        # Simulate the circuit
        compiled_circuit = transpile(qc, simulator)
        circuits.append(compiled_circuit)
        curr_samples += 2 ** num_ancilla - 1 # Each circuit gives us 2^num_ancilla samples
    # Aggregate counts across all random circuits
    counts = {}
    for compiled_circuit in circuits:
        result = simulator.run(compiled_circuit, shots=1).result()
        result_counts = result.get_counts()
        for bitstring, count in result_counts.items():
            if bitstring in counts:
                counts[bitstring] += count
            else:
                counts[bitstring] = count
    # Determine the most probable bitstring
    most_probable = max(counts, key=counts.get)
    estimated_decimal = int(most_probable, 2) / (2 ** num_ancilla)
    estimated_phase = estimated_decimal
    estimated_energy = 2 * np.pi * estimated_phase / tot_simulation_time
    # Calculate error
    eigenvalue_error = np.abs(estimated_energy - largest_eigenvalue)
    # Append directly to CSV
    header = [
        "Num Qubits", "Time", "Shots", "Num Ancilla",
        "Exact Eigenvalue", "Expected Phase",
        "Most Probable Bitstring", "Estimated Phase",
        "Estimated Eigenvalue", "Eigenvalue Error", "Alpha"
    ]
    row = [
        NUM_QUBITS, tot_simulation_time, num_samples, num_ancilla,
        largest_eigenvalue, expected_phase,
        most_probable, estimated_phase,
        estimated_energy, eigenvalue_error, alpha
    ]
    file_exists = os.path.isfile(CSV_FILE_QDRIFT_EXTRA_RANDOM)
    with open(CSV_FILE_QDRIFT_EXTRA_RANDOM, mode='a', newline='') as file:
        writer = csv.writer(file)
        if not file_exists:
            writer.writerow(header)
        writer.writerow(row)
    
    assert most_probable.startswith(expected_bitstring), f"Expected prefix {expected_bitstring}, got {most_probable}"
'''
