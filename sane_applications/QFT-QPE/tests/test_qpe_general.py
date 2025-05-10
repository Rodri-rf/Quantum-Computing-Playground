
import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
import numpy as np
from qiskit.quantum_info import Operator
from qiskit.circuit.library import PhaseGate
from algos import standard_qpe, generate_ising_hamiltonian, exponentiate_hamiltonian, prepare_ground_state_circuit, calculate_ground_state_and_energy
from qiskit_aer import AerSimulator
from qiskit import transpile, QuantumCircuit
from qiskit.visualization import plot_histogram
import pytest
import math


# Setup for the test environment
LOG_FILE = "qpe_histograms_log.txt"
if os.path.exists(LOG_FILE): # Clear log file at start of test run
    os.remove(LOG_FILE)

# Sanity check: QPE on a simple phase gate
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


# Real deal: QPE to estimate eigenvalues of Ising Hamiltonian

# Parameters for the Ising model
NUM_QUBITS = 2
J = 1.2
G = 1.0

# Parameter sweeps
TIME_VALUES = [0.01, 0.1, 0.5, 1.0]
SHOTS_VALUES = [1000]
ANCILLA_VALUES = [4, 6]

@pytest.mark.parametrize("time", TIME_VALUES)
@pytest.mark.parametrize("shots", SHOTS_VALUES)
@pytest.mark.parametrize("num_ancilla", ANCILLA_VALUES)
def test_qpe_hamiltonian_non_qdrift(time, shots, num_ancilla):
    """Test non-qDRIFT QPE with Hamiltonian exponentiation."""

    # Generate Ising Hamiltonian
    H = generate_ising_hamiltonian(NUM_QUBITS, J, G)
    ground_state, ground_energy = calculate_ground_state_and_energy(H)
    eigenstate_circuit = prepare_ground_state_circuit(ground_state)

    # Expected phase calculation
    expected_phase = (ground_energy.real * time) / (2 * np.pi)

    # Exponentiate the Hamiltonian
    U = exponentiate_hamiltonian(H, time)

    # Construct the QPE circuit
    qpe_circuit = standard_qpe(U, eigenstate_circuit, num_ancilla)

    # Simulate the circuit
    simulator = AerSimulator()
    compiled_circuit = transpile(qpe_circuit, simulator)
    result = simulator.run(compiled_circuit, shots=shots).result()
    counts = result.get_counts()

    # Determine the most probable bitstring
    most_probable = max(counts, key=counts.get)
    estimated_decimal = int(most_probable, 2) / (2 ** num_ancilla)
    estimated_phase = 2 * np.pi * estimated_decimal
    estimated_energy = estimated_phase / time

    # Save histogram
    filename = f"non_qdrift_t{time}_shots{shots}_a{num_ancilla}.png"
    plot_histogram(counts).savefig(filename)

    # Save circuit diagram
    circuit_filename = f"circuit_non_qdrift_t{time}_shots{shots}_a{num_ancilla}.png"
    qpe_circuit.draw("mpl").savefig(circuit_filename)

    # Log data with the expected phase and number of shots
    with open(LOG_FILE, "a") as log_file:
        log_file.write(f"{filename},{time},{shots},{num_ancilla},{most_probable},{estimated_phase},{expected_phase},{estimated_energy},{ground_energy}\n")

    # Assertions
    assert np.isclose(estimated_energy, ground_energy.real, atol=0.5), (
        f"Estimated energy {estimated_energy} does not match ground energy {ground_energy.real}"
    )