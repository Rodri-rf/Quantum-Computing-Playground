import multiprocessing as mp
from tqdm import tqdm
import numpy as np
import traceback

from sane_applications.qft_qpe.tests_sanity_checks.test_qpe_higher_throughput_parameter_sweep import (
    test_qdrift_qpe_ising_hamiltonian_general_case,
    test_qpe_ising_hamiltonian_general_case_positive
)

NUM_QUBITS = 2
TIME_VALUES = [0.00001 * i for i in range(1, 100000, 1000)]
SHOTS_VALUES = [10000]
ANCILLA_VALUES = [5, 10, 14]

# Cartesian product of all test parameters
PARAM_COMBINATIONS = [
    (t, s, a)
    for t in TIME_VALUES
    for s in SHOTS_VALUES
    for a in ANCILLA_VALUES
]

def run_test_case(params):
    time, shots, num_ancilla = params
    try:
        test_qdrift_qpe_ising_hamiltonian_general_case(time, shots, num_ancilla)
        test_qpe_ising_hamiltonian_general_case_positive(time, shots, num_ancilla)
        return f"✔️ Time={time}, Ancilla={num_ancilla}"
    except Exception as e:
        return f"❌ Time={time}, Ancilla={num_ancilla}, Error: {str(e)}\n{traceback.format_exc()}"

def main():
    print("Running test suite with multiprocessing...")
    with mp.Pool(processes=mp.cpu_count()) as pool:
        results = list(tqdm(pool.imap_unordered(run_test_case, PARAM_COMBINATIONS), total=len(PARAM_COMBINATIONS)))

    # Print summary
    for res in results:
        print(res)

if __name__ == "__main__":
    main()
