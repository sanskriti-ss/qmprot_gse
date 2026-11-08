# ABOUTME: Times full boosted-VQE matrices, quadrature measurements and determinant rows on frozen glycine.
# ABOUTME: Records all matrix bytes, both overlap quadratures and returned Ritz energies for paired verification.
import argparse
from hashlib import sha256
import json
from pathlib import Path
from time import perf_counter

import numpy as np

from algorithms.vqe_CB import ClassicallyBoostedVQE
from core.hf_verification import _build_hamiltonian_matrix
from experiments.experiment5_term_count_sweep import _prefix_hamiltonian
from experiments.provenance import load_problem


def fingerprint(array):
    return sha256(np.ascontiguousarray(array).tobytes()).hexdigest()


def measure(operation):
    start = perf_counter()
    result = operation()
    return result, perf_counter() - start


def boost(path):
    problem = load_problem(path)
    ham = _prefix_hamiltonian(problem.hamiltonian, 200)
    solver = ClassicallyBoostedVQE(ham, n_layers=2, random_seed=42)
    solver.build_ansatz()
    matrix = _build_hamiltonian_matrix(ham)
    reference = ham.cs_initial_state
    row = int(np.flatnonzero(reference).item())
    observed_indices = np.flatnonzero((matrix[row] != 0) | (reference != 0))
    trials = []
    for parameters in np.random.default_rng(42).normal(size=(3, solver.n_parameters)):
        unitary, build_seconds = measure(lambda: solver._ansatz_unitary(parameters))
        amplitudes, overlap_seconds = measure(lambda: [solver._hadamard_test(unitary, i) for i in observed_indices])
        energy = solver.cost_function(parameters)
        ritz, elapsed = measure(lambda: solver._classical_boost(parameters, energy))
        trials.append(dict(science=dict(unitary_sha256=fingerprint(unitary), overlap_sha256=fingerprint(amplitudes),
                                        raw_energy=energy.hex(), ritz_energy=ritz.hex()),
                           unitary_seconds=build_seconds, overlap_seconds=overlap_seconds, boost_seconds=elapsed))
    return dict(terms=ham.n_terms, qubits=ham.n_qubits, measured_overlaps=len(observed_indices),
                input_sha256=sha256(path.read_bytes()).hexdigest(), trials=trials)


def determinant_row(path):
    import inspect

    full = load_problem(path).hamiltonian
    ham = _prefix_hamiltonian(full, 200)
    occupied = ham.molecule.n_electrons
    row = ((1 << occupied) - 1) << (ham.n_qubits - occupied)
    if "row" in inspect.signature(_build_hamiltonian_matrix).parameters:
        evaluate = lambda: _build_hamiltonian_matrix(ham, row=row)
    else:
        evaluate = lambda: _build_hamiltonian_matrix(ham)[row].copy()
    target, elapsed = measure(evaluate)
    return dict(terms=ham.n_terms, qubits=ham.n_qubits, row=row, sha256=fingerprint(target), seconds=elapsed)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("projected", type=Path)
    parser.add_argument("unreduced", type=Path)
    parser.add_argument("output", type=Path)
    args = parser.parse_args()
    result = dict(contextual=boost(args.projected), determinant=determinant_row(args.unreduced))
    args.output.write_text(json.dumps(result, indent=2, allow_nan=False) + "\n")
