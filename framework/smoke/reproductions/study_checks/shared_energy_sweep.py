# ABOUTME: Measures determinant, dense, contextual-sector and prefix reference work on real glycine operators.
# ABOUTME: Records numeric outputs and array digests for comparison across scientific source trees.
import argparse
from dataclasses import replace
from hashlib import sha256
import json
from pathlib import Path
from time import perf_counter

import numpy as np

from algorithms import get_algorithm
from core.hf_verification import _build_hamiltonian_matrix, compute_hf_energy
from experiments.endpoint_sector import _physical_observables, _state
from experiments.experiment5_term_count_sweep import _HamiltonianPrefixes
from experiments.provenance import load_problem


def measure(operation, repetitions):
    elapsed = []
    for _ in range(repetitions):
        start = perf_counter()
        result = operation()
        elapsed.append(perf_counter() - start)
    return elapsed, result


def prefix_references(hamiltonian):
    prefixes = _HamiltonianPrefixes(hamiltonian)
    results = []
    for count in (10, 50, 200):
        prefix = prefixes.at(count)
        matrix = prefix.ordered_sparse_matrix()
        buffers = matrix.data.tobytes() + matrix.indices.tobytes() + matrix.indptr.tobytes()
        results.append(dict(energy=prefix.molecule.reference_energy, matrix=sha256(buffers).hexdigest()))
    return results


def run(unreduced, projected):
    output = {}
    for n, origin, terms in ((8, projected, 200), (12, unreduced, 40)):
        h = origin.hamiltonian
        determinant = replace(h, molecule=replace(h.molecule, n_electrons=min(n//2, 6)), n_terms=terms,
                              coefficients=h.coefficients[:terms], pauli_strings=h.pauli_strings[:terms],
                              cs_initial_state=None)
        for name, operation in (("determinant", compute_hf_energy), ("dense_matrix", _build_hamiltonian_matrix)):
            elapsed, value = measure(lambda: operation(determinant), 3 if n == 8 else 2)
            output[f"{name}_{n}q_{terms}terms"] = dict(seconds=elapsed, sha256=sha256(np.asarray(value).tobytes()).hexdigest())

    solver = get_algorithm("hardware_efficient_vqe")(projected.hamiltonian, n_layers=1)
    solver.build_ansatz()
    state = _state(solver, np.linspace(-.7, .6, solver.n_parameters), None)
    elapsed, values = measure(lambda: _physical_observables(projected, state), 7)
    output["all_sector_observables_8q"] = dict(seconds=elapsed, measurement=values)
    for problem in (unreduced, projected):
        h = problem.hamiltonian
        elapsed, values = measure(lambda: prefix_references(h), 3)
        output[f"prefix_references_and_endpoint_operators_{h.n_qubits}q"] = dict(seconds=elapsed, outputs=values)
    return output


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("unreduced", type=Path)
    parser.add_argument("projected", type=Path)
    parser.add_argument("output", type=Path)
    args = parser.parse_args()
    report = run(load_problem(args.unreduced), load_problem(args.projected))
    args.output.write_text(json.dumps(report, indent=2, allow_nan=False) + "\n")
