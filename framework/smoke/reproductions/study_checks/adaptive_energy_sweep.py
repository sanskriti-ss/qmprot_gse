# ABOUTME: Compares the original and support-filtered native Lightning energies on real adaptive circuits.
# ABOUTME: Measures unreduced and contextual glycine probes and records the actual retained term counts.
import argparse
from collections import Counter
from hashlib import sha256
import json
from pathlib import Path
from time import perf_counter

import numpy as np

from algorithms.vqe_adapt import AdaptVQE
from algorithms.vqe_qubit_adapt import QubitAdaptVQE
from core.backend_manager import analytic_energy
from core.pauli_measurement import adaptive_state_measurement
from experiments.provenance import load_problem


def probe(path, cls):
    loaded = load_problem(path)
    solver = cls(loaded.hamiltonian, random_seed=42)
    solver.build_ansatz()
    full_terms = len(solver._energy_observable.pauli_rep)
    select = adaptive_state_measurement(solver.n_qubits, solver._energy_observable)
    if select is None:
        raise ValueError("The installed Lightning runtime is outside the single-thread Pauli measurement range")
    counts = Counter()
    times = Counter()
    pairs = Counter()
    digest = sha256()

    def compare(state, expectation, full):
        started = perf_counter()
        expected = expectation(full)
        times["full_native"] += perf_counter() - started
        counts["logical"] += 1
        def score(terms):
            started = perf_counter()
            result = expectation(terms)
            times["selected_native"] += perf_counter() - started
            pairs[(np.count_nonzero(state), len(terms.get_ops()))] += 1
            counts["native_calls"] += 1
            return result
        started = perf_counter()
        measured = select(state, score, full)
        times["selected_including_lookup"] += perf_counter() - started
        if float(measured).hex() != float(expected).hex():
            raise ValueError(f"Energies differ at {solver.selected_operators}: {measured.hex()} vs {expected.hex()}")
        digest.update(np.float64(expected).tobytes())
        return measured

    energy = analytic_energy(solver.cost_fn, solver._energy_observable, measure_state=compare)
    number = 160 if cls is QubitAdaptVQE else 110
    sample = np.linspace(0, len(solver.operator_pool) - 1, min(number, len(solver.operator_pool)), dtype=int)
    for index in sample:
        solver.selected_operators = [index]
        for angle in (1e-5, -1e-5):
            energy(np.array([angle]))
    for steps in (3, 6, 12):
        solver.selected_operators = np.linspace(0, len(solver.operator_pool) - 1, steps, dtype=int).tolist()
        for seed in (42, 43):
            energy(np.random.default_rng(seed).uniform(-.9, 1.2, size=steps))
    if counts["logical"] != 2 * len(sample) + 6:
        raise ValueError("The full native circuit bypassed its analytic measurement control")
    return dict(input_sha256=sha256(path.read_bytes()).hexdigest(), qubits=solver.n_qubits, terms=full_terms,
                pool_size=len(solver.operator_pool), energy_sha256=digest.hexdigest(), counts=dict(counts),
                seconds=dict(times), populated_basis_states_and_retained_terms=[
                    dict(populated=int(size), retained=int(terms), calls=num) for (size, terms), num in sorted(pairs.items())])


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("full_glycine", type=Path)
    parser.add_argument("output", type=Path)
    args = parser.parse_args()
    contextual = Path(__file__).resolve().parents[1] / "layered_vqe/contextual_gly.npz"
    results = {}
    for name, path in (("unreduced", args.full_glycine), ("contextual", contextual)):
        results[name] = {cls.__name__: probe(path, cls) for cls in (QubitAdaptVQE, AdaptVQE)}
    args.output.write_text(json.dumps(results, indent=2, allow_nan=False) + "\n")
