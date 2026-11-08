# ABOUTME: Compares each fixed-circuit scalar energy with its original PennyLane QNode at identical angles.
# ABOUTME: Records interleaved warm timings, binary64 output digests and the hash of each frozen input.
import argparse
from hashlib import sha256
import json
from pathlib import Path
from statistics import median
import sys
from time import perf_counter

import numpy as np


METHODS = ("hardware_efficient_vqe", "pennylane_vqe", "ucc_vqe", "qaoa_inspired_vqe",
           "nn_ae_vqe", "cb_vqe", "hva_vqe")


def check(hamiltonian, method):
    from algorithms import get_algorithm

    solver = get_algorithm(method)(hamiltonian, n_layers=2, random_seed=42)
    solver.build_ansatz()
    n = 4 if method == "hva_vqe" and hamiltonian.n_qubits == 12 else 10
    times = {"qnode": [], "objective": []}
    outputs = []
    for i, theta in enumerate(np.random.default_rng(42).uniform(-1, 1, (n, solver.n_parameters))):
        pair = {}
        order = ("qnode", "objective") if i % 2 else ("objective", "qnode")
        for kind in order:
            evaluate = solver.cost_fn if kind == "qnode" else solver.cost_function
            start = perf_counter()
            value = float(evaluate(theta))
            elapsed = perf_counter() - start
            pair[kind] = value.hex()
            if i:
                times[kind].append(elapsed)
        if pair["qnode"] != pair["objective"]:
            raise ValueError(f"Energy records differ for {method}, probe {i}: {pair}")
        outputs.append(pair["objective"])
    return dict(parameters=solver.n_parameters, vectors=n, energies=outputs,
                qnode_median_ms=median(times["qnode"]) * 1e3,
                objective_median_ms=median(times["objective"]) * 1e3,
                speedup=median(times["qnode"]) / median(times["objective"]))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("framework", type=Path)
    parser.add_argument("output", type=Path)
    parser.add_argument("inputs", type=Path, nargs="+")
    args = parser.parse_args()
    sys.path.insert(0, str(args.framework.resolve()))
    from experiments.provenance import load_problem

    records = []
    for path in args.inputs:
        problem = load_problem(path)
        row = dict(molecule=problem.molecule, qubits=problem.n_qubits_final, terms=problem.hamiltonian.n_terms,
                   input_sha256=sha256(path.read_bytes()).hexdigest(), methods={})
        for method in METHODS:
            row["methods"][method] = check(problem.hamiltonian, method)
        records.append(row)
        args.output.write_text(json.dumps(records, indent=2, allow_nan=False) + "\n")


if __name__ == "__main__":
    main()
