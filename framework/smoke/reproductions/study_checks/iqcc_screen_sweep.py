# ABOUTME: Times true and inspired iQCC on frozen ordered molecular Hamiltonians in a specified framework checkout.
# ABOUTME: Records binary trajectories, logical counters and optional device checks of disjoint gradient supports.
import argparse
from contextlib import redirect_stdout
from hashlib import sha256
from io import StringIO
import json
import logging
from pathlib import Path
import sys
from time import perf_counter

import numpy as np


def run_fit(solver, inspect_screen):
    objective = solver.cost_function
    trace = []
    disjoint = 0
    max_disjoint = 0.

    def measure(theta):
        value = objective(theta)
        trace.append([np.ascontiguousarray(theta, dtype=np.float64).tobytes().hex(), float(value).hex()])
        return value

    solver.cost_function = measure
    if inspect_screen and solver.name == "iqcc_inspired_vqe":
        expectation = solver.compute_expectation

        def examine(observable, params):
            nonlocal disjoint, max_disjoint
            active = {w for op in solver.selected_operators for term in op.terms for w, _ in term}
            zero = all(any(p in "XY" and w not in active for w, p in word.items()) for word in observable.pauli_rep)
            value = expectation(observable, params)
            if zero:
                disjoint += 1
                max_disjoint = max(max_disjoint, abs(value))
                if abs(value) != 0.:
                    raise ValueError(f"Disjoint gradient measured {value.hex()}")
            return value

        if solver.hamiltonian.cs_initial_state is not None and np.count_nonzero(solver.hamiltonian.cs_initial_state) != 1:
            raise ValueError("Inspecting fixed reference wires requires a computational basis ket")
        solver.compute_expectation = examine

    start = perf_counter()
    with redirect_stdout(StringIO()):
        result = solver.run()
    elapsed = perf_counter() - start
    evidence = dict(energy=float(result.calculated_energy).hex(), hf=float(result.hf_energy).hex(),
                    angles=[float(x).hex() for x in result.optimal_parameters],
                    history=[float(x).hex() for x in result.convergence_history], metadata=result.metadata,
                    objective_trace_sha256=sha256(json.dumps(trace).encode()).hexdigest(), objective_count=len(trace))
    if inspect_screen:
        evidence.update(measured_disjoint=disjoint, max_abs_measured_disjoint=max_disjoint)
    return elapsed, evidence


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("framework", type=Path, help="Framework checkout to measure")
    parser.add_argument("input", type=Path, help="Prepared molecular NPZ or frozen four-qubit glycine JSON")
    parser.add_argument("output", type=Path)
    parser.add_argument("--terms", type=int, required=True)
    parser.add_argument("--steps", type=int, default=2)
    parser.add_argument("--repeats", type=int, default=3)
    parser.add_argument("--methods", nargs="+", choices=["true", "inspired"], default=["true", "inspired"])
    parser.add_argument("--inspect-screen", action="store_true", help="Measure disjoint gradients on an original solver")
    args = parser.parse_args()
    sys.path[:0] = [str(args.framework.resolve()), str(Path(__file__).resolve().parents[3])]
    from algorithms import get_algorithm
    from experiments.experiment5_term_count_sweep import _prefix_hamiltonian
    from experiments.provenance import load_problem
    from smoke.iqcc_control import problem

    logging.disable(logging.INFO)
    ham = problem(args.input) if args.input.suffix == ".json" else load_problem(args.input).hamiltonian
    if not 0 < args.terms <= ham.n_terms or args.steps < 0 or args.repeats < 1:
        raise ValueError("Invalid requested workload")
    if args.terms != ham.n_terms:
        ham = _prefix_hamiltonian(ham, args.terms)
    output = dict(input_sha256=sha256(args.input.read_bytes()).hexdigest(),
                  qubits=ham.n_qubits, terms=ham.n_terms, steps=args.steps, seed=42, methods={})
    for method in args.methods:
        times = []
        expected = None
        cls = get_algorithm(f"iqcc_{method}_vqe")
        for _ in range(args.repeats):
            solver = cls(ham, max_operators=args.steps, gradient_threshold=1e-8, random_seed=42)
            elapsed, evidence = run_fit(solver, args.inspect_screen)
            if expected is not None and expected != evidence:
                raise ValueError(f"Repeated {method} solver changed its scientific record")
            expected = evidence
            times.append(elapsed)
        output["methods"][method] = dict(seconds=times, scientific=expected)
    args.output.write_text(json.dumps(output, indent=2, allow_nan=False) + "\n")


if __name__ == "__main__":
    main()
