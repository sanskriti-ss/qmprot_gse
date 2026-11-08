# ABOUTME: Rescores real VQE results against dense Hamiltonians and independently propagated circuit states.
# ABOUTME: Reports physical counterexamples, including particle leakage and complex projected-subspace energies.
"""Run from framework with QMPROT_GEOMETRIES=smoke/geometries python -m smoke.physics_acceptance report.json."""
import argparse
import contextlib
from collections import Counter
import hashlib
import json
from importlib.metadata import version
from pathlib import Path
import warnings

import numpy as np
import pennylane as qml
from pyscf import lib

from algorithms import ALGORITHMS
from experiments._common import load_small_hamiltonian

ATOL = 1e-8
PAULI = dict(I=np.eye(2), X=np.array([[0, 1], [1, 0]]),
             Y=np.array([[0, -1j], [1j, 0]]), Z=np.diag([1, -1]))


def dense_hamiltonian(ham):
    matrix = np.zeros((2**ham.n_qubits, 2**ham.n_qubits), dtype=complex)
    for coefficient, word in zip(ham.coefficients, ham.pauli_strings):
        product = np.array([[1.]])
        for letter in word:
            product = np.kron(product, PAULI[letter])
        matrix += coefficient * product
    np.testing.assert_allclose(matrix, matrix.conj().T, rtol=0, atol=1e-12)
    return matrix


def circuit_state(solver, params, algorithm):
    if algorithm == "adapt_vqe":
        circuit = solver._build_circuit(params)
    elif algorithm == "qubit_adapt_vqe":
        circuit = solver._build_circuit()
    else:
        circuit = solver.cost_fn
    circuit.construct((params,), {})
    wires = list(range(solver.n_qubits))
    state = np.zeros(2**len(wires), dtype=complex)
    state[0] = 1.
    for gate in circuit.qtape.operations:
        if gate.name == "StatePrep":
            state = np.asarray(gate.state_vector(wire_order=wires)).reshape(-1)
        else:
            state = qml.matrix(gate, wire_order=wires) @ state
    np.testing.assert_allclose(np.vdot(state, state), 1., rtol=0, atol=1e-10)
    return state


def projected_energy(projected, overlap):
    weights, vectors = np.linalg.eigh(overlap)
    keep = weights > 1e-12
    basis = vectors[:, keep] / np.sqrt(weights[keep])
    return float(np.linalg.eigvalsh(basis.conj().T @ projected @ basis)[0])


def projections(state, ham, determinant):
    basis = np.column_stack((determinant, state))
    overlap = basis.conj().T @ basis
    projected = basis.conj().T @ ham @ basis
    return dict(real=projected_energy(projected.real, overlap.real),
                complex=projected_energy(projected, overlap))


def leakage(state, n_electrons):
    return float(sum(abs(z)**2 for i, z in enumerate(state) if i.bit_count() != n_electrons))


def check_solver(algorithm, problem, ham, reference):
    solver = ALGORITHMS[algorithm](problem.hamiltonian, n_layers=2, max_operators=3,
                                   gradient_threshold=1e-8, max_iterations=40, random_seed=42)
    result = solver.run()
    energy = float(result.calculated_energy)
    exact = float(np.linalg.eigvalsh(ham)[0])
    row = dict(algorithm=algorithm, reported=energy, above_ground=energy-exact,
               params=np.asarray(result.optimal_parameters).tolist(), violations=[])
    if not np.isfinite(energy) or energy < exact - ATOL:
        row["violations"].append("reported energy below spectrum or nonfinite")
    reference_energy = float(np.vdot(reference, ham @ reference).real)
    row["encoded_reference_energy"] = reference_energy
    row["reported_hf_energy"] = result.hf_energy
    if result.hf_energy is not None and abs(result.hf_energy - reference_energy) > ATOL:
        row["violations"].append("reported HF energy differs from the encoded HF state")
    params = np.asarray(result.optimal_parameters)
    state = circuit_state(solver, params, algorithm)
    zero_state = circuit_state(solver, np.zeros_like(params), algorithm)
    row["zero_angle_reference_fidelity"] = float(abs(np.vdot(zero_state, reference))**2)
    row["zero_angle_energy"] = float(np.vdot(zero_state, ham @ zero_state).real)
    score = float(np.vdot(state, ham @ state).real)
    row["state_energy"] = score
    if problem.hamiltonian.molecule.n_electrons is not None and not problem.cs_reduced:
        row["off_particle_sector"] = leakage(state, problem.hamiltonian.molecule.n_electrons)
    if hasattr(solver, "selected_operators"):
        row["selected"] = [str(x) for x in solver.selected_operators]
    if algorithm == "cb_vqe":
        row["terminal_projection"] = projections(state, ham, reference)
        row["reported_plain_vqe"] = float(result.metadata["vqe_energy"])
        if abs(score - row["reported_plain_vqe"]) > ATOL:
            row["violations"].append("unboosted endpoint disagrees with original Hamiltonian")
        if abs(energy - row["terminal_projection"]["complex"]) > ATOL:
            row["violations"].append("boosted endpoint disagrees with complex Rayleigh-Ritz")
        angles = np.random.default_rng(42).uniform(-np.pi, np.pi, solver.n_parameters)
        trial = circuit_state(solver, angles, algorithm)
        control = projections(trial, ham, reference)
        control["reported"] = float(solver._classical_boost(angles, solver.cost_function(angles)))
        row["complex_state_control"] = control
        if abs(control["complex"] - control["reported"]) > ATOL:
            row["violations"].append("fixed complex trial disagrees with complex Rayleigh-Ritz")
    elif abs(energy - score) > ATOL:
        row["violations"].append("terminal state disagrees with reported energy")

    if algorithm == "adapt_vqe" and not problem.cs_reduced:
        solver.build_ansatz()
        vacuum = circuit_state(solver, np.array([]), algorithm)
        index = next(i for i, op in enumerate(solver.operator_pool) if op[0] == "double")
        solver.selected_operators = [index]
        appended = circuit_state(solver, np.zeros(1), algorithm)
        row["zero_angle_double"] = dict(operator=solver.operator_pool[index],
            fidelity=float(abs(np.vdot(vacuum, appended))**2),
            off_particle_sector=leakage(appended, problem.hamiltonian.molecule.n_electrons))
        if row["zero_angle_double"]["fidelity"] < 1. - ATOL:
            row["violations"].append("zero-angle excitation changes the state")
    return row


def run(destination):
    report = {}
    for molecule in ("hydrogen", "gly"):
        with destination.with_suffix(f".{molecule}.log").open("w") as log:
            with contextlib.redirect_stdout(log), contextlib.redirect_stderr(log), lib.temporary_env(lib.StreamObject, stdout=log):
                problem = load_small_hamiltonian(molecule, "sto-3g", 4)
                ham = dense_hamiltonian(problem.hamiltonian)
                entry = report[molecule] = dict(active_qubits=problem.n_qubits_active,
                    qubits=problem.n_qubits_final, cs_reduced=problem.cs_reduced,
                    terms=problem.hamiltonian.n_terms, casci=problem.casci_energy,
                    exact=float(np.linalg.eigvalsh(ham)[0]),
                    hamiltonian_sha256=hashlib.sha256(ham.astype("<c16").tobytes()).hexdigest(), rows=[])
                reference = problem.hamiltonian.cs_initial_state
                if reference is None:
                    occupied = problem.hamiltonian.molecule.n_electrons
                    reference = np.zeros(2**problem.n_qubits_final, dtype=complex)
                    reference[((1 << occupied) - 1) << (problem.n_qubits_final - occupied)] = 1.
                reference = np.asarray(reference).reshape(-1)
                entry["reference_state_sha256"] = hashlib.sha256(reference.astype("<c16").tobytes()).hexdigest()
                entry["solver_settings"] = dict(n_layers=2, max_operators=3, gradient_threshold=1e-8,
                                               max_iterations=40, random_seed=42, optimizer="COBYLA")
                entry["versions"] = {name: version(name) for name in ("numpy", "scipy", "pennylane", "pennylane-lightning", "pyscf")}
                for algorithm in ALGORITHMS:
                    with warnings.catch_warnings(record=True) as observed:
                        warnings.simplefilter("always")
                        try:
                            row = check_solver(algorithm, problem, ham, reference)
                        except Exception as error:
                            row = dict(algorithm=algorithm, violations=[f"{type(error).__name__}: {error}"])
                    row["warnings"] = dict(Counter(f"{w.category.__name__}: {w.message}" for w in observed))
                    entry["rows"].append(row)
                    destination.write_text(json.dumps(report, indent=2, allow_nan=False) + "\n")
    return report


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("destination", type=Path)
    args = parser.parse_args()
    results = run(args.destination)
    failed = [(m, row["algorithm"], reason) for m, entry in results.items() for row in entry["rows"]
              for reason in row["violations"]]
    for case in failed:
        print(*case, sep=" — ")
    raise SystemExit(bool(failed))
