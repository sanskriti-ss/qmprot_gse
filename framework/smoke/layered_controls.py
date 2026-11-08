# ABOUTME: Independently scores layered VQE starts and endpoints on a frozen molecular operator.
# ABOUTME: Compares the default initialization with angles that prepare the encoded HF determinant.
import argparse
from hashlib import sha256
import json
from pathlib import Path

import numpy as np
from openfermion import get_sparse_operator

from algorithms import get_algorithm
from experiments._common import CostEvalCounter
from experiments.experiment5_term_count_sweep import _prefix_hamiltonian
from experiments.provenance import load_problem


METHODS = ("hardware_efficient_vqe", "pennylane_vqe")


def entanglers(n, hardware):
    if hardware:
        return [(q, q + 1) for q in range(n - 1)] + ([(n - 1, 0)] if n > 2 else [])
    return [(q, q + 1) for parity in (0, 1) for q in range(parity, n - 1, 2)]


def cnot_index(index, n, control, target):
    return index ^ (((index >> (n - 1 - control)) & 1) << (n - 1 - target))


def simulate(reference, parameters, axes, hardware, layers):
    n = len(reference).bit_length() - 1
    blocks = np.asarray(parameters).reshape(layers + int(hardware), n, len(axes))
    state = reference.copy()
    indices = np.arange(len(reference))
    for block, rotations in enumerate(blocks):
        for wire, angles in enumerate(rotations):
            for axis, angle in zip(axes, angles):
                c, s = np.cos(angle / 2), np.sin(angle / 2)
                if axis == "X":
                    gate = [[c, -1j * s], [-1j * s, c]]
                elif axis == "Y":
                    gate = [[c, -s], [s, c]]
                else:
                    gate = np.diag([c - 1j * s, c + 1j * s])
                tensor = state.reshape((2,) * n)
                rotated = np.tensordot(gate, tensor, axes=(1, wire))
                state = np.moveaxis(rotated, 0, wire).reshape(-1)
        if block < layers:
            for control, target in entanglers(n, hardware):
                state = state[cnot_index(indices, n, control, target)]
    return state


def reference_angles(reference, layers, axes, hardware):
    """Invert the entangler permutation and use first-layer Y flips to supply its input."""
    n = len(reference).bit_length() - 1
    index = int(np.argmax(abs(reference)))
    if np.count_nonzero(reference) != 1:
        raise ValueError("This control requires a reference supported on a single basis state")
    source = index
    for control, target in reversed(entanglers(n, hardware) * layers):
        source = cnot_index(source, n, control, target)
    angles = np.zeros((layers + int(hardware), n, len(axes)))
    for wire in range(n):
        if (source ^ index) & (1 << (n - 1 - wire)):
            angles[0, wire, axes.index("Y")] = np.pi
    return angles.reshape(-1)


def measure(path, budgets, seeds, names, rotation_gates, terms):
    problem = load_problem(path)
    h = problem.hamiltonian
    if terms is not None:
        h = _prefix_hamiltonian(h, terms)
    n = h.n_qubits
    reference = h.cs_initial_state
    if reference is None:
        reference = np.zeros(2**n, dtype=complex)
        reference[((1 << h.molecule.n_electrons) - 1) << (n - h.molecule.n_electrons)] = 1
    matrix = get_sparse_operator(h.to_openfermion(), n_qubits=n)
    ground = float(np.linalg.eigvalsh(matrix.toarray())[0])
    hf = float(np.vdot(reference, matrix @ reference).real)
    rows = []
    for name in names:
        hardware = name == "hardware_efficient_vqe"
        axes = rotation_gates.replace("_", "").replace("R", "") if hardware else "XYZ"
        for seed in seeds:
            for budget in budgets:
                for start in ("default", "reference"):
                    solver = get_algorithm(name)(h, n_layers=2, rotation_gates=rotation_gates,
                                                  max_iterations=budget, random_seed=seed,
                                                  optimizer="COBYLA", convergence_threshold=1e-10)
                    solver.build_ansatz()
                    theta = (solver.get_initial_parameters() if start == "default"
                             else reference_angles(reference, 2, axes, hardware))
                    initial = simulate(reference, theta, axes, hardware, 2)
                    initial_energy = float(np.vdot(initial, matrix @ initial).real)
                    np.testing.assert_allclose(solver.cost_function(theta), initial_energy, atol=1e-9, rtol=0)
                    counter = CostEvalCounter()
                    counter.attach(solver)
                    try:
                        opt, energy = solver.optimize(initial_parameters=theta)
                    finally:
                        counter.detach(solver)
                    endpoint = simulate(reference, opt, axes, hardware, 2)
                    rescore = float(np.vdot(endpoint, matrix @ endpoint).real)
                    np.testing.assert_allclose(rescore, energy, atol=1e-9, rtol=0)
                    fidelity = float(abs(np.vdot(initial, reference))**2)
                    if start == "reference":
                        np.testing.assert_allclose([fidelity, initial_energy], [1, hf], rtol=0, atol=1e-9)
                    rows.append(dict(algorithm=name, axes=axes, seed=seed, budget=budget, calls=counter.count,
                                     start=start, initial_angles=theta.tolist(), final_angles=opt.tolist(),
                                     initial_hf_fidelity=fidelity, initial_energy=initial_energy,
                                     endpoint=energy, endpoint_rescore=rescore, error_to_ground=energy-ground,
                                     **solver.fit_diagnostics()))
    return dict(dataset=problem.molecule, input_sha256=sha256(path.read_bytes()).hexdigest(), n_qubits=n,
                terms=h.n_terms, layers=2, hf=hf, molecular_hf=problem.hf_energy,
                sector_casci=problem.casci_energy, ground=ground, rows=rows)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Compare and independently score two-layer molecular fits.")
    parser.add_argument("inputs", type=Path, nargs="+")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--budgets", nargs="+", type=int, default=[100, 1000])
    parser.add_argument("--seeds", nargs="+", type=int, default=[42])
    parser.add_argument("--algorithms", nargs="+", choices=METHODS, default=METHODS)
    parser.add_argument("--rotation-gates", choices=("RY", "RY_RZ"), default="RY_RZ")
    parser.add_argument("--terms", type=int)
    args = parser.parse_args()
    records = [measure(path, args.budgets, args.seeds, args.algorithms, args.rotation_gates, args.terms)
               for path in args.inputs]
    args.output.write_text(json.dumps(records, indent=2, allow_nan=False) + "\n")
