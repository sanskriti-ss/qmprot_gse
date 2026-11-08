# ABOUTME: Compares the registered adaptive RY/CNOT circuit with Jordan–Wigner excitations on small real molecules.
# ABOUTME: Holds the outer solver and budget fixed, and checks matched single and double rotations against dense Fock-space exponentials.
"""Run from framework with python -m smoke.fermionic_adapt; reports JSON on standard output."""
import contextlib
from dataclasses import replace
import hashlib
import io
import json
from pathlib import Path

import numpy as np
import pennylane as qml
from openfermion import FermionOperator, get_sparse_operator, hermitian_conjugated, s_squared_operator
from pyscf import lib
from scipy.linalg import expm

from active_space_truncation.geometry import load_geometry_from_h5
from active_space_truncation.step1_orbitals import run_orbital_analysis
from active_space_truncation.step2_active_space import validate_active_space
from active_space_truncation.step3_hamiltonian import build_qubit_hamiltonian
from algorithms.vqe_adapt import AdaptVQE
from experiments._common import CostEvalCounter
from smoke.active_spaces import sector_ground_energy
from smoke.physics_acceptance import circuit_state, dense_hamiltonian, leakage


class OccupiedVirtualAdapt(AdaptVQE):
    """Registered circuit approximations with the same occupied-to-virtual pool as the JW control."""

    def build_ansatz(self):
        super().build_ansatz()
        singles, doubles = qml.qchem.excitations(self.hamiltonian.molecule.n_electrons, self.n_qubits)
        self.operator_pool = [("single", *wires) for wires in singles] + [("double", *wires) for wires in doubles]
        return self.operator_pool


class FermionicAdapt(OccupiedVirtualAdapt):
    """Analytic control with exact Jordan–Wigner excitations in an unprojected spin-orbital basis."""

    def __init__(self, hamiltonian, **kwargs):
        super().__init__(hamiltonian, **kwargs)
        if hamiltonian.cs_initial_state is not None or hamiltonian.molecule.n_electrons is None:
            raise ValueError("The fermionic control requires unprojected spin orbitals and a known electron count")
        if self.backend_config.is_noisy or self.backend_config.n_shots:
            raise ValueError("The dense fermionic control uses analytic noiseless energies")
        self.name = "fermionic_adapt_control"
        self.description = "Adaptive VQE with spin-projection-preserving fermionic excitations"
        self.excitation_implementation = "Jordan-Wigner"

    def _build_circuit(self, params):
        observable = self._energy_observable

        @qml.qnode(self.device)
        def circuit(angles):
            self._prepare_initial_state()
            for angle, index in zip(angles, self.selected_operators):
                name, *sites = self.operator_pool[index]
                if name == "single":
                    i, a = sites
                    qml.FermionicSingleExcitation(angle, wires=range(i, a+1))
                else:
                    i, j, a, b = sites
                    qml.FermionicDoubleExcitation(angle, wires1=range(i, j+1), wires2=range(a, b+1))
            return qml.expval(observable)

        return circuit


def excitation_matrix(sites, n):
    """Anti-Hermitian JW generator using PennyLane's half-angle orientation."""
    if len(sites) == 2:
        i, a = sites
        word = ((a, 1), (i, 0))
    else:
        i, j, a, b = sites
        word = ((b, 1), (a, 1), (j, 0), (i, 0))
    excitation = FermionOperator(word)
    return get_sparse_operator(excitation - hermitian_conjugated(excitation), n_qubits=n).toarray()


def small_cas(path):
    geometry = load_geometry_from_h5(str(path))
    diagnostics = run_orbital_analysis(geometry, basis="sto-3g")
    if not diagnostics.hf_converged or geometry.spin != 0:
        raise ValueError("A converged closed-shell HF state is required")
    homo = geometry.n_electrons // 2 - 1
    diagnostics = replace(diagnostics, proposed_active_indices=[homo, homo+1],
                          proposed_n_active_orbitals=2, proposed_n_active_electrons=2)
    active = validate_active_space(geometry, diagnostics)
    if not active.casci_converged:
        raise ValueError("CASCI did not converge")
    h = build_qubit_hamiltonian(geometry, active, diagnostics).qubit_hamiltonian
    return h, active.casci_energy, diagnostics.proposed_active_indices


SOLVER_SETTINGS = dict(max_operators=3, max_iterations=100, convergence_threshold=1e-9,
                       gradient_threshold=1e-8, random_seed=42)
ENERGY_EVALUATION_LIMIT = 300


def compare_molecule(path):
    h, casci, orbitals = small_cas(path)
    matrix = dense_hamiltonian(h)
    sector = sector_ground_energy(h, 2)
    np.testing.assert_allclose(sector, casci, atol=1e-9, rtol=0)
    reference = np.eye(16)[:, 0b1100]
    spin = get_sparse_operator(s_squared_operator(2), n_qubits=4).toarray()
    rows, gates = {}, []
    for label, solver_class in (("registered", AdaptVQE), ("approximate_matched_pool", OccupiedVirtualAdapt),
                                ("fermionic_matched_pool", FermionicAdapt)):
        solver = solver_class(h, **SOLVER_SETTINGS)
        counter = CostEvalCounter(max_evals=ENERGY_EVALUATION_LIMIT)
        counter.attach(solver)
        try:
            result = solver.run()
        finally:
            counter.detach(solver)
        state = circuit_state(solver, result.optimal_parameters, "adapt_vqe")
        selected = [solver.operator_pool[i] for i in solver.selected_operators]
        measured = float(np.vdot(state, matrix @ state).real)
        np.testing.assert_allclose(measured, result.calculated_energy, atol=1e-10, rtol=0)
        rows[label] = dict(pool=solver.operator_pool, selected=selected, angles=result.optimal_parameters.tolist(),
            energy=measured, error_mha=1000*(measured-casci), off_particle_sector=leakage(state, 2),
            spin_squared=float(np.vdot(state, spin @ state).real), cost_evals=counter.count,
            outer_iterations=result.n_iterations, history=result.convergence_history)
        if label == "registered":
            continue
        for name, sites in (("single", [0, 2]), ("double", [0, 1, 2, 3])):
            term = (name, *sites)
            solver.selected_operators = [solver.operator_pool.index(term)]
            for angle in (0., 0.4):
                trial = circuit_state(solver, np.array([angle]), "adapt_vqe")
                exact = expm((angle/2) * excitation_matrix(sites, 4)) @ reference
                overlap = float(abs(np.vdot(exact, trial))**2)
                if solver_class is FermionicAdapt:
                    np.testing.assert_allclose(overlap, 1., atol=1e-12, rtol=0)
                gates.append(dict(circuit=label, operator=term, angle=angle, fidelity_to_jw=overlap,
                    off_particle_sector=leakage(trial, 2), energy=float(np.vdot(trial, matrix @ trial).real)))
    return dict(active_orbitals=orbitals, n_active_electrons=2, n_terms=h.n_terms, hf=float(matrix[12,12].real),
                pauli_strings=h.pauli_strings, coefficients=h.coefficients.tolist(),
                casci=casci, sector_energy=sector, geometry_sha256=hashlib.sha256(path.read_bytes()).hexdigest(),
                hamiltonian_sha256=hashlib.sha256(matrix.astype("<c16").tobytes()).hexdigest(), runs=rows,
                matched_rotations=gates)


if __name__ == "__main__":
    geometries = Path(__file__).resolve().parent / "geometries"
    with contextlib.redirect_stdout(io.StringIO()) as log, contextlib.redirect_stderr(log), lib.temporary_env(lib.StreamObject, stdout=log):
        results = {m: compare_molecule(geometries / m / f"{m}.h5") for m in ("hydrogen", "gly")}
    print(json.dumps(dict(solver_settings=SOLVER_SETTINGS, energy_evaluation_limit=ENERGY_EVALUATION_LIMIT,
                          molecules=results), indent=2, allow_nan=False))
