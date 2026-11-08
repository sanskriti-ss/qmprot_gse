# ABOUTME: Checks iQCC dressing and original-Hamiltonian circuit energies against independent dense rotations.
# ABOUTME: Exercises the two hydrogen solvers and the glycine prefix that violates the historical bound.
import contextlib
import io
import json
from pathlib import Path
import unittest
import warnings

import numpy as np
from openfermion import QubitOperator, get_sparse_operator
from pyscf import lib
from scipy.linalg import expm

from active_space_truncation.run_pipeline import run_pipeline
from algorithms.vqe_iqcc_inspired import iQCC_inspired_VQE
from algorithms.vqe_iqcc_true import iQCC_true_VQE, _dress, _rotation_terms
from core.backend_manager import BackendConfig
from core.hamiltonian_loader import Molecule, QubitHamiltonian
from core.iqcc_helpers import of_to_pennylane
from experiments._common import CostEvalCounter
from smoke.physics_acceptance import dense_hamiltonian


class IQCCScreening(unittest.TestCase):
    def test_pauli_observables_require_finite_real_coefficients(self):
        import pennylane as qml
        commutator = 1j * (QubitOperator("Z0") * QubitOperator("X0")
                           - QubitOperator("X0") * QubitOperator("Z0"))
        observable = of_to_pennylane(commutator + QubitOperator((), .3 + 0j))
        self.assertTrue(np.isrealobj(observable.coeffs))
        np.testing.assert_allclose(qml.matrix(observable), np.array([[.3, 2j], [-2j, .3]]))
        for term in ("Y0", ()):
            for coefficient in (1 + 1e-30j, float("nan"), float("inf")):
                with self.subTest(term=term, coefficient=coefficient), self.assertRaisesRegex(ValueError, "finite real"):
                    of_to_pennylane(QubitOperator(term, coefficient))

    def check_rotation_history(self, solver, result, reference):
        h = dense_hamiltonian(solver.hamiltonian)
        lower = np.linalg.eigvalsh(h)[0]
        angles = result.optimal_parameters
        generators = result.metadata["selected_generators"]
        self.assertEqual(len(angles), result.n_parameters)
        self.assertEqual(len(generators), result.n_iterations)
        self.assertEqual(len(result.convergence_history), len(angles))
        self.assertEqual(result.metadata["energy_measurement"], "analytic")
        expected = float(np.vdot(reference, h @ reference).real)
        for step, reported in enumerate(result.convergence_history, 1):
            state = reference.copy()
            for word, angle in reversed(list(zip(generators[:step], angles[:step]))):
                matrix = get_sparse_operator(QubitOperator(word), n_qubits=solver.n_qubits).toarray()
                state = expm(-0.5j * angle * matrix) @ state
            energy = float(np.vdot(state, h @ state).real)
            self.assertLessEqual(energy, expected + 1e-11)
            self.assertGreaterEqual(energy, lower - 1e-11)
            self.assertAlmostEqual(energy, reported, places=11)
            expected = energy
        self.assertAlmostEqual(expected, result.calculated_energy, places=11)
        self.assertAlmostEqual(solver.cost_function(angles), expected, places=11)

    def test_dressing_and_tangent_match_dense_unitary(self):
        h = (0.21 * QubitOperator(()) + 0.37 * QubitOperator("X0 Z1")
             - 0.66 * QubitOperator("Y0") + 0.12 * QubitOperator("Z0 Y1"))
        p = QubitOperator("X0 Z1")
        hm, pm = (get_sparse_operator(op, n_qubits=2).toarray() for op in (h, p))
        anti, slope = _rotation_terms(h, p)
        np.testing.assert_allclose(get_sparse_operator(slope, n_qubits=2).toarray(),
                                   0.5j * (pm @ hm - hm @ pm), atol=1e-15, rtol=0)
        for angle in (0., -0.57, 2.31):
            unitary = expm(-0.5j * angle * pm)
            target = unitary.conj().T @ hm @ unitary
            dressed = get_sparse_operator(_dress(h, anti, slope, angle), n_qubits=2).toarray()
            np.testing.assert_allclose(dressed, target, atol=1e-14, rtol=0)
            np.testing.assert_allclose(np.linalg.eigvalsh(dressed), np.linalg.eigvalsh(hm), atol=1e-14)

    def test_single_qubit_angle_and_noncommuting_replay_order(self):
        molecule = Molecule("one", "one", 1, 1, -1., "", n_electrons=1)
        solver = iQCC_true_VQE(QubitHamiltonian.from_dict({"X": 1.}, molecule), bound=np.pi)
        result = solver.run()
        self.assertEqual(result.metadata["selected_generators"], ["Y0"])
        self.assertAlmostEqual(result.optimal_parameters[0], np.pi/2)
        self.assertAlmostEqual(result.calculated_energy, -1.)
        self.assertTrue(result.converged)

        solver = iQCC_true_VQE(QubitHamiltonian.from_dict({"Y": 1.}, molecule))
        solver.build_ansatz()
        solver.selected_operators = ["X0", "Z0"]
        angles = np.array([0.7, -0.38])
        self.assertAlmostEqual(solver.cost_function(angles), np.sin(angles[0]))
        self.assertNotAlmostEqual(solver.cost_function(angles), np.sin(angles[0]) * np.cos(angles[1]))
        with self.assertRaisesRegex(ValueError, "one angle"):
            solver.cost_function(angles[:1])

    def test_nonstationary_hydrogen_trajectories(self):
        path = Path(__file__).resolve().parent / "geometries/hydrogen/hydrogen.h5"
        log = io.StringIO()
        with contextlib.redirect_stdout(log), lib.temporary_env(lib.StreamObject, stdout=log):
            problem = run_pipeline("hydrogen", basis="sto-3g", h5_path=str(path), quiet=True)
        hamiltonian = problem["hamiltonian"].qubit_hamiltonian
        state = np.zeros(2**hamiltonian.n_qubits, dtype=complex)
        state[0b0011], state[0b1100] = -0.48 + 0.36j, 0.8
        hamiltonian.cs_initial_state = state
        for algorithm in (iQCC_true_VQE, iQCC_inspired_VQE):
            with self.subTest(algorithm=algorithm.__name__):
                solver = algorithm(hamiltonian, max_operators=2, gradient_threshold=1e-3,
                                   random_seed=42, backend_config=BackendConfig.statevector(4))
                counter = CostEvalCounter()
                counter.attach(solver)
                try:
                    with contextlib.redirect_stdout(log), warnings.catch_warnings(record=True) as observed:
                        warnings.simplefilter("always")
                        result = solver.run()
                finally:
                    counter.detach(solver)
                self.assertEqual(counter.count, result.metadata["n_molecular_energy_evals"])
                self.assertEqual(result.metadata["n_screen_sweeps"], 2)
                self.assertGreater(result.metadata["n_screen_expectations"], 0)
                self.assertFalse(observed)
                if algorithm is iQCC_true_VQE:
                    self.assertEqual(result.n_iterations, 2)
                    self.assertEqual(result.metadata["n_angle_expectations"], 2)
                    self.assertEqual(result.metadata["n_commutator_builds"], 2 * len(solver.operator_pool))
                    self.check_rotation_history(solver, result, state)
                    self.assertGreater(abs(solver.cost_function(np.zeros(2))-result.calculated_energy), 1e-5)
                else:
                    np.testing.assert_allclose(result.convergence_history, [-0.6960877439972046], atol=1e-10, rtol=0)
                    self.assertAlmostEqual(result.calculated_energy, -0.6960877439972046, places=10)
                    self.assertEqual(solver.selected_operators, [QubitOperator("Z0")])
                    self.assertEqual(result.metadata["selected_generators"], ["Z0"])
                    self.assertEqual(result.n_iterations, 1)
                    self.assertTrue(result.converged)
                    self.assertEqual(result.metadata["n_commutator_builds"], len(solver.operator_pool))
                    self.assertEqual(result.metadata["n_screen_expectations"], 120)
                    self.assertGreater(counter.count, result.n_iterations + 1)
                    supplied = np.array([0.73])
                    self.assertEqual(solver.cost_function(supplied), float(solver.cost_fn(supplied)))
                    self.assertNotAlmostEqual(solver.cost_function(supplied), result.calculated_energy)
                    solver.max_operators = 0
                    with contextlib.redirect_stdout(log):
                        empty = solver.run()
                    self.assertFalse(empty.converged)
                    self.assertEqual(empty.convergence_history, [])
                    self.assertEqual(empty.metadata["selected_generators"], [])
                    self.assertEqual(empty.metadata["n_molecular_energy_evals"], 1)

    def test_glycine_prefix_and_repeated_run(self):
        folder = Path(__file__).resolve().parent / "reproductions/iqcc_glycine"
        data = json.loads((folder / "glycine.json").read_text())
        observed = json.loads((folder / "observations.json").read_text())
        count = observed["used_terms"]
        state = np.array([complex(*pair) for pair in data["state"]])
        molecule = Molecule("gly", "glycine", 4, count, observed["exact_prefix_energy"], "")
        h = QubitHamiltonian(molecule, np.array(data["coefficients"][:count]), data["pauli_strings"][:count], 4,
                             count, cs_initial_state=state)
        solver = iQCC_true_VQE(h, **observed["requested_params"])
        result = solver.run()
        self.assertGreater(result.n_iterations, 0)
        self.check_rotation_history(solver, result, state)
        self.assertGreater(result.calculated_energy, observed["endpoint"] + 0.011)
        solver.max_operators = 0
        self.check_rotation_history(solver, solver.run(), state)

    def test_noisy_rejection_and_sampled_label(self):
        h = QubitHamiltonian.from_dict({"Z": 1.}, Molecule("one", "one", 1, 1, -1., "", n_electrons=1))
        for algorithm in (iQCC_true_VQE, iQCC_inspired_VQE):
            with self.assertRaisesRegex(ValueError, "does not implement noise"):
                algorithm(h, backend_config=BackendConfig.noisy(1, "phaseflip", 0.1))
        solver = iQCC_true_VQE(h, backend_config=BackendConfig.statevector(1, n_shots=25), max_operators=0)
        result = solver.run()
        self.assertEqual(result.calculated_energy, -1.)
        self.assertEqual(result.metadata["energy_measurement"], "sampled")
