# ABOUTME: Checks every single and double in the small physical ADAPT control against full Fock-space unitaries.
# ABOUTME: Uses the published hydrogen geometry to verify charge preservation and the variational endpoint.
import contextlib
import io
from pathlib import Path
import unittest

import numpy as np
import pennylane as qml
from pyscf import lib
from scipy.linalg import expm

from algorithms import ALGORITHMS
from core.backend_manager import BackendConfig
from core.hamiltonian_loader import Molecule, QubitHamiltonian
from experiments._common import CostEvalCounter
from smoke.fermionic_adapt import excitation_matrix, FermionicAdapt, small_cas
from smoke.physics_acceptance import PAULI, circuit_state, dense_hamiltonian, leakage


class FermionicRotations(unittest.TestCase):
    def test_observable_reuse_preserves_requests_and_sampling(self):
        mol = Molecule("control", "One electron", 2, 2, -1., "", n_electrons=1)
        ham = QubitHamiltonian.from_dict({"ZI": .7, "XX": .2}, mol)
        for shots in (0, 64):
            solver = ALGORITHMS["adapt_vqe"](ham, backend_config=BackendConfig.statevector(2, n_shots=shots))
            solver.build_ansatz()
            solver.selected_operators = [0]
            params = np.array([.47])
            with qml.Tracker(solver.device) as tracker:
                scores = [solver.cost_function(params) for _ in range(2)]
            self.assertGreaterEqual(tracker.totals["executions"], 2)
            if shots == 0:
                state = circuit_state(solver, params, "adapt_vqe")
                expected = np.vdot(state, dense_hamiltonian(ham) @ state).real
                np.testing.assert_allclose(scores, expected, rtol=0, atol=1e-14)
                self.assertEqual(scores[0].hex(), scores[1].hex())
            else:
                self.assertIsNone(solver._energy_observable)

    def test_inconclusive_sample_on_hydrogen(self):
        path = Path(__file__).resolve().parent / "geometries/hydrogen/hydrogen.h5"
        with contextlib.redirect_stdout(io.StringIO()) as log, lib.temporary_env(lib.StreamObject, stdout=log):
            ham, _, _ = small_cas(path)
        state = np.eye(16)[:, 12]
        generator = np.kron(np.kron(PAULI["Y"], PAULI["X"]), np.kron(PAULI["X"], PAULI["X"]))
        H = dense_hamiltonian(ham)
        gradient = np.vdot(state, 1j * (generator @ H - H @ generator) @ state).real
        self.assertGreater(abs(gradient), .35)
        for cap, threshold, sampled in ((1, 1e-8, True), (12, 1., False)):
            solver = ALGORITHMS["qubit_adapt_vqe"](ham, max_operators=1, random_seed=42,
                                                   max_gradient_candidates=cap, gradient_threshold=threshold)
            result = solver.run()
            self.assertEqual(result.n_parameters, 0)
            self.assertEqual(result.converged, not sampled)
            self.assertEqual(result.metadata["stopped_on_sample"], sampled)
            self.assertAlmostEqual(result.calculated_energy, np.vdot(state, H @ state).real, places=12)

    def test_empty_and_exhausted_pools(self):
        for name in ("adapt_vqe", "qubit_adapt_vqe"):
            for n_qubits, word in ((1, "Z"), (2, "II")):
                with self.subTest(algorithm=name, hamiltonian=word):
                    mol = Molecule("control", "Control", n_qubits, 1, -1., "", n_electrons=1)
                    ham = QubitHamiltonian.from_dict({word: 1.}, mol)
                    solver = ALGORITHMS[name](ham, max_operators=6, max_iterations=16, gradient_threshold=0.)
                    result = solver.run()
                    self.assertTrue(result.converged)
                    self.assertEqual(len(set(solver.selected_operators)), result.n_parameters)
                    self.assertLessEqual(result.n_parameters, len(solver.operator_pool))
                    self.assertAlmostEqual(result.calculated_energy, -1. if n_qubits == 1 else 1., places=13)

    def test_single_evaluation_records_no_accepted_operator(self):
        path = Path(__file__).resolve().parent / "geometries/hydrogen/hydrogen.h5"
        with contextlib.redirect_stdout(io.StringIO()) as log, lib.temporary_env(lib.StreamObject, stdout=log):
            h, _, _ = small_cas(path)
        for name in ("adapt_vqe", "qubit_adapt_vqe"):
            with self.subTest(algorithm=name):
                solver = ALGORITHMS[name](h, max_operators=2, random_seed=42)
                counter = CostEvalCounter(max_evals=1)
                counter.attach(solver)
                try:
                    result = solver.run()
                finally:
                    counter.detach(solver)
                self.assertEqual(counter.count, 1)
                self.assertEqual(solver.selected_operators, [])
                field = "selected_operators_str" if name == "qubit_adapt_vqe" else "selected_operators"
                self.assertEqual(result.metadata[field], [])
                self.assertEqual(result.metadata["n_operators_selected"], 0)
                self.assertEqual(result.metadata["evaluation_budget"], "gradient screening")
                self.assertEqual(result.n_iterations, 0)
                self.assertEqual(result.n_parameters, 0)
                self.assertEqual(len(result.optimal_parameters), 0)
                self.assertEqual(result.convergence_history, [])

    def test_hydrogen_pool_rotations_and_endpoint(self):
        path = Path(__file__).resolve().parent / "geometries/hydrogen/hydrogen.h5"
        with contextlib.redirect_stdout(io.StringIO()) as log, lib.temporary_env(lib.StreamObject, stdout=log):
            h, casci, _ = small_cas(path)
        solver = FermionicAdapt(h, max_operators=3, max_iterations=100, gradient_threshold=1e-8,
                                convergence_threshold=1e-9, random_seed=42)
        self.assertNotIn(solver.name, ALGORITHMS)
        solver.build_ansatz()
        self.assertEqual(solver.operator_pool, [("single", 0, 2), ("single", 1, 3), ("double", 0, 1, 2, 3)])
        for index, (_, *sites) in enumerate(solver.operator_pool):
            solver.selected_operators = [index]
            for angle in (0., 0.4, -0.83):
                with self.subTest(sites=sites, angle=angle):
                    params = np.array([angle])
                    circuit = solver._build_circuit(params)
                    circuit.construct((params,), {})
                    unitary = qml.matrix(circuit.qtape.operations[-1], wire_order=range(4))
                    target = expm(angle/2 * excitation_matrix(sites, 4))
                    np.testing.assert_allclose(unitary, target, atol=1e-13, rtol=0)
                    self.assertLess(leakage(unitary[:, 12], 2), 1e-20)
        with contextlib.redirect_stderr(log):
            result = solver.run()
        state = circuit_state(solver, result.optimal_parameters, "adapt_vqe")
        self.assertAlmostEqual(result.calculated_energy, np.vdot(state, dense_hamiltonian(h) @ state).real, places=11)
        self.assertAlmostEqual(result.calculated_energy, casci, places=10)
        self.assertEqual(result.metadata["excitation_implementation"], "Jordan-Wigner")
