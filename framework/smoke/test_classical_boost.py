# ABOUTME: Exercises the classically boosted VQE Hadamard overlap measurement on a real simulator.
# ABOUTME: Compares the measured projection with an independent small Hamiltonian calculation.
import csv
from contextlib import redirect_stdout
from io import StringIO
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest

import numpy as np
import pennylane as qml
from pyscf import lib
from scipy.linalg import eigvalsh

from algorithms.vqe_CB import ClassicallyBoostedVQE
from active_space_truncation.run_pipeline import run_pipeline
from core.backend_manager import BackendConfig
from core.hamiltonian_loader import Molecule, QubitHamiltonian


class CountedOverlaps(ClassicallyBoostedVQE):
    def _hadamard_test(self, unitary, determinant):
        self.overlaps += 1
        return super()._hadamard_test(unitary, determinant)


class ClassicalBoost(unittest.TestCase):
    def test_contextual_preparation_reuses_the_native_matrix_for_matching_state_bytes(self):
        from algorithms.vqe_CB import _contextual_unitary

        rng = np.random.default_rng(1337)
        vectors = [np.eye(16)[11].astype(complex), rng.normal(size=16),
                   rng.normal(size=16) + 1j * rng.normal(size=16)]
        for vector in vectors:
            state = vector / np.linalg.norm(vector)
            args = (4, state.dtype.str, state.tobytes())
            original = qml.matrix(qml.StatePrep(state, wires=range(4)), wire_order=range(4))
            matrix = _contextual_unitary(*args)
            self.assertEqual(matrix.tobytes(), np.asarray(original).tobytes())
            self.assertIs(_contextual_unitary(*args), matrix)
            self.assertFalse(matrix.flags.writeable)

    def test_zero_matrix_elements_and_hf_overlap(self):
        molecule = Molecule("h2", "Hydrogen", 4, 2, -1.0, "", n_electrons=2)
        hamiltonian = QubitHamiltonian.from_dict({"ZIII": 0.7, "XIIX": 0.2}, molecule)
        parameters = np.linspace(-0.6, 0.7, 12)
        vqe = CountedOverlaps(hamiltonian, n_layers=1)
        vqe.build_ansatz()
        vqe.overlaps = 0
        vqe._classical_boost(parameters, vqe.cost_function(parameters))
        self.assertEqual(vqe.overlaps, 2)

    def test_requires_analytic_pure_state(self):
        molecule = Molecule("pair", "Two orbitals", 2, 1, -1., "", n_electrons=1)
        ham = QubitHamiltonian.from_dict({"ZZ": 1.}, molecule)
        for backend in (BackendConfig.statevector(2, n_shots=100),
                        BackendConfig.noisy(2, noise_model="phaseflip", noise_strength=.01)):
            with self.subTest(backend=backend), self.assertRaisesRegex(ValueError, "analytic.*statevector"):
                ClassicallyBoostedVQE(ham, n_layers=1, backend_config=backend).build_ansatz()

    def test_batch_reports_an_unavailable_overlap_device(self):
        framework = Path(__file__).resolve().parents[1]
        with tempfile.TemporaryDirectory() as folder:
            env = dict(os.environ, QMPROT_GEOMETRIES=str(framework / "smoke/geometries"),
                       PENNYLANE_DEVICE="unavailable.overlap", OMP_NUM_THREADS="1", OPENBLAS_NUM_THREADS="1")
            args = [sys.executable, "-m", "experiments.experiment5_term_count_sweep", "--molecules", "hydrogen",
                    "--algorithms", "cb_vqe", "--term-counts", "1", "--cs-target-qubits", "0",
                    "--max-iter", "32", "--no-plots", "--output-dir", folder]
            result = subprocess.run(args, cwd=framework, env=env, capture_output=True, text=True)
            self.assertNotEqual(result.returncode, 0, result.stdout + result.stderr)
            with (Path(folder) / "batch_all_results.csv").open() as table:
                row, = csv.DictReader(table)
            self.assertEqual(row["status"], "failed")
            self.assertIn("unavailable.overlap", row["notes"])
            self.assertIn("unavailable.overlap", result.stderr)

    def test_projected_energy_and_measured_overlap(self):
        molecule = Molecule("h2", "Hydrogen", 2, 2, -1.0, "", n_electrons=1)
        hamiltonian = QubitHamiltonian.from_dict({"ZI": 0.7, "XX": 0.2}, molecule)
        vqe = ClassicallyBoostedVQE(hamiltonian, n_layers=1)
        vqe.build_ansatz()
        parameters = np.linspace(0.1, 0.6, 6)
        unitary = vqe._ansatz_unitary(parameters)
        state = unitary[:, 0]
        hf = np.eye(4)[:, 2]
        basis = np.column_stack((hf, state))
        x = np.array([[0, 1], [1, 0]])
        matrix = 0.7 * np.kron(np.diag([1, -1]), np.eye(2)) + 0.2 * np.kron(x, x)

        measured = vqe._hadamard_test(unitary, 2)
        self.assertAlmostEqual(measured, state[2], places=12)
        vqe_energy = vqe.cost_function(parameters)
        self.assertAlmostEqual(vqe_energy, np.vdot(state, matrix @ state).real, places=12)
        expected = eigvalsh(basis.conj().T @ matrix @ basis, basis.conj().T @ basis)[0]
        self.assertAlmostEqual(vqe._classical_boost(parameters, vqe_energy), expected, places=12)
        self.assertLess(expected, vqe_energy)

    def test_rotated_complex_reference_and_nonconserving_terms(self):
        molecule = Molecule("pair", "Projected orbitals", 2, 1, -1., "", n_electrons=1)
        hamiltonian = QubitHamiltonian.from_dict({"ZI": .7, "IY": .4, "XI": .3}, molecule)
        reference = np.array([1, 2j, 0, -1j]) / np.sqrt(6)
        hamiltonian.cs_initial_state = reference
        vqe = CountedOverlaps(hamiltonian, n_layers=1)
        vqe.overlaps = 0
        vqe.build_ansatz()
        params = np.linspace(-.7, .4, 6)
        state = vqe._ansatz_unitary(params)[:, 0]
        x = np.array([[0, 1], [1, 0]])
        y = np.array([[0, -1j], [1j, 0]])
        matrix = .7 * np.kron(np.diag([1, -1]), np.eye(2)) + .4 * np.kron(np.eye(2), y) + .3 * np.kron(x, np.eye(2))
        basis = np.column_stack((reference, state))
        expected = eigvalsh(basis.conj().T @ matrix @ basis, basis.conj().T @ basis)[0]
        energy = vqe.cost_function(params)
        self.assertAlmostEqual(energy, np.vdot(state, matrix @ state).real, places=12)
        self.assertAlmostEqual(vqe._classical_boost(params, energy), expected, places=12)
        self.assertEqual(vqe.overlaps, 4)

    def test_physical_hydrogen_matches_dense_complex_projection(self):
        path = Path(__file__).resolve().parent / "geometries/hydrogen/hydrogen.h5"
        log = StringIO()
        with redirect_stdout(log), lib.temporary_env(lib.StreamObject, stdout=log):
            pipeline = run_pipeline("hydrogen", basis="sto-3g", h5_path=str(path), quiet=True)
        ham = pipeline["hamiltonian"].qubit_hamiltonian
        vqe = ClassicallyBoostedVQE(ham, n_layers=1)
        vqe.build_ansatz()
        matrix = qml.matrix(ham.to_pennylane(), wire_order=range(4))
        reference = np.eye(16)[:, 12]
        rng = np.random.default_rng(42)
        gaps = []
        for parameters in rng.uniform(-np.pi, np.pi, size=(3, 12)):
            state = vqe._ansatz_unitary(parameters)[:, 0]
            basis = np.column_stack((reference, state))
            h = basis.conj().T @ matrix @ basis
            s = basis.conj().T @ basis
            expected = eigvalsh(h, s)[0]
            gaps.append(abs(expected - eigvalsh(h.real, s.real)[0]))
            self.assertAlmostEqual(vqe._classical_boost(parameters, vqe.cost_function(parameters)), expected, places=12)
        self.assertGreater(max(gaps), 1e-5)

    def test_coincident_states_have_a_single_ritz_energy(self):
        molecule = Molecule("orbital", "Single orbital", 1, 1, -.7, "", n_electrons=1)
        ham = QubitHamiltonian.from_dict({"Z": .7}, molecule)
        vqe = ClassicallyBoostedVQE(ham, n_layers=1)
        vqe.build_ansatz()
        parameters = np.array([0., 0., .7])
        self.assertAlmostEqual(vqe._classical_boost(parameters, vqe.cost_function(parameters)), -.7)

    def test_nearly_parallel_states_still_span_two_dimensions(self):
        molecule = Molecule("orbital", "Single orbital", 1, 1, -1., "", n_electrons=1)
        ham = QubitHamiltonian.from_dict({"Z": -1.}, molecule)
        vqe = ClassicallyBoostedVQE(ham, n_layers=1)
        vqe.build_ansatz()
        parameters = np.array([1e-8, 0., 0.])
        self.assertAlmostEqual(vqe._classical_boost(parameters, vqe.cost_function(parameters)), -1.)
