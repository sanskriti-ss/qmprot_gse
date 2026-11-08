# ABOUTME: Checks native initialization trials against the intended molecular final state and energy.
# ABOUTME: Exercises boosted Ritz fits and validates unsupported study combinations before chemistry.
import contextlib
import csv
import json
from io import StringIO
from pathlib import Path
import tempfile
import unittest

import numpy as np

from algorithms import get_algorithm
from experiments.barren_plateau import run_optimization_comparison, run_barren_plateau
from experiments.provenance import load_problem
from smoke.test_initialization import launch
from smoke.physics_acceptance import circuit_state, dense_hamiltonian, projections


class InitializationDispatch(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.input = load_problem(Path(__file__).parent / "reproductions/layered_vqe/hydrogen.npz")

    def test_cb_endpoints_are_ritz_energies(self):
        reference = np.eye(16)[:, 12]
        matrix = dense_hamiltonian(self.input.hamiltonian)
        for optimizer in ("COBYLA", "BAYESIAN"):
            with self.subTest(optimizer=optimizer), contextlib.redirect_stderr(StringIO()):
                row, = run_optimization_comparison(
                    self.input, algorithm="cb_vqe", n_layers=2, strategies=("small_random",), n_trials=1,
                    max_iterations=32 if optimizer == "COBYLA" else 5, optimizer=optimizer,
                    convergence_threshold=1e-10, learning_rate=.1, random_seed=42,
                    noise_model=None, noise_strength=0.)
                solver = get_algorithm("cb_vqe")(self.input.hamiltonian, n_layers=2)
                solver.build_ansatz()
                for angles, energy in ((row.initial_parameters, row.initial_energy),
                                       (row.optimal_parameters, row.final_energy)):
                    state = circuit_state(solver, np.array(angles), "cb_vqe")
                    expected = projections(state, matrix, reference)["complex"]
                    self.assertAlmostEqual(energy, expected, places=10)
                self.assertLess(row.final_energy, row.variational_energy - .1)

    def test_part_b_batch_saves_ritz_result_and_sector(self):
        with tempfile.TemporaryDirectory() as tmp:
            folder = Path(tmp)
            result = launch(folder, "--algorithm", "cb_vqe", "--optimizer", "COBYLA", "--max-iter", "32",
                            "--opt-layers", "2", "--opt-strategies", "small_random", "--n-trials", "1")
            log = result.stdout + result.stderr
            self.assertEqual(result.returncode, 0, log)
            for marker in ("[WARNING]", "[ERROR]", "Warning:", "Traceback"):
                self.assertNotIn(marker, log)
            with (folder / "all_molecules_optimization_results.csv").open() as stream:
                row, = csv.DictReader(stream)
            trial, = json.loads((folder / "hydrogen/trial_endpoints.json").read_text())
            self.assertEqual(trial["endpoint"]["state_kind"], "ritz_state")
            self.assertAlmostEqual(float(row["final_energy"]), trial["endpoint"]["rescored_energy"], places=9)
            self.assertAlmostEqual(float(row["wrong_electron_probability"]), trial["endpoint"]["wrong_electron_probability"])
            self.assertLess(float(row["final_energy"]), trial["variational_energy"] - .1)

    def test_growing_circuits_and_noisy_boost_fail_before_loading(self):
        for name in ("adapt_vqe", "qubit_adapt_vqe", "iqcc_true_vqe", "iqcc_inspired_vqe"):
            with self.subTest(name=name), self.assertRaisesRegex(ValueError, "fixed"):
                run_barren_plateau(molecule="missing", algorithm=name, skip_part_a=True, optimizer="COBYLA")
        with self.assertRaisesRegex(ValueError, "analytic pure"):
            run_barren_plateau(molecule="missing", algorithm="cb_vqe", skip_part_a=True, optimizer="BAYESIAN",
                               noise_model="phaseflip", noise_strength=.01)
