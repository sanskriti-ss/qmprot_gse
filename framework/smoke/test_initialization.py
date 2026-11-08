# ABOUTME: Checks molecular initialization gradients against an independent dense H₂ circuit.
# ABOUTME: Verifies paired Bayesian trials and failure reporting through the Part B batch launcher.
import contextlib
import csv
import io
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest

import numpy as np
from openfermion import get_sparse_operator
from pyscf import lib

from active_space_truncation.run_pipeline import run_pipeline
from algorithms.vqe_hardware_efficient import HardwareEfficientVQE
from core.base_vqe import wrap_parameters_into_bounds
from experiments._common import LoadedHamiltonian, parameter_shift_gradient
from experiments.barren_plateau import measure_gradient_variance
from smoke.study_checks import dense_energy


FRAMEWORK = Path(__file__).resolve().parents[1]
GEOMETRIES = FRAMEWORK / "smoke/geometries"


def launch(folder, *args):
    return subprocess.run(
        [sys.executable, str(FRAMEWORK.parent / "run_barren_plateau_part_b_all_molecules.py"),
         "--molecules", "hydrogen", "--cs-target-qubits", "0", "--output-root", str(folder),
         "--no-plots", *args], cwd=FRAMEWORK, env=dict(os.environ, QMPROT_GEOMETRIES=str(GEOMETRIES)),
        capture_output=True, text=True,
    )


class InitializationStudy(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        log = io.StringIO()
        with contextlib.redirect_stdout(log), lib.temporary_env(lib.StreamObject, stdout=log):
            molecule = run_pipeline("hydrogen", basis="sto-3g", quiet=True,
                                    h5_path=str(GEOMETRIES / "hydrogen/hydrogen.h5"))
        chemistry = molecule["hamiltonian"]
        cls.loaded = LoadedHamiltonian(chemistry.qubit_hamiltonian, "hydrogen", "sto-3g", 4, 4, False)
        cls.matrix = get_sparse_operator(chemistry.openfermion_qubit_op, n_qubits=4).toarray()

    def test_nonzero_gradients_and_variance(self):
        draws = 5
        rows = measure_gradient_variance(self.loaded, algorithm="hardware_efficient_vqe", layers=(1, 2),
                                         strategies=("small_random",), n_samples=draws, gradient_index=None,
                                         max_gradient_probes=3, random_seed=42)
        for row, probes in zip(rows, ((0, 4, 7), (0, 6, 11)), strict=True):
            solver = HardwareEfficientVQE(self.loaded.hamiltonian, n_layers=row.n_layers)
            solver.build_ansatz()
            gradients = []
            for trial in range(draws):
                rng = np.random.default_rng(42 + 1009 * trial)
                angles = rng.uniform(-np.pi/8, np.pi/8, row.n_parameters)
                delta = 1e-5 * np.eye(row.n_parameters)[list(probes)]
                gradient = [(dense_energy(self.matrix, angles + d) - dense_energy(self.matrix, angles - d))
                            / (2e-5) for d in delta]
                self.assertAlmostEqual(dense_energy(self.matrix, angles), solver.cost_function(angles), places=12)
                np.testing.assert_allclose(parameter_shift_gradient(solver.cost_fn, angles, indices=probes),
                                           gradient, rtol=0, atol=1e-9)
                gradients.append(gradient)
            values = np.asarray(gradients)
            variances = ((values - values.mean(axis=0))**2).sum(axis=0) / draws
            magnitudes = np.abs(values).mean(axis=0)
            indices, saved_variances, saved_magnitudes = zip(*row.per_parameter)
            self.assertEqual(indices, probes)
            np.testing.assert_allclose(saved_variances, variances, rtol=0, atol=1e-10)
            np.testing.assert_allclose(saved_magnitudes, magnitudes, rtol=0, atol=1e-9)
            self.assertAlmostEqual(row.var_grad, variances.mean(), places=10)
            self.assertAlmostEqual(row.mean_abs_grad, magnitudes.mean(), places=9)

    def test_gradient_trials_are_paired_across_depth_and_strategy_rosters(self):
        def collect(layers, strategies):
            rows = measure_gradient_variance(self.loaded, algorithm="pennylane_vqe", layers=layers,
                                             strategies=strategies, n_samples=3, gradient_index=0,
                                             max_gradient_probes=1, random_seed=17)
            return {(row.n_layers, row.strategy): row.per_parameter for row in rows}

        solo = collect((2,), ("small_random",))
        mixed = collect((1, 2), ("near_identity", "small_random"))
        self.assertEqual(solo[2, "small_random"], mixed[2, "small_random"])

    def test_bayesian_trials(self):
        with tempfile.TemporaryDirectory() as tmp:
            for layers, backend, budget in ((1, "gp", 14), (5, "forest", 27)):
                folder = Path(tmp) / backend
                strategies = ("small_random", "near_identity")
                result = launch(folder, "--optimizer", "BAYESIAN", "--max-iter", str(budget),
                                "--opt-layers", str(layers), "--opt-strategies", *strategies, "--n-trials", "2")
                log = result.stdout + result.stderr
                self.assertEqual(result.returncode, 0, log)
                for marker in ("[WARNING]", "[ERROR]", "Warning:", "Traceback", "falling back"):
                    self.assertNotIn(marker, log)
                self.assertEqual(log.count(f"Bayesian optimizer backend={backend},"), 4)
                with (folder / "all_molecules_optimization_results.csv").open() as stream:
                    rows = list(csv.DictReader(stream))
                with (folder / "all_molecules_convergence_traces.csv").open() as stream:
                    trace = list(csv.DictReader(stream))
                self.assertEqual([(r["strategy"], int(r["trial"])) for r in rows],
                                 [(s, t) for s in strategies for t in range(2)])
                for row in rows:
                    trial = int(row["trial"])
                    size = 4 * (layers + 1)
                    width = np.pi / 8 if row["strategy"] == "small_random" else 0.05
                    angles = np.random.default_rng(42 + 1009 * trial).uniform(-width, width, size)
                    energy = dense_energy(self.matrix, angles)
                    history = [float(r["energy"]) for r in trace
                               if (r["strategy"], r["trial"]) == (row["strategy"], row["trial"])]
                    self.assertEqual(len(history), budget)
                    self.assertEqual(int(row["n_iterations"]), budget)
                    self.assertAlmostEqual(history[0], energy, places=9)
                    self.assertAlmostEqual(float(row["initial_energy"]), energy, places=9)
                    self.assertEqual(float(row["final_energy"]), min(history))

    def test_bayesian_start_at_box_edges(self):
        angles = np.random.default_rng(42).uniform(-0.8, 0.8, 8)
        angles[:3] = [np.pi - 1e-5, np.nextafter(np.pi, 0), np.nextafter(-np.pi, 0)]
        wrapped = wrap_parameters_into_bounds(angles, -np.pi, np.pi)
        np.testing.assert_array_equal(wrapped, angles)
        angles[3:5] = [np.pi, 3 * np.pi - 1e-5]
        wrapped = wrap_parameters_into_bounds(angles, -np.pi, np.pi)
        self.assertTrue(np.all((wrapped >= -np.pi) & (wrapped < np.pi)))
        self.assertAlmostEqual(dense_energy(self.matrix, angles), dense_energy(self.matrix, wrapped), places=14)

    def test_failed_batch_exits_nonzero(self):
        for args, folders in (((), ("",)), (("--noise-model", "phaseflip", "--noise-strengths", "0", "0.01"),
                                            ("none_p0", "phaseflip_p0.01"))):
            with self.subTest(args=args), tempfile.TemporaryDirectory() as tmp:
                result = launch(tmp, "--optimizer", "invalid", "--max-iter", "14", "--opt-layers", "1",
                                "--opt-strategies", "small_random", "--n-trials", "1", *args)
                self.assertEqual(result.returncode, 1, result.stdout + result.stderr)
                self.assertEqual(result.stdout.count("Unknown solver invalid"), len(folders))
                for folder in folders:
                    with (Path(tmp) / folder / "batch_manifest.json").open() as stream:
                        manifest = json.load(stream)
                    self.assertEqual(manifest["n_rows"], 0)
                    self.assertEqual(manifest["molecules_failed"], {"hydrogen": "Unknown solver invalid"})


if __name__ == "__main__":
    unittest.main()
