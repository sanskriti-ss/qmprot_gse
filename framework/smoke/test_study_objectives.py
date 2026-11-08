# ABOUTME: Verifies molecular shift-rule gradients, physical mixed-state executions and full optimizer histories.
# ABOUTME: Checks that incomplete molecule cohorts and closely spaced noise settings retain their identities.
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
import pennylane as qml
from pyscf import lib

from active_space_truncation.run_pipeline import run_pipeline
from experiments._common import build_vqe, make_backend_config, parameter_shift_gradient
from experiments.barren_plateau import _gradient_cost, run_barren_plateau
from experiments.noise_resilience import _optimize_one
from smoke.batch import FRAMEWORK


class StudyObjectives(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        geometry = FRAMEWORK / "smoke/geometries/hydrogen/hydrogen.h5"
        with contextlib.redirect_stdout(io.StringIO()) as log, lib.temporary_env(lib.StreamObject, stdout=log):
            cls.ham = run_pipeline("hydrogen", h5_path=str(geometry), basis="sto-3g", quiet=True)["hamiltonian"].qubit_hamiltonian

    def test_forward_cost_and_general_shift(self):
        for algorithm in ("hardware_efficient_vqe", "pennylane_vqe"):
            for noise in (None, "phaseflip"):
                with self.subTest(algorithm=algorithm, noise=noise):
                    config = make_backend_config(4, noise_model=noise, noise_strength=.04)
                    solver = build_vqe(self.ham, algorithm, n_layers=2, backend_config=config)
                    angles = np.random.default_rng(42).uniform(-.8, .8, solver.n_parameters)
                    fn = _gradient_cost(solver.cost_fn)
                    np.testing.assert_array_equal(fn(angles), solver.cost_fn(angles))
                    delta = np.eye(solver.n_parameters)[2] * 1e-5
                    independent = (solver.cost_fn(angles + delta) - solver.cost_fn(angles - delta)) / 2e-5
                    np.testing.assert_allclose(parameter_shift_gradient(fn, angles, [2], shift=.37),
                                               [independent], rtol=0, atol=2e-9)
                    for shift in (0., np.pi, -2*np.pi, np.inf):
                        with self.assertRaisesRegex(ValueError, "nonzero sine"):
                            parameter_shift_gradient(fn, angles, [2], shift=shift)

    def test_qaoa_requires_a_different_derivative(self):
        algorithm = "qaoa_inspired_vqe"
        solver = build_vqe(self.ham, algorithm, n_layers=2, supported_algorithms=(algorithm,))
        angles = np.linspace(-.27, .61, solver.n_parameters)
        delta = np.eye(solver.n_parameters) * 1e-5
        finite = [(solver.cost_fn(angles + d) - solver.cost_fn(angles - d))/2e-5 for d in delta]
        self.assertGreater(np.max(np.abs(parameter_shift_gradient(solver.cost_fn, angles) - finite)), .01)
        for kwargs in ({}, {"skip_part_a": True, "optimizer": "GradientDescent"}):
            with self.assertRaisesRegex(ValueError, "one Pauli rotation per parameter"):
                run_barren_plateau(molecule="missing-geometry", algorithm=algorithm, **kwargs)

    def test_mixed_state_optimizer_reuses_exact_physical_evaluations(self):
        config = make_backend_config(4, noise_model="depolarizing", noise_strength=.02)
        def solver():
            return build_vqe(self.ham, "hardware_efficient_vqe", n_layers=1, backend_config=config,
                             max_iterations=42, random_seed=42)
        raw, cached = solver(), solver()
        theta = np.linspace(-.31, .56, raw.n_parameters)
        with contextlib.redirect_stderr(io.StringIO()), qml.Tracker(raw.device) as direct:
            angles, energy = raw.optimize(initial_parameters=theta.copy())
        with contextlib.redirect_stderr(io.StringIO()), qml.Tracker(cached.device) as reused:
            optimum, value, history, iterations = _optimize_one(cached, theta)
        self.assertLess(reused.totals["executions"], direct.totals["executions"])
        np.testing.assert_array_equal(optimum, angles)
        self.assertEqual((value, history, iterations), (energy, raw.convergence_history, raw.iteration_count))

    def test_incomplete_cohorts_and_precise_noise_strengths(self):
        cases = {"barren_plateau_batch": ["--molecules", "hydrogen", "missing-geometry", "--layers", "1",
                  "--opt-layers", "1", "--n-samples", "2", "--n-trials", "1", "--max-iter", "1",
                  "--optimizer", "GradientDescent", "--bp-strategies", "small_random",
                  "--opt-strategies", "small_random", "--output-dir"],
                 "noise_resilience": ["--batch-molecules", "hydrogen", "missing-geometry", "--n-layers", "1",
                  "--max-iter", "14", "--noise-models", "phaseflip", "--noise-strengths", ".01234561", ".01234562",
                  "--batch-output-dir"]}
        for module, args in cases.items():
            with self.subTest(module=module), tempfile.TemporaryDirectory() as tmp:
                output = Path(tmp)
                run = subprocess.run([sys.executable, "-m", "experiments." + module, *args, tmp,
                                      "--cs-target-qubits", "0", "--no-plots"],
                                     cwd=FRAMEWORK, capture_output=True, text=True, env=dict(
                                         os.environ, QMPROT_GEOMETRIES=str(FRAMEWORK / "smoke/geometries")))
                self.assertNotEqual(run.returncode, 0)
                self.assertIn("cohort: missing-geometry", run.stderr)
                self.assertIn("H5 dataset not found", run.stderr)
                if module == "noise_resilience":
                    folder = output / "per_molecule/hydrogen"
                    parameters = json.loads((folder / "optimum_parameters.json").read_text())
                    self.assertEqual(set(parameters["theta_noisy_star"]), {"phaseflip@0.01234561", "phaseflip@0.01234562"})
                    for table in ("results.csv", "convergence_traces.csv"):
                        with (folder / table).open() as stream:
                            self.assertEqual({float(r["noise_strength"]) for r in csv.DictReader(stream)}, {0., .01234561, .01234562})
                    self.assertEqual(json.loads((output / "batch_run_config.json").read_text())["n_fail"], 1)
                else:
                    with (output / "per_molecule_summary.csv").open() as stream:
                        self.assertEqual([r["molecule"] for r in csv.DictReader(stream)], ["hydrogen", "missing-geometry"])
