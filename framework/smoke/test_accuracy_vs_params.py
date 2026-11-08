# ABOUTME: Exercises ADAPT parameter-count reports on the pinned hydrogen geometry.
# ABOUTME: Checks early convergence, per-budget cohorts and saved plots through a real batch.
import csv
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest

import numpy as np
from matplotlib.figure import Figure
from PIL import Image
from scipy.linalg import expm

from experiments.accuracy_vs_params import _plot_cohort_mean, run_accuracy_vs_params_batch
from experiments.provenance import load_problem
from smoke.physics_acceptance import PAULI, dense_hamiltonian


FRAMEWORK = Path(__file__).resolve().parents[1]


def read_rows(path):
    with path.open(newline="") as stream:
        return list(csv.DictReader(stream))


class AccuracyVsParams(unittest.TestCase):
    def run_study(self, directory, *args):
        env = dict(os.environ, QMPROT_GEOMETRIES=str(FRAMEWORK / "smoke/geometries"),
                   OMP_NUM_THREADS="1", OPENBLAS_NUM_THREADS="1", MKL_NUM_THREADS="1",
                   VECLIB_MAXIMUM_THREADS="1", MPLBACKEND="Agg")
        return subprocess.run(
            [sys.executable, "-m", "experiments.accuracy_vs_params", "--output-dir", str(directory),
             "--inner-max-iter", "100", *args], cwd=FRAMEWORK, env=env, text=True, capture_output=True,
        )

    def check_success(self, result):
        self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
        self.assertNotIn("Traceback", result.stderr)
        self.assertNotIn("UserWarning", result.stderr)

    def test_early_stop_keeps_requests_and_counts_each_molecule_once(self):
        with tempfile.TemporaryDirectory() as tmp:
            directory = Path(tmp)
            self.check_success(self.run_study(directory, "--molecules", "hydrogen"))
            molecule = directory / "hydrogen_qubit_adapt_vqe"
            trace = read_rows(molecule / "full_convergence.csv")
            maximum = len(trace)
            self.assertGreater(maximum, 0)
            self.assertLess(maximum, 10)
            cfg = json.loads((molecule / "run_config.json").read_text())
            self.assertEqual(cfg["total_cost_evals"], int(trace[-1]["cumulative_cost_evals"]) +
                             2 * (cfg["n_pool_operators"] - maximum))
            ham = load_problem(molecule / "inputs/hydrogen.npz").hamiltonian
            matrix = dense_hamiltonian(ham)
            generators = []
            for step in trace:
                label = json.loads(step["added_generator"])
                terms = {int(token[1:]): token[0] for token in label.split()}
                operator = np.eye(1)
                for wire in range(ham.n_qubits):
                    operator = np.kron(operator, PAULI[terms.get(wire, "I")])
                generators.append(operator)
                state = np.eye(16)[:, 0b1100]
                for angle, pauli in zip(json.loads(step["parameters"]), generators, strict=True):
                    state = expm(-1j * angle * pauli) @ state
                self.assertAlmostEqual(float(step["energy"]), np.vdot(state, matrix @ state).real, places=11)
                self.assertAlmostEqual(float(step["rescored_energy"]), float(step["energy"]), places=11)
                wrong_number = [i for i in range(len(state)) if i.bit_count() != 2]
                self.assertAlmostEqual(float(step["wrong_electron_probability"]), sum(abs(state[wrong_number])**2))
            expected = [1, 2, 3, 5, 10]
            summary = read_rows(directory / "summary_by_k.csv")
            self.assertEqual([int(r["k"]) for r in summary], expected)
            for record in summary:
                attained = int(record["k"]) <= maximum
                self.assertEqual(int(record["n_requested"]), 1)
                self.assertEqual(int(record["n_molecules"]), int(attained))
                self.assertEqual(int(record["n_stopped_early"]), int(not attained))
                self.assertEqual(int(record["n_failed"]), 0)
                if attained:
                    self.assertEqual(float(record["std_abs_error_vs_casci"]), 0)
                else:
                    self.assertEqual(record["mean_abs_error_vs_casci"], "")
            for result_file in (directory / "batch_all_results.csv", molecule / "results.csv"):
                rows = read_rows(result_file)
                self.assertEqual([int(r["k"]) for r in rows], expected)
                for row in rows:
                    requested = int(row["k"])
                    attained = min(requested, maximum)
                    self.assertEqual(int(row["attained_k"]), attained)
                    self.assertEqual(row["status"], "ok" if attained == requested else "stopped_early")
                    self.assertAlmostEqual(float(row["energy"]), float(trace[attained - 1]["energy"]))
                    self.assertEqual(row["cumulative_cost_evals"], trace[attained - 1]["cumulative_cost_evals"])
            for name in ("mean_abs_error_vs_k.png", "mean_cost_evals_vs_k.png", "overview_all_molecules.png"):
                with Image.open(directory / name) as plot:
                    plot.verify()

    def test_no_accepted_operator_has_no_exact_count_statistics(self):
        with tempfile.TemporaryDirectory() as tmp:
            directory = Path(tmp)
            self.check_success(self.run_study(directory, "--molecules", "hydrogen", "--gradient-threshold", "1e3"))
            rows = read_rows(directory / "batch_all_results.csv")
            self.assertEqual({r["status"] for r in rows}, {"stopped_early"})
            self.assertEqual({r["attained_k"] for r in rows}, {"0"})
            self.assertEqual(len({r["energy"] for r in rows}), 1)
            for r in rows:
                self.assertAlmostEqual(float(r["energy"]), -1.117505884204331)
            self.assertFalse(read_rows(directory / "hydrogen_qubit_adapt_vqe/full_convergence.csv"))
            for r in read_rows(directory / "summary_by_k.csv"):
                self.assertEqual((r["n_molecules"], r["n_stopped_early"], r["n_failed"]), ("0", "1", "0"))
                self.assertEqual(r["mean_abs_error_vs_casci"], "")

    def test_single_cli_and_partial_failure(self):
        with tempfile.TemporaryDirectory() as tmp:
            directory = Path(tmp)
            args = ("--no-plots", "--k-list", "1")
            self.check_success(self.run_study(directory / "single", "--molecule", "hydrogen", *args))
            row = read_rows(directory / "single/results.csv")[0]
            self.assertEqual(row["attained_k"], "1")
            self.assertAlmostEqual(float(row["energy"]), float(row["rescored_energy"]), places=11)
            self.assertLess(float(row["wrong_electron_probability"]), 1e-12)
            result = self.run_study(directory / "batch", "--molecules", "missing-input", "hydrogen", *args)
            self.assertEqual(result.returncode, 1)
            failures = json.loads((directory / "batch/batch_failures.json").read_text())
            self.assertEqual([f["molecule"] for f in failures], ["missing-input"])
            self.assertIn("H5 dataset not found:", failures[0]["error"])
            self.assertIn("FileNotFoundError: " + failures[0]["error"], result.stderr)
            summary = read_rows(directory / "batch/summary_by_k.csv")[0]
            self.assertEqual((summary["n_requested"], summary["n_molecules"], summary["n_failed"]),
                             ("2", "1", "1"))
            single_cfg = json.loads((directory / "single/run_config.json").read_text())
            batch_cfg = json.loads((directory / "batch/hydrogen_qubit_adapt_vqe/run_config.json").read_text())
            self.assertEqual(batch_cfg["random_seed"], single_cfg["random_seed"])
            missing = read_rows(directory / "batch/batch_all_results.csv")[0]
            self.assertEqual((missing["status"], missing["attained_k"], missing["energy"]), ("failed", "", ""))

    def test_rejects_duplicate_molecules(self):
        with self.assertRaisesRegex(ValueError, "once per batch"):
            run_accuracy_vs_params_batch(molecules=("hydrogen", " Hydrogen "))

    def test_plotted_dispersion_matches_reported_sd(self):
        summary = np.array([(1, 2, 3, .01, .008), (2, 1, 3, .001, .002)],
                           dtype=[("k", int), ("n_molecules", int), ("n_requested", int),
                                  ("mean_abs_error_vs_casci", float), ("std_abs_error_vs_casci", float)])
        ax = Figure().subplots()
        _plot_cohort_mean(ax, summary, "abs_error_vs_casci", "tab:red")
        for segment, row in zip(ax.collections[0].get_segments(), summary, strict=True):
            mean, sd = row["mean_abs_error_vs_casci"], row["std_abs_error_vs_casci"]
            np.testing.assert_allclose(segment[:, 1], [mean - sd, mean + sd])
        self.assertEqual([label.get_text() for label in ax.get_xticklabels()], ["1\n(2/3)", "2\n(1/3)"])


if __name__ == "__main__":
    unittest.main()
