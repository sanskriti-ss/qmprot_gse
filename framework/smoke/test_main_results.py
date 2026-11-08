# ABOUTME: Runs the public VQE CLI and compares each molecular energy export with a real H₂ solve.
# ABOUTME: Checks that batch failures preserve successful results and that plot JSON can be read back.
import csv
import json
import os
from pathlib import Path
import subprocess
import sys
from tempfile import TemporaryDirectory
import unittest

from core.results_manager import ResultsManager

FRAMEWORK = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(FRAMEWORK.parent))
from utils.csv_export import build_summary_table


class MainResults(unittest.TestCase):
    def launch(self, root, *args):
        command = [sys.executable, str(FRAMEWORK / "main.py"), "--hamiltonians-dir", str(FRAMEWORK / "smoke/geometries"),
                   "--active-space-basis", "sto-3g", "--n-layers", "1", "--max-iterations", "28",
                   "--results-dir", str(root / "results"), "--plots-dir", str(root / "plots"), *args]
        return subprocess.run(command, cwd=root, env={**os.environ, "MPLBACKEND": "Agg", "LOG_FILE": "",
                              "OMP_NUM_THREADS": "1", "OPENBLAS_NUM_THREADS": "1"}, text=True,
                              stdout=subprocess.PIPE, stderr=subprocess.STDOUT, timeout=120)

    def test_partial_batch_saves_hydrogen_and_reports_failure(self):
        with TemporaryDirectory() as tmp:
            root = Path(tmp)
            run = self.launch(root, "--all", "--molecules", "hydrogen", "--algorithms",
                              "iqcc_true_vqe", "hardware_efficient_vqe", "--backend-type", "noisy",
                              "--noise-model", "depolarizing", "--noise-strength", "0.01")
            self.assertNotEqual(run.returncode, 0, run.stdout)
            self.assertIn("True iQCC does not implement noise channels", run.stdout)
            self.assertIn("1 requested solver run(s) failed", run.stdout)
            self.assertNotIn("Traceback", run.stdout)
            self.assertNotIn("UserWarning", run.stdout)
            plot_json, = (root / "plots").glob("*/all_results.json")
            manager = ResultsManager(root / "results")
            result, = manager.load_results(plot_json)
            self.assertEqual(result.algorithm_name, "hardware_efficient_vqe")
            self.assertIn(result.algorithm_name, build_summary_table([result]))
            self.assertIn(f"{result.calculated_energy:.8f}", build_summary_table([result]))
            self.assertAlmostEqual(result.reference_energy, -1.136846575472, places=10)
            saved = json.loads(plot_json.read_text())
            self.assertEqual(saved[0]["calculated_energy"], result.calculated_energy)
            for path in (plot_json.with_suffix(".csv"), root / "plots/vqe_energies_summary.csv",
                         root / "results/results/csv/molecule_plots_summary.csv"):
                with path.open() as handle:
                    row, = csv.DictReader(handle)
                self.assertEqual(row["molecule"], "hydrogen")
                self.assertEqual(row["method"], result.algorithm_name)
                self.assertEqual(int(row["iterations"]), result.n_iterations)
                for key, value in (("vqe_energy", result.calculated_energy), ("exact_energy", result.reference_energy),
                                   ("hf_energy", result.hf_energy), ("error_vs_exact", abs(result.error)),
                                   ("error_vs_hf", result.calculated_energy - result.hf_energy)):
                    self.assertEqual(row[key], f"{value:.10f}")
            replay = self.launch(root, "--plot-only", "--results-file", str(plot_json),
                                 "--plots-dir", str(root / "replay"), "--no-save")
            self.assertEqual(replay.returncode, 0, replay.stdout)
            replay_csv, = (root / "replay").glob("*/all_results.csv")
            self.assertEqual(replay_csv.read_bytes(), plot_json.with_suffix(".csv").read_bytes())

    def test_success_and_empty_inventory(self):
        with TemporaryDirectory() as tmp:
            root = Path(tmp)
            run = self.launch(root, "--all-algorithms", "--molecule", "hydrogen",
                              "--algorithms", "hardware_efficient_vqe", "--no-plot")
            self.assertEqual(run.returncode, 0, run.stdout)
            self.assertNotIn("ERROR", run.stdout)
            empty = root / "empty"
            empty.mkdir()
            empty_run = self.launch(root, "--all-molecules", "--algorithm", "hardware_efficient_vqe",
                                    "--hamiltonians-dir", str(empty), "--no-plot")
            self.assertNotEqual(empty_run.returncode, 0, empty_run.stdout)
            self.assertIn("at least one molecule", empty_run.stdout)
