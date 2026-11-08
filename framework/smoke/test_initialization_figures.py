# ABOUTME: Checks cross-molecule initialization summaries against each trial's molecular reference.
# ABOUTME: Verifies plotted trial lines and common-step means against genuine multi-trial output.
import csv
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest

from matplotlib.figure import Figure
import numpy as np
from PIL import Image

from experiments.barren_plateau_batch import _plot_strategy_trials
from smoke.batch import FRAMEWORK
from experiments.noise_resilience import plot_noise_resilience_multi_molecule


def rows(path):
    with path.open() as stream:
        return list(csv.DictReader(stream))


class InitializationFigures(unittest.TestCase):
    def execute(self, args):
        result = subprocess.run([sys.executable, *args], cwd=FRAMEWORK, capture_output=True, text=True,
                                env=dict(os.environ, QMPROT_GEOMETRIES=str(FRAMEWORK / "smoke/geometries")))
        log = result.stdout + result.stderr
        self.assertEqual(result.returncode, 0, log)
        for token in ("[ERROR]", "[WARNING]", "Warning:", "Traceback"):
            self.assertNotIn(token, log)

    def test_molecule_relative_summary(self):
        with tempfile.TemporaryDirectory() as tmp:
            directory = Path(tmp)
            self.execute([str(FRAMEWORK.parent / "run_barren_plateau_part_b_all_molecules.py"),
                          "--molecules", "hydrogen", "gly", "--optimizer", "COBYLA", "--cs-target-qubits", "0",
                          "--opt-layers", "1", "--n-trials", "1", "--opt-strategies", "small_random",
                          "--max-iter", "32", "--output-root", tmp])
            trials = rows(directory / "all_molecules_optimization_results.csv")
            summary, = rows(directory / "all_molecules_strategy_summary.csv")
            self.assertEqual([r["molecule"] for r in trials], ["hydrogen", "gly"])
            errors = []
            for record in trials:
                config = json.loads((directory / record["molecule"] / "run_config.json").read_text())
                self.assertAlmostEqual(float(record["casci_energy"]), config["casci_energy"], places=9)
                errors.append(abs(float(record["final_energy"]) - config["casci_energy"]))
            self.assertEqual((summary["n_runs"], summary["n_molecules"]), ("2", "2"))
            np.testing.assert_allclose([float(summary[key]) for key in
                                        ("mean_final_abs_error_vs_casci", "std_final_abs_error_vs_casci",
                                         "min_final_abs_error_vs_casci")],
                                       [np.mean(errors), np.std(errors), min(errors)], atol=1e-9, rtol=0)
            for name in ("all_molecules_final_energy_by_strategy", "all_molecules_convergence_by_strategy"):
                with Image.open(directory / (name + ".png")) as plot:
                    plot.verify()

    def test_trial_lines_use_recorded_trials(self):
        with tempfile.TemporaryDirectory() as tmp:
            directory = Path(tmp)
            self.execute(["-m", "experiments.barren_plateau_batch", "--molecules", "hydrogen",
                          "--optimizer", "GradientDescent", "--cs-target-qubits", "0", "--layers", "1",
                          "--bp-strategies", "small_random", "--n-samples", "3", "--opt-layers", "1",
                          "--n-trials", "2", "--opt-strategies", "small_random", "--max-iter", "2",
                          "--output-dir", tmp])
            summary, = rows(directory / "per_molecule_summary.csv")
            self.assertEqual(summary["n_gradient_samples"], "3")
            per_trial = rows(directory / "hydrogen/convergence_traces.csv")
            aggregate = rows(directory / "convergence_traces_all.csv")
            ax = Figure().subplots()
            _plot_strategy_trials(ax, aggregate, "small_random", "blue")
            first, second, average = ax.lines
            series = []
            for trial, line in enumerate((first, second)):
                values = [float(r["energy"]) for r in per_trial if int(r["trial"]) == trial]
                self.assertEqual(len(values), 3)
                np.testing.assert_equal(line.get_xdata(), [0, 1, 2])
                np.testing.assert_allclose(line.get_ydata(), values, atol=1e-9, rtol=0)
                series.append(values)
            np.testing.assert_allclose(average.get_ydata(), np.mean(series, axis=0), atol=1e-9, rtol=0)

    def test_hf_chart_labels_reduced_hydrogen_and_checks_cohort(self):
        with tempfile.TemporaryDirectory() as tmp:
            directory = Path(tmp)
            self.execute([str(FRAMEWORK.parent / "run_barren_plateau_part_b_all_molecules.py"), "--molecules",
                          "hydrogen", "--optimizer", "COBYLA", "--cs-target-qubits", "2", "--opt-layers", "1",
                          "--n-trials", "1", "--opt-strategies", "small_random", "--max-iter", "32",
                          "--output-root", tmp, "--no-plots"])
            script = str(FRAMEWORK.parent / "plot_hf_verification_from_part_b.py")
            self.execute([script, "--output-root", tmp])
            config = json.loads((directory / "hydrogen/run_config.json").read_text())
            self.assertEqual((config["n_qubits_active"], config["n_qubits_final"]), (4, 2))
            record, = rows(directory / "hf_verification_all_molecules.csv")
            self.assertEqual(float(record["molecular_hf_energy"]), config["hf_energy"])
            self.assertEqual(float(record["active_space_casci_energy"]), config["casci_energy"])
            batch = directory / "batch_manifest.json"
            manifest = json.loads(batch.read_text())
            manifest.update(molecules_succeeded=[], molecules_failed={"hydrogen": "study failed"})
            batch.write_text(json.dumps(manifest))
            failed = subprocess.run([sys.executable, script, "--output-root", tmp], capture_output=True, text=True)
            self.assertNotEqual(failed.returncode, 0)
            self.assertIn("ValueError: Incomplete Part B molecule cohort", failed.stderr)

    def test_noise_figures_use_one_row_per_cell(self):
        with tempfile.TemporaryDirectory() as tmp:
            directory = Path(tmp)
            self.execute(["-m", "experiments.noise_resilience", "--batch-molecules", "hydrogen", "--cs-target-qubits",
                          "0", "--n-layers", "1", "--max-iter", "24", "--noise-models", "phaseflip", "depolarizing",
                          "--noise-strengths", "0", "0.02", "--batch-output-dir", tmp, "--no-per-molecule-plots"])
            figures = {p.stem for p in directory.glob("*.png")}
            self.assertEqual(figures, {"heatmap_param_drift_l2", "heatmap_param_drift_rms",
                                       "heatmap_param_drift_cosine", "heatmap_energy_gap_clean_noisy",
                                       "bars_reference_strength"})
            for path in directory.glob("*.png"):
                with Image.open(path) as plot:
                    plot.verify()
            table = directory / "batch_all_results.csv"
            text = table.read_text().splitlines()
            table.write_text("\n".join(text + [text[-1]]) + "\n")
            with self.assertRaisesRegex(ValueError, "Duplicate noise figure cell"):
                plot_noise_resilience_multi_molecule(table, directory)


if __name__ == "__main__":
    unittest.main()
