# ABOUTME: Checks that repeated study settings cannot count the same molecule or seeded trial twice.
# ABOUTME: Exercises the native trainability and initialization runners and retrospective cohort summaries.
import csv
import json
from pathlib import Path
import tempfile
import unittest

from experiments.barren_plateau import run_barren_plateau
from experiments.barren_plateau_batch import run_barren_plateau_batch
from experiments.cost_to_accuracy import analyze_existing
from experiments.trainability import run_trainability, run_trainability_all_molecules
from smoke.test_initialization import launch


class StudyAxes(unittest.TestCase):
    def test_trainability_rejects_duplicate_settings_before_creating_results(self):
        with tempfile.TemporaryDirectory() as temp:
            output = Path(temp) / "study"
            for key, value in (("algorithms", ("qubit_adapt_vqe",) * 2), ("param_targets", (10, 10))):
                for runner, destination in ((run_trainability, "output_dir"),
                                            (run_trainability_all_molecules, "output_root")):
                    with self.subTest(runner=runner.__name__, key=key), self.assertRaisesRegex(ValueError, key):
                        runner(**{key: value, destination: output})
                    self.assertFalse(output.exists())

    def test_initialization_rejects_duplicate_trials_and_gradient_groups(self):
        with tempfile.TemporaryDirectory() as temp:
            output = Path(temp) / "study"
            for key, value in (("bp_strategies", ("zeros",) * 2), ("opt_strategies", ("zeros",) * 2),
                               ("layers", (1, 1)), ("molecules", ("hydrogen",) * 2)):
                for runner in (run_barren_plateau_batch, run_barren_plateau):
                    if key == "molecules" and runner is run_barren_plateau:
                        continue
                    with self.subTest(runner=runner.__name__, key=key), self.assertRaisesRegex(ValueError, key):
                        runner(output_dir=output, **{key: value})
                    self.assertFalse(output.exists())
            native = launch(output, "--opt-strategies", "zeros", "zeros")
            self.assertNotEqual(native.returncode, 0)
            self.assertIn("opt_strategies entry and supply each only once", native.stderr)
            self.assertFalse(output.exists())

    def test_reanalysis_checks_identity_before_emitting_a_cohort(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            mol = root / "hydrogen"
            mol.mkdir()
            (mol / "run_config.json").write_text(json.dumps({"hf_energy": -1.1, "casci_energy": -1.12}))
            with (mol / "results.csv").open("w", newline="") as stream:
                writer = csv.writer(stream)
                writer.writerow(["algorithm", "actual_n_params", "target_n_params", "n_cost_evals",
                                 "final_energy", "error_vs_casci"])
                writer.writerows([["qubit_adapt_vqe", 1, 1, 10, -1.115, .005]] * 2)
            with self.assertRaisesRegex(ValueError, "Duplicate trainability observation.*hydrogen.*qubit_adapt_vqe"):
                analyze_existing(root, root / "analysis")
            self.assertFalse((root / "analysis").exists())

    def test_initialization_noise_sweep_preserves_each_probability_and_seeded_trial(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp) / "sweep"
            for flags, message in ((("--noise-strengths", ".01", ".010"), "supply each only once"),
                                   (("--noise-strengths", "0", ".01"), "requires a noise model"),
                                   (("--noise-strength", "nan"), "between zero and one")):
                result = launch(root, *flags)
                self.assertNotEqual(result.returncode, 0)
                self.assertIn(message, result.stderr)
                self.assertFalse(root.exists())
            strengths = ["0.010000001", "0.010000002"]
            result = launch(root, "--noise-strengths", *strengths, "--noise-model", "phaseflip",
                            "--opt-strategies", "small_random", "--opt-layers", "1",
                            "--optimizer", "COBYLA", "--max-iter", "4", "--n-trials", "2")
            self.assertEqual(result.returncode, 0, result.stderr)
            for strength in strengths:
                directory = root / f"phaseflip_p{strength}"
                config = json.loads((directory / "batch_manifest.json").read_text())
                self.assertEqual(config["noise_strength"], float(strength))
                self.assertEqual(config["n_rows"], 2)
                endpoints = json.loads((directory / "hydrogen" / "trial_endpoints.json").read_text())
                self.assertEqual([row["seed"] for row in endpoints], [42, 1051])
