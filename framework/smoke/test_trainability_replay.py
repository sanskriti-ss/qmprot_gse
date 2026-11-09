# ABOUTME: Checks scientific replay of saved trainability fits on the published H₂ geometry.
# ABOUTME: Exercises damaged fit parameters, operator identities, thresholds and case coverage.
import csv
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest

from experiments.replay_trainability import replay


class TrainabilityReplay(unittest.TestCase):
    def test_replay_and_damaged_records(self):
        framework = Path(__file__).resolve().parents[1]
        with tempfile.TemporaryDirectory() as tmp:
            directory = Path(tmp)
            proc = subprocess.run([
                sys.executable, "-m", "experiments.trainability", "--molecule", "hydrogen",
                "--cs-target-qubits", "0", "--algorithms", "hardware_efficient_vqe", "qubit_adapt_vqe",
                "--param-targets", "2", "--eval-budget", "64", "--max-iter", "30", "--no-plots",
                "--output-dir", str(directory),
            ], env=dict(os.environ, QMPROT_GEOMETRIES=str(framework / "smoke/geometries")),
               cwd=framework, capture_output=True, text=True)
            self.assertEqual(proc.returncode, 0, proc.stdout + proc.stderr)
            self.assertNotRegex(proc.stdout + proc.stderr, r"\[WARNING\]|Warning:|Traceback")
            report = replay(directory)
            self.assertEqual(report["endpoints"], 2)
            self.assertGreater(report["crossing_witnesses"], 0)
            config = json.loads((directory / "run_config.json").read_text())
            with (directory / "results.csv").open() as stream:
                for record in csv.DictReader(stream):
                    self.assertEqual(float(record["hf_energy"]), config["hf_energy"])
                    self.assertEqual(float(record["casci_energy"]), config["casci_energy"])
            path = directory / "endpoints.jsonl"
            original = path.read_text().splitlines()

            def mutate(key, value, algorithm="qubit_adapt_vqe"):
                rows = [json.loads(line) for line in original]
                next(row for row in rows if row["algorithm"] == algorithm)[key] = value
                path.write_text("".join(json.dumps(row) + "\n" for row in rows))

            mutate("generators", ["IIII"])
            with self.assertRaisesRegex(ValueError, "generator"):
                replay(directory)
            mutate("parameters", [1., 1.], "hardware_efficient_vqe")
            with self.assertRaisesRegex(ValueError, "terminal energy"):
                replay(directory)
            mutate("input_sha256", "0"*64)
            with self.assertRaisesRegex(ValueError, "incorrectly bound"):
                replay(directory)
            table = directory / "results.csv"
            with table.open() as stream:
                rows = list(csv.DictReader(stream))
            rows[0]["accepted_evals_to_1.6mHa"] = "40"
            path.write_text("\n".join(original) + "\n")
            with table.open("w", newline="") as stream:
                writer = csv.DictWriter(stream, rows[0].keys())
                writer.writeheader()
                writer.writerows(rows)
            with self.assertRaisesRegex(ValueError, "CSV accepted"):
                replay(directory)
            path.write_text("")
            with self.assertRaisesRegex(ValueError, "Incomplete"):
                replay(directory)

    def test_batch_replay_checks_settings_and_aggregate_including_resumed_molecules(self):
        framework = Path(__file__).resolve().parents[1]
        with tempfile.TemporaryDirectory() as tmp:
            directory = Path(tmp)
            command = [sys.executable, "-m", "experiments.trainability", "--molecules", "hydrogen", "glu",
                       "--algorithms", "hardware_efficient_vqe", "--param-targets", "2", "--eval-budget", "16",
                       "--no-plots", "--output-root", str(directory)]
            env = dict(os.environ, QMPROT_GEOMETRIES=str(framework / "smoke/tiny_geometries"))
            for args in (command, [*command, "--resume"]):
                result = subprocess.run(args, cwd=framework, env=env, capture_output=True, text=True)
                self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
                self.assertNotRegex(result.stdout + result.stderr, r"\[WARNING\]|Warning:|Traceback")
                self.assertEqual(set(replay(directory)), {"hydrogen", "glu"})
                with (directory / "all_molecules_results.csv").open() as stream:
                    for record in csv.DictReader(stream):
                        config = json.loads((directory / record["molecule"] / "run_config.json").read_text())
                        self.assertEqual(float(record["hf_energy"]), config["hf_energy"])
                        self.assertEqual(float(record["casci_energy"]), config["casci_energy"])
            # Exercise the public plotting CLIs against a fresh/resumed native batch CSV.
            for view in ("additional_02_trainability_energy", "additional_03_trainability_runtime",
                         "additional_05_trainability_combined"):
                output = directory / (view + ".png")
                plotted = subprocess.run([
                    sys.executable, str(framework.parent / "prof_figures" / view / "generate.py"),
                    "--input", str(directory / "all_molecules_results.csv"), "--target", "2",
                    "--label", "Integration test", "--output", str(output),
                ], cwd=framework, env=env, capture_output=True, text=True)
                self.assertEqual(plotted.returncode, 0, plotted.stdout + plotted.stderr)
                self.assertTrue(output.is_file())
                self.assertTrue(output.with_suffix(".pdf").is_file())
            path = directory / "batch_manifest.json"
            manifest = json.loads(path.read_text())
            self.assertEqual(manifest["molecules_succeeded"], ["hydrogen", "glu"])
            self.assertEqual(manifest["molecules_resumed"], ["hydrogen", "glu"])
            path.write_text(json.dumps(dict(manifest, optimizer="BAYESIAN")))
            with self.assertRaisesRegex(ValueError, "optimizer"):
                replay(directory)
            path.write_text(json.dumps(manifest))
            path = directory / "all_molecules_results.csv"
            with path.open() as stream:
                rows = list(csv.DictReader(stream))
            rows[1]["wrong_electron_probability"] = "0.123"
            with path.open("w", newline="") as stream:
                writer = csv.DictWriter(stream, rows[0])
                writer.writeheader()
                writer.writerows(rows)
            with self.assertRaisesRegex(ValueError, "aggregate results"):
                replay(directory)
