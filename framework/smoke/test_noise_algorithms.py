# ABOUTME: Checks fixed-ansatz noise experiments and rejects solvers with a different objective.
# ABOUTME: Exercises the public dispatch and saved results on the published hydrogen geometry.
import csv
import json
import os
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

from algorithms import ALGORITHMS
from experiments.noise_resilience import NOISE_ALGORITHMS, run_noise_resilience, run_noise_resilience_batch
from smoke.batch import FRAMEWORK


class NoiseAlgorithmsTest(unittest.TestCase):
    def test_frozen_input_matches_requested_configuration(self):
        frozen = FRAMEWORK / "smoke/reproductions/layered_vqe"
        for molecule, basis, target in (("hydrogen", "sto-3g", 8), ("gly", "6-31g", 8), ("gly", "sto-3g", None)):
            with self.subTest(molecule=molecule, basis=basis, target=target), self.assertRaisesRegex(ValueError, "Prepared model"):
                run_noise_resilience(molecule=molecule, basis=basis, cs_target_qubits=target,
                                     input_snapshot=frozen / "contextual_gly.npz")
        snapshot = frozen / "hydrogen.npz"
        base = [sys.executable, "-m", "experiments.noise_resilience", "--molecule", "hydrogen",
                "--cs-target-qubits", "0", "--input-snapshot", str(snapshot), "--noise-models", "depolarizing",
                "--noise-strengths", "0", "--max-iter", "16", "--no-plots"]
        with tempfile.TemporaryDirectory() as tmp:
            env = dict(os.environ, QMPROT_GEOMETRIES=tmp)
            out = Path(tmp) / "fit"
            completed = subprocess.run([*base, "--output-dir", str(out)], cwd=FRAMEWORK, env=env,
                                       capture_output=True, text=True)
            self.assertEqual(completed.returncode, 0, completed.stderr)
            self.assertNotRegex(completed.stderr, r"\[WARNING\]|\[ERROR\]|Warning:")
            self.assertEqual((out / "inputs/hydrogen.npz").read_bytes(), snapshot.read_bytes())
            invalid = subprocess.run([*base, "--batch-molecules", "hydrogen"], cwd=FRAMEWORK, env=env,
                                     capture_output=True, text=True)
            self.assertEqual(invalid.returncode, 2)
            self.assertIn("--input-snapshot requires a single-molecule fit", invalid.stderr)

    def test_unsupported_objectives_rejected_before_loading(self):
        excluded = {"adapt_vqe", "qubit_adapt_vqe", "iqcc_inspired_vqe", "iqcc_true_vqe", "cb_vqe"}
        self.assertEqual(set(ALGORITHMS) - set(NOISE_ALGORITHMS), excluded)
        with tempfile.TemporaryDirectory() as tmp:
            destination = Path(tmp) / "experiment"
            for name in excluded:
                with self.subTest(algorithm=name), self.assertRaisesRegex(ValueError, "fixed.*ansatz"):
                    run_noise_resilience(molecule="missing", algorithm=name, output_dir=destination)
                with self.subTest(batch=name), self.assertRaisesRegex(ValueError, "fixed.*ansatz"):
                    run_noise_resilience_batch(["missing"], algorithm=name, output_dir=destination)
            with self.assertRaisesRegex(ValueError, "gradient descent"):
                run_noise_resilience(molecule="missing", optimizer="GD", output_dir=destination)
            self.assertFalse(destination.exists())

    def test_supported_circuits_train_with_noise(self):
        env = dict(os.environ, QMPROT_GEOMETRIES=str(FRAMEWORK / "smoke/geometries"),
                   OMP_NUM_THREADS="1", OPENBLAS_NUM_THREADS="1", VECLIB_MAXIMUM_THREADS="1",
                   MPLBACKEND="Agg")
        with tempfile.TemporaryDirectory() as tmp:
            for name in NOISE_ALGORITHMS:
                with self.subTest(algorithm=name):
                    out = Path(tmp) / name
                    command = [sys.executable, "-m", "experiments.noise_resilience", "--algorithm", name,
                               "--molecule", "hydrogen", "--cs-target-qubits", "0", "--n-layers", "1",
                               "--max-iter", "32", "--noise-models", "depolarizing", "--noise-strengths",
                               "0", "0.05", "--no-plots", "--output-dir", str(out)]
                    result = subprocess.run(command, cwd=FRAMEWORK, env=env, capture_output=True, text=True)
                    log = result.stdout + result.stderr
                    self.assertEqual(result.returncode, 0, log)
                    for marker in ("[ERROR]", "[WARNING]", "Warning:"):
                        self.assertNotIn(marker, log)
                    config = json.loads((out / "run_config.json").read_text())
                    self.assertEqual(config["algorithm"], name)
                    self.assertEqual(config["prepare_hf"], name != "qaoa_inspired_vqe")
                    self.assertEqual(config["n_layers"], None if name in ("ucc_vqe", "nn_ae_vqe") else 1)
                    self.assertGreater(config["n_parameters"], 0)
                    endpoints = json.loads((out / "state_endpoints.json").read_text())
                    self.assertEqual(endpoints["none@0"], endpoints["depolarizing@0"])
                    mixed = endpoints["depolarizing@0.05"]["noisy"]
                    pure = endpoints["depolarizing@0.05"]["clean_at_noisy_params"]
                    self.assertEqual(mixed["state_kind"], "density_matrix")
                    self.assertEqual(pure["state_kind"], "statevector")
                    with (out / "results.csv").open() as stream:
                        clean, zero, noisy = csv.DictReader(stream)
                    self.assertEqual(clean["energy_noisy"], zero["energy_noisy"])
                    self.assertAlmostEqual(float(noisy["wrong_electron_probability_noisy"]), mixed["wrong_electron_probability"])
                    self.assertAlmostEqual(float(noisy["wrong_electron_probability_clean_at_noisy_params"]), pure["wrong_electron_probability"])
                    self.assertEqual(noisy["param_drift_l2_wrapped"] == "", name == "hva_vqe")
                    self.assertNotEqual(float(clean["energy_noisy"]), float(noisy["energy_noisy"]))
