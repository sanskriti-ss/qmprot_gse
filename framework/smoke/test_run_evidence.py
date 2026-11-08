# ABOUTME: Replays saved prefix inputs and verifies the recorded case outcomes and source archive.
# ABOUTME: Checks reference energies across supported Pauli encodings and projected states.
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest
from zipfile import ZipFile

import numpy as np

from core.hamiltonian_loader import HamiltonianLoader, Molecule, QubitHamiltonian
from core.hf_verification import compute_hf_energy, compute_hf_energy_pennylane
from experiments.provenance import load_problem
from experiments._common import LoadedHamiltonian
from experiments.trainability import _run_adapt_at_target

FRAMEWORK = Path(__file__).resolve().parents[1]


class RunEvidence(unittest.TestCase):
    def test_interrupted_screen_scores_the_encoded_initial_state(self):
        mol = Molecule("pair", "pair", 2, 2, -2., "", n_electrons=1)
        ham = QubitHamiltonian.from_dict({"ZI": 2., "IX": 1.}, mol)
        ham.cs_initial_state = np.array([1., 0., 0., 0.], dtype=complex)
        loaded = LoadedHamiltonian(ham, "pair", "basis", 4, 2, True, casci_energy=-3., hf_energy=-2.)
        row = _run_adapt_at_target(loaded, algorithm="qubit_adapt_vqe", target_n_params=2,
                                   gradient_threshold=1e-8, qubit_adapt_max_gradient_candidates=None,
                                   optimizer="COBYLA", max_iterations=16, convergence_threshold=1e-9,
                                   random_seed=42, eval_budget=1)
        self.assertEqual(row.actual_n_params, 0)
        self.assertEqual(row.n_cost_evals, 1)
        self.assertAlmostEqual(row.final_energy, 2.)

    def test_wire_products_and_projected_reference(self):
        encoded = "1\tIdentity(3) @ PauliZ(0) @ Z(1)\n2\tIdentity(1) @ Z(0)"
        loader = HamiltonianLoader(FRAMEWORK / "data/hamiltonians")
        _, words = loader._parse_hamiltonian_string(encoded, n_qubits=4)
        self.assertEqual(words, ["ZZII", "ZIII"])
        for word in (words[0], "Z(0) @ Z(1)", "Identity(3) @ PauliZ(0) @ PauliZ(1)"):
            mol = Molecule("h", "h", 4, 1, -1., "", n_electrons=2)
            ham = QubitHamiltonian(mol, np.array([1.]), [word], 4, 1)
            self.assertAlmostEqual(compute_hf_energy(ham), 1.)
            self.assertAlmostEqual(compute_hf_energy_pennylane(ham), 1.)
            ket = np.zeros(16, dtype=complex)
            ket[[0, 4]] = [0.8, 0.6j]
            ham.cs_initial_state = ket
            self.assertAlmostEqual(compute_hf_energy(ham), 0.28)
            self.assertAlmostEqual(compute_hf_energy_pennylane(ham), 0.28)

    def test_trainability_failure_and_resume_require_the_requested_model(self):
        base = [sys.executable, "-m", "experiments.trainability", "--molecules", "hydrogen", "--algorithms",
                "hardware_efficient_vqe", "--eval-budget", "16", "--max-iter", "16", "--no-plots"]
        env = dict(os.environ, QMPROT_GEOMETRIES=str(FRAMEWORK / "smoke/geometries"))
        with tempfile.TemporaryDirectory() as tmp:
            def run(target, *extra):
                return subprocess.run([*base, "--output-root", tmp, "--param-targets", str(target), *extra],
                                      cwd=FRAMEWORK, env=env, capture_output=True, text=True)
            first = run(2)
            self.assertEqual(first.returncode, 0, first.stdout + first.stderr)
            same = run(2, "--resume")
            self.assertEqual(same.returncode, 0, same.stdout + same.stderr)
            manifest = Path(tmp) / "batch_manifest.json"
            self.assertEqual(json.loads(manifest.read_text())["molecules_resumed"], ["hydrogen"])
            snapshot = Path(tmp) / "hydrogen/inputs/hydrogen.npz"
            snapshot.write_bytes(b"incomplete write")
            retry = run(2, "--resume")
            self.assertEqual(retry.returncode, 0, retry.stdout + retry.stderr)
            self.assertEqual(json.loads(manifest.read_text())["molecules_resumed"], [])
            self.assertEqual(load_problem(snapshot).molecule, "hydrogen")
            changed = run(6, "--resume")
            self.assertNotEqual(changed.returncode, 0)
            self.assertIn("Resume settings differ", changed.stderr)
            failed = run(2, "--optimizer", "invalid-readiness-optimizer")
            self.assertNotEqual(failed.returncode, 0)
            self.assertIn("Incomplete trainability results", failed.stderr)

    def test_saved_sweep_replays_without_geometry_and_accounts_for_cases(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            base = [sys.executable, "-m", "experiments.experiment5_term_count_sweep", "--molecules", "hydrogen",
                    "--algorithms", "hardware_efficient_vqe", "--term-counts", "2", "15", "999",
                    "--max-iter", "24", "--cs-target-qubits", "0", "--no-plots"]
            env = dict(os.environ, QMPROT_GEOMETRIES=str(FRAMEWORK / "smoke/geometries"))
            def run(args, destination):
                return subprocess.run([*base, *args, "--output-dir", str(destination)], cwd=FRAMEWORK,
                                      env=env, capture_output=True, text=True)
            original, replay = root / "original", root / "replay"
            result = run([], original)
            self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
            source = original / "framework-source.zip"
            runtime = json.loads((original / "runtime.json").read_text())
            self.assertEqual(hashlib.sha256(source.read_bytes()).hexdigest(), runtime["source_sha256"])
            with ZipFile(source) as zip_file:
                self.assertEqual(zip_file.read("experiments/experiment5_term_count_sweep.py"),
                                 (FRAMEWORK / "experiments/experiment5_term_count_sweep.py").read_bytes())
            env["QMPROT_GEOMETRIES"] = str(root / "unavailable")
            result = run(["--prepared-inputs", str(original / "inputs")], replay)
            self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
            def cases(folder):
                return [json.loads(line) for line in (folder / "cases.jsonl").read_text().splitlines()]
            self.assertEqual(cases(original), cases(replay))
            snapshot = original / "inputs/hydrogen.npz"
            self.assertEqual(cases(original)[0]["input_sha256"], hashlib.sha256(snapshot.read_bytes()).hexdigest())
            self.assertEqual(load_problem(snapshot).preparation["atoms"], ["H", "H"])
            self.assertEqual(len(json.loads((replay / "skipped_cases.json").read_text())), 1)
            self.assertNotEqual(run([], original).returncode, 0)
            self.assertEqual(len(cases(original)), 2)
            bad = run([], root / "failed")
            self.assertNotEqual(bad.returncode, 0)
            self.assertEqual(json.loads((root / "failed/progress.json").read_text())["status"], "failed")
