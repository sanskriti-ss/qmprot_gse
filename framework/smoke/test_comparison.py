# ABOUTME: Exercises scientific comparisons with actual trainability and noise runs.
# ABOUTME: Checks endpoint drift, parameter precision, trajectory changes and binary evidence.
import csv
import hashlib
import json
import math
import os
import shutil
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

from smoke.batch import FRAMEWORK
from smoke.comparison import compare


class ScientificComparison(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.directory = tempfile.TemporaryDirectory()
        cls.addClassCleanup(cls.directory.cleanup)
        cls.reference = Path(cls.directory.name) / "reference"
        cls.reference.mkdir()
        env = dict(os.environ, QMPROT_WORKLOAD="hydrogen", QMPROT_GEOMETRIES=str(FRAMEWORK / "smoke/geometries"),
                   OMP_NUM_THREADS="1", OPENBLAS_NUM_THREADS="1", VECLIB_MAXIMUM_THREADS="1",
                   MPLCONFIGDIR=cls.directory.name, MPLBACKEND="Agg")
        train = ["-m", "experiments.trainability", "--algorithms", "hardware_efficient_vqe",
                 "--param-targets", "1", "--eval-budget", "12", "--cs-target-qubits", "0",
                 "--no-plots"]
        runs = [
            ["-m", "smoke.record", "experiments.trainability", str(cls.reference / "amino-events.jsonl"),
             *train[2:], "--molecules", "gly", "--output-root", str(cls.reference / "amino")],
            ["-m", "smoke.record", "experiments.trainability", str(cls.reference / "hydrogen-events.jsonl"),
             *train[2:], "--molecule", "hydrogen", "--output-dir", str(cls.reference / "hydrogen")],
            ["-m", "experiments.noise_resilience", "--molecule", "hydrogen", "--n-layers", "1",
             "--max-iter", "12", "--noise-models", "phaseflip", "--noise-strengths", "0", "0.01",
             "--no-plots", "--output-dir", str(cls.reference / "noise")],
        ]
        for arguments in runs:
            env["QMPROT_WORKLOAD"] = "amino" if "--molecules" in arguments else "hydrogen"
            proc = subprocess.run([sys.executable, *arguments], cwd=FRAMEWORK, env=env, text=True,
                                  stdout=subprocess.PIPE, stderr=subprocess.STDOUT)
            if proc.returncode:
                raise RuntimeError(proc.stdout)

    def setUp(self):
        self.candidate = Path(self.enterContext(tempfile.TemporaryDirectory())) / "candidate"
        shutil.copytree(self.reference, self.candidate)

    def change_field(self, name, row, field, value):
        path = self.candidate / name
        with path.open(newline="") as stream:
            reader = csv.DictReader(stream)
            fields = reader.fieldnames
            data = list(reader)
        data[row][field] = value
        with path.open("w", newline="") as stream:
            writer = csv.DictWriter(stream, fieldnames=fields)
            writer.writeheader()
            writer.writerows(data)

    def test_matching_science_with_different_runtime(self):
        self.change_field("amino/all_molecules_results.csv", 0, "runtime_seconds", "135")
        summary = compare(self.reference, self.candidate)
        self.assertGreater(summary["csv_rows_compared"], 15)
        self.assertIsNone(summary["event_records_compared"])

    def test_reports_he_endpoint_and_trace(self):
        self.change_field("amino/all_molecules_results.csv", 0, "final_energy", "-260")
        self.change_field("amino/all_molecules_convergence_traces.csv", 0, "energy", "-261")
        with self.assertRaises(ValueError) as mismatch:
            compare(self.reference, self.candidate)
        report = str(mismatch.exception)
        self.assertIn("molecule=gly algorithm=hardware_efficient_vqe target_n_params=1", report)
        self.assertIn("Largest HE final energy (Ha) difference", report)
        self.assertIn("First differing trajectory record", report)
        self.assertIn("iteration=0", report)

    def test_counts_and_first_threshold_crossing(self):
        self.change_field("hydrogen/results.csv", 0, "evals_to_1.6mHa", "1001")
        with self.assertRaisesRegex(ValueError, "hydrogen/results.csv.*evals_to_1.6mHa"):
            compare(self.reference, self.candidate)

    def test_parameter_changes_at_binary_precision(self):
        path = self.candidate / "noise/optimum_parameters.json"
        data = json.loads(path.read_text())
        data["theta_clean_star"][0] = math.nextafter(data["theta_clean_star"][0], math.inf)
        path.write_text(json.dumps(data))
        with self.assertRaisesRegex(ValueError, r"theta_clean_star\[0\]"):
            compare(self.reference, self.candidate)

    def test_missing_file_in_both_runs(self):
        empty = self.candidate / "empty"
        empty.mkdir()
        with self.assertRaisesRegex(ValueError, "batch_manifest.json"):
            compare(empty, empty)

    def test_suite_scopes_are_checked_before_results(self):
        baseline = self.candidate.with_name("saved-suite")
        shutil.copytree(self.reference, baseline)
        (baseline / "suite.json").write_text(json.dumps({"tiny": True, "full": True}))
        (self.candidate / "amino/all_molecules_results.csv").unlink()
        with self.assertRaisesRegex(ValueError, "Both runs must provide suite.json"):
            compare(baseline, self.candidate)
        profiles = [
            ({"full": True}, "full"),
            ({"tiny": True}, "tiny"),
            ({"tiny": True, "campaign_budget": True}, "tiny-full"),
        ]
        for scope, profile in profiles:
            with self.subTest(profile=profile):
                (self.candidate / "suite.json").write_text(json.dumps(scope))
                with self.assertRaisesRegex(ValueError, f"scope.*tiny-grid.*{profile}$"):
                    compare(baseline, self.candidate)

    def test_suite_geometry_provenance(self):
        baseline = self.candidate.with_name("saved-suite")
        shutil.copytree(self.reference, baseline)
        for directory in (baseline, self.candidate):
            (directory / "suite.json").write_text(json.dumps({"exact": True}))
            for workload, molecule in (("hydrogen", "hydrogen"), ("amino", "gly"), ("noise", "hydrogen")):
                geometry = FRAMEWORK / "smoke/geometries" / molecule / f"{molecule}.h5"
                hashes = {molecule: hashlib.sha256(geometry.read_bytes()).hexdigest()}
                (directory / workload / "measurement.json").write_text(json.dumps({"geometries": hashes}))
        self.assertGreater(compare(baseline, self.candidate)["csv_rows_compared"], 0)
        path = self.candidate / "noise/measurement.json"
        gly = json.loads((self.candidate / "amino/measurement.json").read_text())["geometries"]["gly"]
        path.write_text(json.dumps({"geometries": {"hydrogen": gly}}))
        with self.assertRaisesRegex(ValueError, "noise/measurement.json.*hydrogen"):
            compare(baseline, self.candidate)
        path.write_text(json.dumps({"geometries": {}}))
        with self.assertRaisesRegex(ValueError, "noise/measurement.json.*geometr"):
            compare(baseline, self.candidate)
        path.unlink()
        with self.assertRaisesRegex(ValueError, "noise/measurement.json"):
            compare(baseline, self.candidate)

    def test_binary_event_data(self):
        records = [json.loads(line) for line in
                   (self.reference / "hydrogen-events.jsonl").read_text().splitlines()]
        path = self.candidate / "events.jsonl"
        path.write_text("\n".join(map(json.dumps, records)) + "\n")
        with self.assertRaisesRegex(ValueError, "Both runs must provide"):
            compare(self.reference, self.candidate)
        baseline = self.candidate.with_name("with-events")
        shutil.copytree(self.candidate, baseline)
        self.assertEqual(compare(baseline, self.candidate)["event_records_compared"], len(records))
        energy = float.fromhex(records[0]["energy"])
        records[0]["energy"] = math.nextafter(energy, math.inf).hex()
        records[1]["parameters"] = records[-1]["parameters"]
        path.write_text("\n".join(map(json.dumps, records)) + "\n")
        with self.assertRaises(ValueError) as mismatch:
            compare(baseline, self.candidate)
        report = str(mismatch.exception)
        self.assertIn(f"event 1 workload=hydrogen molecule={records[0]['molecule']} algorithm=hardware_efficient_vqe target=1 energy", report)
        self.assertIn("Largest fixed-input event energy (Ha) difference", report)
        self.assertIn("First differing optimizer input: events.jsonl event 2", report)

    def test_missing_callback_preserves_fixed_input_comparisons(self):
        path = self.candidate / "events.jsonl"
        gly = (self.reference / "amino-events.jsonl").read_text().splitlines()
        hydrogen = (self.reference / "hydrogen-events.jsonl").read_text().splitlines()
        path.write_text("\n".join(gly + hydrogen) + "\n")
        baseline = self.candidate.with_name("reference-events")
        shutil.copytree(self.candidate, baseline)
        removed = next(i for i, line in enumerate(gly) if json.loads(line)["callback"])
        for position, case in [(len(gly) - 1, "gly"), (len(gly) + 2, "hydrogen")]:
            with self.subTest(case=case):
                damaged = gly + hydrogen
                item = json.loads(damaged[position])
                item["energy"] = (float.fromhex(item["energy"]) + 0.25).hex()
                damaged[position] = json.dumps(item)
                del damaged[removed]
                path.write_text("\n".join(damaged) + "\n")
                with self.assertRaises(ValueError) as mismatch:
                    compare(baseline, self.candidate)
                report = str(mismatch.exception)
                self.assertIn(f"First differing optimizer input: events.jsonl event {removed + 1} workload=amino molecule=gly", report)
                self.assertRegex(report, rf"Largest fixed-input event energy \(Ha\) difference 0.25[0]* at .*molecule={case} ")

    def test_missing_reordered_and_split_cases(self):
        gly = (self.reference / "amino-events.jsonl").read_text().splitlines()
        hydrogen = (self.reference / "hydrogen-events.jsonl").read_text().splitlines()
        path = self.candidate / "events.jsonl"
        path.write_text("\n".join(gly + hydrogen) + "\n")
        baseline = self.candidate.with_name("reference-events")
        shutil.copytree(self.candidate, baseline)
        for records, problem in [(hydrogen, "case order"), (hydrogen + gly, "case order"),
                                 (gly[:-1] + hydrogen + gly[-1:], "noncontiguous case")]:
            with self.subTest(problem=problem):
                path.write_text("\n".join(records) + "\n")
                with self.assertRaisesRegex(ValueError, problem):
                    compare(baseline, self.candidate)
        first = json.loads(gly[0])
        del first["target"]
        path.write_text(json.dumps(first) + "\n")
        with self.assertRaisesRegex(ValueError, "invalid case.*target"):
            compare(baseline, self.candidate)
        invalid = json.loads(gly[0])
        del invalid["parameters"]
        path.write_text(json.dumps(invalid) + "\n")
        with self.assertRaisesRegex(ValueError, "malformed event.*parameters"):
            compare(baseline, self.candidate)
