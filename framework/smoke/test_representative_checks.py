# ABOUTME: Exercises representative artifact checks using hydrogen experiments on the published geometry.
# ABOUTME: Damages individual saved files to verify missing cases and inconsistent scientific values are detected.
import csv
import json
import os
from pathlib import Path
import shutil
import subprocess
import tempfile
import unittest

from campaigns.run_campaign import argv_from_args
from smoke.batch import FRAMEWORK, read_rows
from smoke.representative import specification
from smoke.representative_checks import validate_noise, validate_trainability


def change_row(path, transform):
    rows = read_rows(path)
    header = list(rows[0])
    transform(rows)
    with path.open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=header)
        writer.writeheader()
        writer.writerows(rows)


class RepresentativeArtifacts(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.workspace = tempfile.TemporaryDirectory()
        cls.addClassCleanup(cls.workspace.cleanup)
        root = Path(cls.workspace.name)
        cls.train = root / "train"
        cls.noise = root / "noise"
        env = dict(os.environ, QMPROT_GEOMETRIES=str(FRAMEWORK / "smoke/geometries"),
                   OMP_NUM_THREADS="1", OPENBLAS_NUM_THREADS="1", VECLIB_MAXIMUM_THREADS="1",
                   MPLBACKEND="Agg", MPLCONFIGDIR=str(root / "mpl"))
        for name, dest in (("hydrogen", cls.train), ("noise", cls.noise)):
            spec = specification(name)
            if dest == cls.train:
                spec["args"].pop("molecule")
                spec["args"].update({"molecules": ["hydrogen"], "param-targets": [2]})
            flag = "--output-root" if dest == cls.train else "--output-dir"
            result = subprocess.run([*argv_from_args(spec), flag, str(dest)], cwd=FRAMEWORK,
                                    env=env, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)
            if result.returncode:
                raise RuntimeError(result.stdout)

    def copy(self, source):
        dest = Path(self.enterContext(tempfile.TemporaryDirectory())) / "outputs"
        shutil.copytree(source, dest)
        return dest

    def train_check(self, path):
        validate_trainability(path, ("hydrogen",), batch=True, targets=(2,), args=specification("hydrogen")["args"])

    def test_real_batch_single_and_noise(self):
        self.train_check(self.train)
        validate_trainability(self.train / "hydrogen", ("hydrogen",), batch=False, targets=(2,), args=specification("hydrogen")["args"])
        validate_noise(self.noise, args=specification("noise")["args"])

    def test_duplicate_noise_row(self):
        out = self.copy(self.noise)
        change_row(out / "results.csv", lambda r: r.__setitem__(-1, r[-2]))
        with self.assertRaisesRegex(ValueError, "noise grid"):
            validate_noise(out, args=specification("noise")["args"])

    def test_damaged_optimum(self):
        out = self.copy(self.noise)
        path = out / "optimum_parameters.json"
        params = json.loads(path.read_text())
        params["theta_noisy_star"]["phaseflip@0.01"][0] += 0.5
        path.write_text(json.dumps(params))
        with self.assertRaisesRegex(ValueError, "param_drift"):
            validate_noise(out, args=specification("noise")["args"])

    def test_noise_channel_matches_clean_evaluations(self):
        out = self.copy(self.noise)
        def erase_channel_effect(rows):
            for row in rows:
                if row["noise_model"] == "phaseflip" and float(row["noise_strength"]):
                    row["energy_clean_at_noisy_params"] = row["energy_noisy"]
        change_row(out / "results.csv", erase_channel_effect)
        with self.assertRaisesRegex(ValueError, "No measured noise effect"):
            validate_noise(out, args=specification("noise")["args"])

    def test_missing_noise_trace(self):
        out = self.copy(self.noise)
        change_row(out / "convergence_traces.csv", lambda r: r.pop())
        with self.assertRaisesRegex(ValueError, "iterations"):
            validate_noise(out, args=specification("noise")["args"])

    def test_wrong_campaign_configuration(self):
        out = self.copy(self.train)
        p = out / "hydrogen/run_config.json"
        cfg = json.loads(p.read_text())
        cfg["random_seed"] += 1
        p.write_text(json.dumps(cfg))
        with self.assertRaisesRegex(ValueError, "random_seed"):
            self.train_check(out)

    def test_budget_and_nonfinite_energy(self):
        for field, value in (("n_cost_evals", "2001"), ("final_energy", "nan")):
            with self.subTest(field=field):
                out = self.copy(self.train)
                for p in ("hydrogen/results.csv", "all_molecules_results.csv"):
                    change_row(out / p, lambda rows: rows[0].update({field: value}))
                with self.assertRaises(ValueError):
                    self.train_check(out)

    def test_aggregate_duplicate(self):
        out = self.copy(self.train)
        change_row(out / "all_molecules_results.csv", lambda r: r.__setitem__(-1, r[0]))
        with self.assertRaisesRegex(ValueError, "aggregate"):
            self.train_check(out)

    def test_inconsistent_accuracy_crossing(self):
        out = self.copy(self.train)
        for p in ("hydrogen/results.csv", "all_molecules_results.csv"):
            change_row(out / p, lambda rows: rows[0].update({"evals_to_100mHa": "2001", "reached_100mHa": "1"}))
        with self.assertRaisesRegex(ValueError, "threshold crossing"):
            self.train_check(out)

    def test_missing_final_evaluation_trace(self):
        out = self.copy(self.train)
        change_row(out / "hydrogen/eval_traces.csv", lambda r: r.pop())
        with self.assertRaisesRegex(ValueError, "evaluation trace"):
            self.train_check(out)

    def test_damaged_plot(self):
        out = self.copy(self.noise)
        (out / "param_drift_vs_noise.png").write_bytes(b"")
        with self.assertRaises(OSError):
            validate_noise(out, args=specification("noise")["args"])
