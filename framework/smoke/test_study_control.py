# ABOUTME: Checks acceptance of the native two-molecule fermionic ADAPT comparison and replay inputs.
# ABOUTME: Ensures missing sector controls and mismatched molecular inputs fail after retrieval.
import json
import os
from pathlib import Path
import tempfile
import unittest

from smoke.iqcc_control import run as run_iqcc, validate as validate_iqcc
from smoke.study_campaign import run_fermionic_control
from smoke.study_checks import _check_fermionic_control


class FermionicStudyControl(unittest.TestCase):
    def test_fixed_glycine_history(self):
        with tempfile.TemporaryDirectory() as tmp:
            directory = Path(tmp)
            run_iqcc(directory)
            report = validate_iqcc(directory)
            self.assertGreaterEqual(report["rotations"], 2)
            self.assertLess(report["largest_rescore_difference_ha"], 1e-9)
            path = directory / "iqcc-control.json"
            case = json.loads(path.read_text())
            case["angles"][0] += 0.5
            path.write_text(json.dumps(case))
            with self.assertRaises(AssertionError):
                validate_iqcc(directory)

    def test_two_molecule_result_and_damaged_artifacts(self):
        env = dict(os.environ, OMP_NUM_THREADS="1", OPENBLAS_NUM_THREADS="1", VECLIB_MAXIMUM_THREADS="1")
        with tempfile.TemporaryDirectory() as tmp:
            directory = Path(tmp)
            run_fermionic_control(directory, env)
            self.assertEqual((directory / "campaign.log").read_text(), "")
            self.assertEqual(set(_check_fermionic_control(directory)), {"hydrogen", "gly"})
            path = directory / "fermionic-control.json"
            original = path.read_text()
            control = json.loads(original)
            control["molecules"]["gly"]["matched_rotations"].pop()
            path.write_text(json.dumps(control))
            with self.assertRaisesRegex(ValueError, "Missing gly gate comparisons"):
                _check_fermionic_control(directory)
            control = json.loads(original)
            control["molecules"]["gly"]["geometry_sha256"] = control["molecules"]["hydrogen"]["geometry_sha256"]
            path.write_text(json.dumps(control))
            with self.assertRaisesRegex(ValueError, "Incorrect gly fermionic geometry"):
                _check_fermionic_control(directory)
