# ABOUTME: Runs both campaign experiments through the production shell launcher.
# ABOUTME: Checks the dispatch manifest, batch and single-molecule outputs using tiny geometries.
import json
import os
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

from smoke.batch import FRAMEWORK
from smoke.representative import fixture_root, specification
from smoke.representative_checks import validate_noise, validate_trainability


class CampaignLauncher(unittest.TestCase):
    def test_dispatches_trainability_batch_and_noise(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary).resolve()
            campaign = root / "campaigns" / "smoke"
            campaign.mkdir(parents=True)
            train = specification("amino", "tiny")
            noise = specification("noise", "tiny")
            for filename, spec in (("01_trainability.json", train), ("02_noise_resilience.json", noise)):
                (campaign / filename).write_text(json.dumps(spec))

            env = dict(os.environ, QMPROT_GEOMETRIES=str(fixture_root("tiny")),
                       MPLBACKEND="Agg", MPLCONFIGDIR=os.environ.get("MPLCONFIGDIR", str(root / "matplotlib")),
                       OMP_NUM_THREADS="1", OPENBLAS_NUM_THREADS="1", MKL_NUM_THREADS="1",
                       VECLIB_MAXIMUM_THREADS="1")
            env["PATH"] = os.pathsep.join([str(Path(sys.executable).parent), env["PATH"]])
            env.pop("PYTHONPATH", None)
            launch = subprocess.run([str(FRAMEWORK / "campaigns/run.sh"), str(campaign)],
                                    cwd=root, env=env, capture_output=True, text=True)
            diagnostics = launch.stdout + launch.stderr
            self.assertEqual(launch.returncode, 0, diagnostics)
            for marker in ("[ERROR]", "[WARNING]", "UserWarning:", "Traceback (most recent call last)"):
                self.assertNotIn(marker, diagnostics)

            manifest = json.loads((campaign / "manifest.json").read_text())
            self.assertEqual(manifest["campaign"], campaign.name)
            self.assertEqual(len(manifest["runs"]), 1)
            commands = manifest["runs"][0]["experiments"]
            self.assertEqual([entry["file"] for entry in commands],
                             ["01_trainability.json", "02_noise_resilience.json"])
            results = campaign / "results"
            expected = [("experiments.trainability", "--output-root", results / "01_trainability"),
                        ("experiments.noise_resilience", "--output-dir", results / "02_noise_resilience")]
            for entry, (module, flag, path) in zip(commands, expected):
                self.assertEqual(entry["argv"][1:3], ["-m", module])
                self.assertEqual(entry["argv"][-2:], [flag, str(path)])
                self.assertEqual(entry["results"], str(path))

            args = train["args"]
            validate_trainability(results / "01_trainability", tuple(args["molecules"]), batch=True,
                                  targets=args["param-targets"], args=args, tiny_inputs=True)
            validate_noise(results / "02_noise_resilience", args=noise["args"], tiny_inputs=True)

            failed_spec = {"module": "experiments.missing_for_dispatch_check", "args": {}}
            (campaign / "00_missing.json").write_text(json.dumps(failed_spec))
            failed = subprocess.run([str(FRAMEWORK / "campaigns/run.sh"), str(campaign)],
                                    cwd=root, env=env, capture_output=True, text=True)
            self.assertNotEqual(failed.returncode, 0)
            self.assertIn("No module named experiments.missing_for_dispatch_check", failed.stderr)
            first, retry = json.loads((campaign / "manifest.json").read_text())["runs"]
            self.assertEqual(first["status"], "completed")
            self.assertEqual(retry["status"], "failed")
            self.assertEqual(len(retry["experiments"]), 1)
            self.assertEqual(retry["experiments"][0]["returncode"], failed.returncode)
