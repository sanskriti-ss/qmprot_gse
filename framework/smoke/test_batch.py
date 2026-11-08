# ABOUTME: Exercises a real two-molecule batch and verifies its saved outputs.
# ABOUTME: Checks that incomplete or damaged run artifacts fail validation.
import csv
from contextlib import redirect_stdout
from io import StringIO
import json
import shutil
import tempfile
import unittest
from pathlib import Path

from smoke.batch import read_rows, run, validate


class BatchArtifacts(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.directory = tempfile.TemporaryDirectory()
        cls.addClassCleanup(cls.directory.cleanup)
        cls.complete = Path(cls.directory.name) / "complete"
        with redirect_stdout(StringIO()) as progress:
            run(cls.complete)
        cls.progress = progress.getvalue()

    def setUp(self):
        root = self.enterContext(tempfile.TemporaryDirectory())
        self.output = Path(root) / "batch"
        shutil.copytree(self.complete, self.output)

    def test_complete_batch(self):
        self.assertEqual(self.progress, f"Running trainability batch. Log: {self.complete.resolve() / 'batch.log'}\n")
        validate(self.output)

    def test_skipped_molecule(self):
        path = self.output / "batch_manifest.json"
        manifest = json.loads(path.read_text())
        manifest["molecules_selected"].remove("arg")
        path.write_text(json.dumps(manifest))
        with self.assertRaisesRegex(ValueError, "molecules_selected"):
            validate(self.output)

    def test_duplicate_replaces_missing_algorithm(self):
        path = self.output / "all_molecules_results.csv"
        rows = path.read_text().splitlines()
        rows[-1] = rows[-2]
        path.write_text("\n".join(rows))
        with self.assertRaisesRegex(ValueError, "Unexpected result combinations"):
            validate(self.output)

    def test_nonfinite_energy(self):
        for name in ("all_molecules_results.csv", "gly/results.csv"):
            path = self.output / name
            rows = read_rows(path)
            rows[0]["final_energy"] = "nan"
            with path.open("w", newline="") as stream:
                writer = csv.DictWriter(stream, fieldnames=rows[0].keys())
                writer.writeheader()
                writer.writerows(rows)
        with self.assertRaisesRegex(ValueError, "inconsistent final energy"):
            validate(self.output)

    def test_truncated_trace_download(self):
        path = self.output / "all_molecules_convergence_traces.csv"
        path.write_text("\n".join(path.read_text().splitlines()[:-1]))
        with self.assertRaisesRegex(ValueError, "traces differ"):
            validate(self.output)

    def test_missing_plot(self):
        (self.output / "all_molecules_cost_evals_vs_params.png").unlink()
        with self.assertRaises(FileNotFoundError):
            validate(self.output)

    def test_backend_fallback(self):
        path = self.output / "batch.log"
        path.write_text(path.read_text().replace("Created device: lightning.qubit",
                                                 "Created device: default.qubit", 1))
        with self.assertRaisesRegex(ValueError, "Unexpected simulation backend"):
            validate(self.output)
