# ABOUTME: Checks standalone Git restoration and verifies all retained evidence in a research handoff.
# ABOUTME: Exercises a fresh repository, cross-directory native outputs, external links and corrupted archives.
import json
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest
from zipfile import ZipFile


SCRIPT = Path(__file__).resolve().parents[2] / "utils/research_handoff.py"


class ResearchHandoff(unittest.TestCase):
    def test_offline_restore_and_changed_evidence(self):
        with tempfile.TemporaryDirectory() as tmp:
            directory = Path(tmp)
            root = directory / "study"
            subprocess.run(["git", "init", "-q", "-b", "study", str(root)], check=True)
            (root / "README.md").write_text("Trial record\n")
            (root / ".gitignore").write_text("framework/results/\nframework/experiments/results/\nframework/logs/\n")
            subprocess.run(["git", "add", "--", "README.md", ".gitignore"], cwd=root, check=True)
            subprocess.run(["git", "-c", "user.name=Test", "-c", "user.email=test@example.org",
                            "commit", "-qm", "Store study"], cwd=root, check=True)
            members = ["framework/results/cloud/events.jsonl", "framework/experiments/results/matrix/results.csv",
                       "framework/logs/solver.log"]
            for name in members:
                path = root / name
                path.parent.mkdir(parents=True)
                path.write_text(name + "\n")
            (root / "framework/results/uninstalled-package").symlink_to(directory / "external")
            releases = root / "framework/results/handoffs"
            releases.mkdir()
            (releases / "prior.zip").write_bytes(b"published archive")
            (releases / "prior.json").write_text("{}\n")
            archive = directory / "run.zip"
            def run(*args):
                return subprocess.run([sys.executable, str(SCRIPT), *args], cwd=root, capture_output=True, text=True)
            built = run("build", str(archive))
            self.assertEqual((built.returncode, built.stderr), (0, ""))
            verified = run("verify", str(archive))
            self.assertEqual((verified.returncode, verified.stderr), (0, ""))
            self.assertEqual(json.loads(verified.stdout)["tracked_files"], 2)
            with ZipFile(archive) as source:
                self.assertFalse(any(name.startswith("framework/results/handoffs/") for name in source.namelist()))
                manifest = json.loads(source.read("EVIDENCE_MANIFEST.json"))
                self.assertEqual(set(manifest["omitted_links"]), {"framework/results/uninstalled-package"})
                for name in members:
                    self.assertEqual(source.read(name), (name + "\n").encode())
                changed = directory / "changed.zip"
                with ZipFile(changed, "w") as target:
                    for info in source.infolist():
                        content = b"changed" if info.filename == members[0] else source.read(info)
                        target.writestr(info, content)
            damaged = run("verify", str(changed))
            self.assertNotEqual(damaged.returncode, 0)
            self.assertIn("Content differs from manifest for " + members[0], damaged.stderr)
