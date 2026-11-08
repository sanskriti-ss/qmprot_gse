# ABOUTME: Checks that representative runs retain the documented budgets and geometry inputs.
# ABOUTME: Verifies the real configuration and published coordinates used by both execution environments.
import hashlib
import contextlib
import gzip
import io
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest

from active_space_truncation.geometry import load_geometry_from_h5
from smoke.representative import (
    AMINO_GRID, AMINO_SAMPLE, CAMPAIGN, FIXTURES, TINY_FIXTURES, WORKLOADS,
    fixture_root, geometry_hashes, specification,
)


class CampaignInputs(unittest.TestCase):
    def test_only_amino_acid_selection_varies(self):
        train = json.loads((CAMPAIGN / "01_trainability.json").read_text())
        noise = json.loads((CAMPAIGN / "02_noise_resilience.json").read_text())
        self.assertEqual(specification("hydrogen", "campaign"), train)
        self.assertEqual(specification("noise", "campaign"), noise)
        for profile, expected in (("campaign", AMINO_SAMPLE), ("full", AMINO_GRID)):
            spec = specification("amino", profile)
            self.assertEqual(spec["args"].pop("molecules"), list(expected))
            spec["args"]["molecule"] = "hydrogen"
            self.assertEqual(spec, train)
        self.assertEqual(AMINO_SAMPLE, ("gly", "arg", "gln"))
        self.assertEqual(len(set(AMINO_GRID)), 19)

    def test_exact_profile_limits_work_and_keeps_noise_channels(self):
        h = specification("hydrogen")["args"]
        amino = specification("amino")["args"]
        noise = specification("noise")["args"]
        self.assertEqual((h["param-targets"], h["eval-budget"]), ([2, 21], 44))
        self.assertEqual((amino["molecules"], amino["eval-budget"]), (list(AMINO_GRID), 8))
        self.assertEqual(noise["noise-models"], specification("noise", "campaign")["args"]["noise-models"])
        self.assertEqual(noise["noise-strengths"], [0, 0.01])

    def test_campaign_budget_with_tiny_inputs_only_changes_geometry_root(self):
        self.assertEqual(fixture_root("full"), FIXTURES)
        self.assertEqual(fixture_root("tiny-full"), TINY_FIXTURES)
        for name in WORKLOADS:
            self.assertEqual(specification(name, "tiny-full"), specification(name, "full"))
            self.assertEqual(specification(name, "tiny"), specification(name, "exact"))
        self.assertEqual(specification("amino", "tiny")["args"]["molecules"], ["gln", "glu"])
        tiny, published = geometry_hashes("tiny-full"), geometry_hashes("full")
        self.assertEqual(set(tiny), set(published))
        self.assertEqual({key for key in tiny if tiny[key] == published[key]}, {"gly"})

    def test_published_inputs_through_geometry_loader(self):
        records = {}
        for path in sorted(FIXTURES.glob("*/*.h5")):
            g = load_geometry_from_h5(str(path))
            coordinates = [[atom, *[float(x).hex() for x in xyz]] for atom, xyz in zip(g.atoms, g.coords)]
            records[path.parent.name] = [g.name, g.formula, g.charge, g.spin, g.n_atoms, coordinates]
        self.assertEqual(set(records), set(AMINO_GRID) | {"hydrogen"})
        data = json.dumps(records, sort_keys=True, separators=(",", ":")).encode()
        # Computed from the QMProt source fields at revision 71260c18, before HDF5 conversion.
        self.assertEqual(hashlib.sha256(data).hexdigest(),
                         "efe0f988629f5d7e6e9dbe8ae38e6ef203f8ae828273f9c84ac257fb24c6be60")

    def test_hamiltonian_identifiers_survive_chemistry_and_compression(self):
        from active_space_truncation.run_pipeline import run_pipeline
        from pyscf import lib

        with tempfile.TemporaryDirectory() as temp:
            for key in ("gln", "glu"):
                path = FIXTURES / key / f"{key}.h5"
                if key == "gln":
                    compressed = Path(temp) / "gln.h5.gz"
                    compressed.write_bytes(gzip.compress(path.read_bytes()))
                    path = compressed
                log = io.StringIO()
                with contextlib.redirect_stdout(log), lib.temporary_env(lib.StreamObject, stdout=log):
                    problem = run_pipeline(molecule=key, h5_path=str(path), basis="sto-3g", quiet=True)
                self.assertIn("CASCI E =", log.getvalue())
                ham = problem["hamiltonian"].qubit_hamiltonian
                self.assertEqual(ham.molecule.abbreviation, key)
                self.assertEqual(problem["geometry"].abbreviation, key)

    def test_event_names_distinguish_glutamine_and_glutamic_acid(self):
        framework = FIXTURES.parents[1]
        with tempfile.TemporaryDirectory() as temp:
            folder = Path(temp)
            events = folder / "events.jsonl"
            env = dict(os.environ, PYTHONPATH=str(framework), QMPROT_WORKLOAD="amino",
                       QMPROT_GEOMETRIES=str(FIXTURES), OMP_NUM_THREADS="1", OPENBLAS_NUM_THREADS="1",
                       VECLIB_MAXIMUM_THREADS="1", MPLBACKEND="Agg")
            command = [sys.executable, str(framework / "smoke/record.py"), "experiments.trainability",
                       str(events), "--molecules", "gln", "glu", "--algorithms", "hardware_efficient_vqe",
                       "--param-targets", "1", "--eval-budget", "8", "--no-plots", "--output-root", str(folder/"out")]
            run = subprocess.run(command, env=env, cwd=folder, capture_output=True, text=True)
            self.assertEqual(run.returncode, 0, run.stdout + run.stderr)
            self.assertEqual({json.loads(line)["molecule"] for line in events.read_text().splitlines()}, {"gln", "glu"})
