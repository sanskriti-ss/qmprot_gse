# ABOUTME: Checks the small H₂ fixtures and intact glycine against the published dataset naming conventions.
# ABOUTME: Verifies distinct references and identifiers through the real active-space pipeline.
import contextlib
import hashlib
import io
import json
import os
from pathlib import Path
import shutil
import subprocess
import tempfile
import unittest

import h5py
import numpy as np
from pyscf import lib

from active_space_truncation.geometry import load_geometry_from_h5
from active_space_truncation.run_pipeline import run_pipeline
from campaigns.run_campaign import argv_from_args
from smoke.representative import run_workload
from smoke.representative_checks import (
    validate_short_batch_objectives, validate_short_hydrogen, validate_tiny_reference, validate_trainability,
)


SMOKE = Path(__file__).resolve().parent
TINY = SMOKE / "tiny_geometries"
PUBLISHED = SMOKE / "geometries"


class TinyGeometries(unittest.TestCase):
    def test_names_coordinates_and_metadata(self):
        paths = sorted(p.relative_to(PUBLISHED) for p in PUBLISHED.glob("*/*.h5"))
        self.assertEqual(sorted(p.relative_to(TINY) for p in TINY.glob("*/*.h5")), paths)
        self.assertEqual(len(paths), 20)
        amino_keys = [path.stem for path in paths if path.stem != "hydrogen"]
        bond_pm = {key: 84 + 5 * i for i, key in enumerate(amino_keys)}
        bond_pm["hydrogen"] = 75
        for path in paths:
            with self.subTest(key=path.stem):
                original = load_geometry_from_h5(str(PUBLISHED / path))
                small = load_geometry_from_h5(str(TINY / path))
                self.assertEqual(path.parent.name, path.stem)
                self.assertEqual((small.abbreviation, small.name), (path.stem, original.name))
                if path.stem == "gly":
                    self.assertEqual((TINY / path).read_bytes(), (PUBLISHED / path).read_bytes())
                    continue
                self.assertEqual((small.formula, small.atoms), ("H2", ["H", "H"]))
                self.assertEqual((small.n_atoms, small.n_electrons, small.charge, small.spin), (2, 2, 0, 0))
                distance = np.linalg.norm(small.coords[1] - small.coords[0])
                self.assertAlmostEqual(distance * 100, bond_pm[path.stem], places=12)

    def test_active_spaces_match_reference_energies(self):
        with lib.with_omp_threads(1 if lib.num_threads() > 1 else None):
            for path in sorted(TINY.glob("*/*.h5")):
                key = path.stem
                with self.subTest(key=key):
                    log = io.StringIO()
                    with contextlib.redirect_stdout(log), lib.temporary_env(lib.StreamObject, stdout=log):
                        result = run_pipeline(molecule=key, h5_path=str(path), basis="sto-3g", quiet=True)
                    self.assertIn("CASCI E =", log.getvalue())
                    hamiltonian = result["hamiltonian"].qubit_hamiltonian
                    name = load_geometry_from_h5(str(PUBLISHED / key / path.name)).name
                    self.assertEqual((hamiltonian.molecule.abbreviation, hamiltonian.molecule.name), (key, name))
                    self.assertEqual(hamiltonian.n_qubits, 12 if key == "gly" else 4)
                    self.assertGreater(hamiltonian.n_terms, 1)
                    if key == "gly":
                        self.assertEqual(result["diagnostics"].n_core_orbitals, 5)
                        self.assertEqual(result["diagnostics"].proposed_active_indices, list(range(17, 23)))
                        self.assertEqual(hamiltonian.n_terms, 1819)
                    energy = result["active_space"].casci_energy
                    validate_tiny_reference(key, energy)
                    self.assertLess(energy, result["diagnostics"].hf_energy)

    def test_swapped_geometries_under_distinct_names_fail_batch(self):
        keys = ("gln", "glu")
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            inputs = root / "inputs"
            for key in keys:
                shutil.copytree(TINY / key, inputs / key)
            with h5py.File(inputs / "gln/gln.h5", "r+") as gln, h5py.File(inputs / "glu/glu.h5", "r+") as glu:
                for index in range(2):
                    path = f"coordinates/{index}/2"
                    gln[path][()], glu[path][()] = glu[path][()], gln[path][()]

            spec = json.loads((SMOKE.parent / "campaigns/20261105/01_trainability.json").read_text())
            args = spec["args"]
            del args["molecule"]
            args.update({"molecules": list(keys), "param-targets": [1], "eval-budget": 44, "max-iter": 12})
            env = dict(os.environ, QMPROT_GEOMETRIES=str(inputs), OMP_NUM_THREADS="1", OPENBLAS_NUM_THREADS="1",
                       VECLIB_MAXIMUM_THREADS="1", MPLBACKEND="Agg", MPLCONFIGDIR=str(root / "mpl"))
            output = root / "batch"
            result = subprocess.run([*argv_from_args(spec), "--output-root", str(output)], cwd=SMOKE.parent,
                                    env=env, capture_output=True, text=True)
            self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
            for key in keys:
                with self.subTest(key=key), self.assertRaisesRegex(ValueError, f"Unexpected tiny-fixture CASCI energy for {key}"):
                    validate_trainability(output / key, (key,), batch=False, targets=(1,), args=args, tiny_inputs=True)
            with self.assertRaisesRegex(ValueError, "Unexpected tiny-fixture CASCI energy for gln"):
                validate_trainability(output, keys, batch=True, targets=(1,), args=args, tiny_inputs=True)

    def test_swapped_zero_angle_objectives_fail_batch(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            with contextlib.redirect_stdout(io.StringIO()):
                run_workload("amino", root, profile="tiny")
            folder = root / "amino"
            keys = ("gln", "glu")
            validate_short_batch_objectives(folder, keys)
            path = folder / "events.jsonl"
            events = [json.loads(line) for line in path.read_text().splitlines()]
            zero = hashlib.sha256(np.array([0.0], dtype="<f8").tobytes()).hexdigest()
            zero_events = {e["molecule"]: e for e in events
                           if e["algorithm"] == "qubit_adapt_vqe" and e["parameters"] == zero}
            self.assertEqual(zero_events.keys(), set(keys))
            gln, glu = (zero_events[key] for key in keys)
            gln["energy"], glu["energy"] = glu["energy"], gln["energy"]
            path.write_text("".join(json.dumps(e) + "\n" for e in events))
            with self.assertRaisesRegex(ValueError, "Zero-angle ADAPT energy differs from HF for gln"):
                validate_short_batch_objectives(folder, keys)

    def test_short_hydrogen_detects_missing_solver_paths(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            with contextlib.redirect_stdout(io.StringIO()):
                run_workload("hydrogen", root, profile="tiny")
            folder = root / "hydrogen"
            validate_short_hydrogen(folder)
            path = folder / "events.jsonl"
            original = path.read_text().splitlines()
            for missing in ("second screen", "inner budget", "repeated HE callback"):
                with self.subTest(missing=missing):
                    events = [json.loads(line) for line in original]
                    for event in events:
                        if missing == "repeated HE callback" and event["callback"]:
                            event["parameters"] = "0" * 64
                        if event["algorithm"] == "qubit_adapt_vqe":
                            if missing == "second screen" and event["target"] == 2:
                                event["operators"] = event["operators"][:1]
                            if missing == "inner budget" and event["target"] == 21 and not event["operators"]:
                                event["operators"] = [0]
                    path.write_text("".join(json.dumps(e) + "\n" for e in events))
                    with self.assertRaisesRegex(ValueError, missing):
                        validate_short_hydrogen(folder)
