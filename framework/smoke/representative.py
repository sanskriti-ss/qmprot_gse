# ABOUTME: Runs the production trainability and noise campaigns on the pinned smoke geometries.
# ABOUTME: Verifies the on-disk results and records full-precision optimizer decisions for comparison.
import argparse
from collections import Counter
import hashlib
import importlib.metadata
import json
import math
import os
from pathlib import Path
import platform
import shutil
import subprocess
import sys
import time
from uuid import uuid4

from campaigns.run_campaign import argv_from_args, output_flag
from smoke.batch import FRAMEWORK, read_rows, require

WORKLOADS = ("hydrogen", "amino", "noise")
AMINO_SAMPLE = ("gly", "arg", "gln")
AMINO_GRID = tuple("ala arg asp cys gln glu gly his ile leu lys met phe pro ser thr trp tyr val".split())
SCOPES = {"tiny-grid": {"tiny": True, "full": True}, "tiny": {"tiny": True},
          "tiny-full": {"tiny": True, "campaign_budget": True}, "exact": {"exact": True},
          "campaign": {"full": False}, "full": {"full": True}}
CAMPAIGN = FRAMEWORK / "campaigns/20261105"
FIXTURES = FRAMEWORK / "smoke/geometries"
TINY_FIXTURES = FRAMEWORK / "smoke/tiny_geometries"
MAXFUN_WARNING = ("UserWarning: COBYLA: Invalid MAXFUN; "
                  "it should be at least num_vars + 2; it is set to 3")


def fixture_root(profile):
    return TINY_FIXTURES if SCOPES[profile].get("tiny") else FIXTURES


def geometry_hashes(profile):
    root = fixture_root(profile)
    return {key: hashlib.sha256((root / key / f"{key}.h5").read_bytes()).hexdigest()
            for key in ("hydrogen", *AMINO_GRID)}


def specification(name, profile="tiny-grid"):
    require(profile in SCOPES and name in WORKLOADS, f"Unknown smoke selection {profile}/{name}")
    filename = "02_noise_resilience.json" if name == "noise" else "01_trainability.json"
    spec = json.loads((CAMPAIGN / filename).read_text())
    if name == "amino":
        del spec["args"]["molecule"]
        samples = {"campaign": AMINO_SAMPLE, "tiny": ("gln", "glu"), "exact": ("gln", "glu")}
        spec["args"]["molecules"] = list(samples.get(profile, AMINO_GRID))
    if profile in ("tiny-grid", "tiny", "exact"):
        settings = {
            "hydrogen": {"param-targets": [2, 21], "eval-budget": 44, "max-iter": 12},
            "amino": {"param-targets": [1], "eval-budget": 8,
                      "max-iter": 8, "qubit-adapt-max-gradient-candidates": 2},
            "noise": {"max-iter": 24, "noise-strengths": [0.0, 0.01]},
        }
        spec["args"].update(settings[name])
    return spec


def validate_workload(name, folder, *, profile="tiny-grid"):
    from smoke.representative_checks import (
        validate_noise, validate_short_batch_objectives, validate_short_hydrogen, validate_trainability,
    )

    spec = specification(name, profile)["args"]
    tiny_inputs = SCOPES[profile].get("tiny", False)
    if name == "noise":
        validate_noise(folder, args=spec, tiny_inputs=tiny_inputs)
    else:
        batch = name == "amino"
        molecules = tuple(spec["molecules"]) if batch else (spec["molecule"],)
        validate_trainability(folder, molecules, batch=batch,
                              targets=spec["param-targets"], args=spec, tiny_inputs=tiny_inputs)
    devices = []
    for line in (folder / "run.log").read_text().splitlines():
        if "Created device:" in line:
            devices.append(line.partition("Created device: ")[2])
        if name == "amino" and line.endswith(MAXFUN_WARNING):
            continue
        require(not any(w in line for w in ("[ERROR]", "[WARNING]", "Warning:", "Traceback (most recent call last)")),
                f"{name} diagnostic: {line}")
    expected = {"lightning.qubit", "default.mixed"} if name == "noise" else {"lightning.qubit"}
    require({s.split(" |")[0] for s in devices} == expected and all("shots=None" in s for s in devices),
            f"{name} analytic backend mismatch: {set(devices)}")

    counts = Counter()
    with (folder / "events.jsonl").open() as stream:
        for line in stream:
            e = json.loads(line)
            require(e["workload"] == name and isinstance(e["callback"], bool)
                    and len(e["parameters"]) == 64 and len(bytes.fromhex(e["parameters"])) == 32
                    and math.isfinite(float.fromhex(e["energy"]))
                    and isinstance(e["operators"], list) and all(type(i) is int and i >= 0 for i in e["operators"]),
                    f"Invalid {name} event")
            if name == "noise":
                require(e["molecule"] == spec["molecule"] and e["algorithm"] == spec["algorithm"],
                        "Unexpected noise event")
                key = (e["noise_model"] or "none", e["strength"], e["callback"])
            else:
                key = (e["molecule"], e["algorithm"], e["target"], e["callback"])
            counts[key] += 1
    if name == "noise":
        optima = [r for r in read_rows(folder / "results.csv") if r["noise_model"] == "none" or float(r["noise_strength"])]
        cases = {(r["noise_model"], float(r["noise_strength"])) for r in optima}
        require(set(counts) == {(*case, flag) for case in cases for flag in (True, False)}, "Incomplete noise event grid")
        for r in optima:
            case = (r["noise_model"], float(r["noise_strength"]))
            require(counts[*case, True] == int(r["n_iterations"]) and
                    1 < counts[*case, False] <= spec["max-iter"], f"Noise optimizer count differs for {case}")
    else:
        rows = read_rows(folder / ("all_molecules_results.csv" if name == "amino" else "results.csv"))
        expected_counts = Counter()
        for row in rows:
            key = (row["molecule"], row["algorithm"], int(row["target_n_params"]))
            expected_counts[(*key, False)] = int(row["n_cost_evals"])
            expected_counts[(*key, True)] = int(row["n_callback_evals"])
        require(counts == expected_counts, f"{name} objective or callback stream differs from results")
        if name == "hydrogen" and profile in ("tiny", "tiny-grid"):
            validate_short_hydrogen(folder)
        if name == "amino" and profile in ("tiny", "tiny-grid"):
            validate_short_batch_objectives(folder, molecules)


def source_fingerprint(source):
    digest = hashlib.sha256()
    files = [source / "config.py"]
    for package in ("algorithms", "core", "experiments", "visualize_molecules", "active_space_truncation"):
        files.extend(sorted((source / package).glob("*.py")))
    for path in files:
        digest.update(path.relative_to(source).as_posix().encode() + b"\0")
        digest.update(hashlib.sha256(path.read_bytes()).digest())
    return digest.hexdigest()


def run_workload(name, directory, *, profile="tiny-grid"):
    require(name in WORKLOADS, f"Unknown workload {name}")
    directory = Path(directory).resolve()
    folder = directory / name
    folder.mkdir(parents=True, exist_ok=False)
    source = Path(os.getenv("QMPROT_FRAMEWORK", str(FRAMEWORK))).resolve()
    spec = specification(name, profile)
    args = argv_from_args(spec)
    command = [sys.executable, str(FRAMEWORK / "smoke/record.py"), spec["module"],
               str(folder / "events.jsonl"), *args[3:], output_flag(spec), str(folder)]
    env = dict(os.environ, PYTHONPATH=str(source), QMPROT_GEOMETRIES=str(fixture_root(profile)),
               QMPROT_WORKLOAD=name, OMP_NUM_THREADS="1", OPENBLAS_NUM_THREADS="1",
               MKL_NUM_THREADS="1", VECLIB_MAXIMUM_THREADS="1", MPLBACKEND="Agg")
    start = time.monotonic()
    print(f"Running {name} in {folder}", flush=True)
    with (folder / "run.log").open("x") as log:
        rc = subprocess.run(command, cwd=folder, env=env, stdout=log, stderr=subprocess.STDOUT).returncode
    seconds = time.monotonic() - start
    if rc:
        raise RuntimeError(f"{name} exited {rc}. See {folder / 'run.log'}")
    validate_workload(name, folder, profile=profile)
    provenance = dict(seconds=seconds, experiment=spec, platform=platform.platform(),
                      python=platform.python_version(), source_sha256=source_fingerprint(source),
                      geometry_set=fixture_root(profile).name, geometries=geometry_hashes(profile),
                      packages={dep: importlib.metadata.version(dep) for dep in
                                ("pennylane", "pennylane-lightning", "numpy", "scipy", "pyscf")})
    (folder / "measurement.json").write_text(json.dumps(provenance, indent=2) + "\n")


def saved_profile(directory):
    scope = json.loads((Path(directory) / "suite.json").read_text())
    for profile, expected in SCOPES.items():
        if scope == expected and all(type(v) is bool for v in scope.values()):
            return profile
    raise ValueError(f"Invalid smoke suite selection {scope}")


def validate(directory):
    directory = Path(directory)
    profile = saved_profile(directory)
    geometries = geometry_hashes(profile)
    combined = hashlib.sha256()
    for name in WORKLOADS:
        folder = directory / name
        validate_workload(name, folder, profile=profile)
        info = json.loads((folder / "measurement.json").read_text())
        require(info["experiment"] == specification(name, profile) and info["seconds"] > 0, f"{name} measurement")
        require(info["geometries"] == geometries, f"{name} geometry hashes differ from the selected inputs")
        if "geometry_set" in info or SCOPES[profile].get("tiny"):
            require(info.get("geometry_set") == fixture_root(profile).name, f"{name} geometry set differs")
        with (folder / "events.jsonl").open("rb") as events:
            while block := events.read(1024 * 1024):
                combined.update(block)
    with (directory / "events.jsonl").open("rb") as events:
        require(hashlib.file_digest(events, "sha256").digest() == combined.digest(), "Combined event stream differs")


def run(directory, *, profile="tiny-grid"):
    directory = Path(directory).resolve()
    directory.mkdir(parents=True, exist_ok=False)
    (directory / "suite.json").write_text(json.dumps(SCOPES[profile]) + "\n")
    for name in WORKLOADS:
        run_workload(name, directory, profile=profile)
    with (directory / "events.jsonl").open("xb") as aggregate:
        for name in WORKLOADS:
            with (directory / name / "events.jsonl").open("rb") as events:
                shutil.copyfileobj(events, aggregate)
    validate(directory)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run the representative campaign suite.")
    group = parser.add_mutually_exclusive_group()
    group.add_argument("--output", type=Path, help="Fresh run directory")
    group.add_argument("--check", type=Path, help="Validate an existing suite")
    parser.add_argument("--reference", type=Path, help="Strictly compare against a saved suite")
    profiles = parser.add_mutually_exclusive_group()
    profiles.add_argument("--quick", dest="profile", action="store_const", const="tiny",
                          help="Short budgets with just gln and glu in the tiny batch")
    profiles.add_argument("--real-inputs", dest="profile", action="store_const", const="exact",
                          help="Short budgets with the published hydrogen, gln and glu geometries")
    profiles.add_argument("--tiny-full", dest="profile", action="store_const", const="tiny-full",
                          help="Use all production campaign settings and molecule IDs with tiny geometries")
    profiles.add_argument("--campaign", dest="profile", action="store_const", const="campaign",
                          help="Use full budgets on three representative published amino acids")
    profiles.add_argument("--full", dest="profile", action="store_const", const="full",
                          help="Use full budgets on all 19 documented published amino acids")
    parser.set_defaults(profile="tiny-grid")
    args = parser.parse_args()
    if args.reference:
        validate(args.reference)
    profile = args.profile
    if args.check:
        directory = args.check
        validate(directory)
    else:
        directory = args.output or FRAMEWORK / "results/smoke" / f"representative-{uuid4().hex}"
        if args.reference and saved_profile(args.reference) != profile:
            parser.error("Reference workload differs from the selected workload")
        run(directory, profile=profile)
    if args.reference:
        from smoke.comparison import compare
        print(compare(args.reference, directory))
    print(f"PASS: representative campaign suite verified in {directory}")
