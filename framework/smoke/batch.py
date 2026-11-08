# ABOUTME: Runs a small trainability batch on published glycine and arginine geometries.
# ABOUTME: Checks the saved batch and per-molecule artifacts against the requested work.
import argparse
import csv
import json
import math
import os
import subprocess
import sys
from collections import Counter
from pathlib import Path
from uuid import uuid4

FRAMEWORK = Path(__file__).resolve().parents[1]
MOLECULES = ("gly", "arg")
ALGORITHMS = ("qubit_adapt_vqe", "hardware_efficient_vqe")


def read_rows(path):
    with path.open(newline="") as stream:
        return list(csv.DictReader(stream))


def require(condition, message):
    if not condition:
        raise ValueError(message)


def validate(directory):
    from PIL import Image

    manifest = json.loads((directory / "batch_manifest.json").read_text())
    for field in ("molecules_requested", "molecules_selected", "molecules_succeeded"):
        require(Counter(manifest[field]) == Counter(MOLECULES), f"{field}: {manifest[field]}")
    require(not manifest["molecules_failed"], f"Failed molecules: {manifest['molecules_failed']}")
    rows = read_rows(directory / "all_molecules_results.csv")
    expected = [(mol, algo, "1") for mol in MOLECULES for algo in ALGORITHMS]
    keys = [(r["molecule"], r["algorithm"], r["target_n_params"]) for r in rows]
    require(Counter(keys) == Counter(expected), f"Unexpected result combinations: {keys}")
    require(manifest["n_rows"] == len(rows), "Manifest row count differs from CSV")
    all_traces = read_rows(directory / "all_molecules_convergence_traces.csv")
    combined_traces = []
    for molecule in MOLECULES:
        folder = directory / molecule
        config = json.loads((folder / "run_config.json").read_text())
        require(config["molecule"] == molecule and config["n_qubits_final"] == 12,
                f"Unexpected run configuration for {molecule}")
        reference = config["casci_energy"]
        require(math.isfinite(reference) and math.isfinite(config["hf_energy"]),
                f"Missing reference energy for {molecule}")
        mol_rows = [r for r in rows if r["molecule"] == molecule]
        require(read_rows(folder / "results.csv") == mol_rows, f"Results differ for {molecule}")
        traces = read_rows(folder / "convergence_traces.csv")
        require({r["algorithm"] for r in traces} == set(ALGORITHMS), f"Missing traces for {molecule}")
        for row in mol_rows:
            case = f"{molecule}/{row['algorithm']}"
            require(int(row["actual_n_params"]) == 1 and 0 < int(row["n_cost_evals"]) <= 64,
                    f"{case}: parameter count or evaluation count out of range")
            energy = float(row["final_energy"])
            require(math.isclose(energy - reference, float(row["error_vs_casci"]),
                                 rel_tol=0, abs_tol=1e-8) and math.isfinite(energy),
                    f"{case}: inconsistent final energy or CASCI error")
            trace = [t for t in traces if t["algorithm"] == row["algorithm"]]
            require(all(t["target_n_params"] == "1" and int(t["iteration"]) == i
                        and math.isfinite(float(t["energy"])) for i, t in enumerate(trace)),
                    f"{case}: invalid convergence trace")
            require(math.isclose(float(trace[-1]["energy"]), energy, rel_tol=0, abs_tol=1e-8),
                    f"{case}: trace and final energy differ")
        combined_traces.extend(dict(molecule=molecule, **t) for t in traces)
    require(all_traces == combined_traces, "Aggregate and per-molecule traces differ")
    plot = directory / "all_molecules_cost_evals_vs_params.png"
    with Image.open(plot) as chart:
        require(chart.format == "PNG", f"Invalid batch plot: {plot}")
        chart.verify()
    lines = (directory / "batch.log").read_text().splitlines()
    devices = [line for line in lines if "Created device:" in line]
    require(len(devices) >= len(expected) and all("Created device: lightning.qubit " in line
                                                  for line in devices), "Unexpected simulation backend")
    for line in lines:
        require(not any(marker in line for marker in ("[ERROR]", "[WARNING]", "Traceback (most recent call last)")),
                f"Batch diagnostic: {line}")


def run(directory):
    directory = directory.resolve()
    directory.mkdir(parents=True, exist_ok=False)
    env = dict(os.environ, QMPROT_GEOMETRIES=str(FRAMEWORK / "smoke/geometries"),
               OMP_NUM_THREADS="1", OPENBLAS_NUM_THREADS="1", VECLIB_MAXIMUM_THREADS="1",
               MPLBACKEND="Agg")
    command = [sys.executable, "-m", "experiments.trainability", "--molecules", *MOLECULES,
               "--algorithms", *ALGORITHMS, "--param-targets", "1", "--basis", "sto-3g",
               "--max-iter", "12", "--eval-budget", "64", "--qubit-adapt-max-gradient-candidates", "8",
               "--output-root", str(directory)]
    log = directory / "batch.log"
    print(f"Running trainability batch. Log: {log}", flush=True)
    with log.open("w") as stream:
        subprocess.run(command, cwd=FRAMEWORK, env=env, stdout=stream, stderr=subprocess.STDOUT, check=True)
    validate(directory)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run or validate the two-molecule trainability smoke test.")
    parser.add_argument("--check", type=Path, help="Validate a previously saved smoke batch")
    args = parser.parse_args()
    if args.check:
        destination = args.check
        validate(destination)
    else:
        destination = FRAMEWORK / "results/smoke" / uuid4().hex
        run(destination)
    print(f"PASS: four trainability results verified in {destination}")
