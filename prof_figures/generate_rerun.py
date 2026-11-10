"""Generate appendix views from five explicit saved experiment directories."""
import argparse
import hashlib
import json
from pathlib import Path
import re
import subprocess
import sys
from datetime import datetime, timezone

import pandas as pd

HERE = Path(__file__).resolve().parent


def main():
    p = argparse.ArgumentParser(description=__doc__)
    for name in ("noise", "initialization", "trainability", "adapt", "prefix"):
        p.add_argument(name, type=Path, help=f"Experiment output directory: {name}")
    p.add_argument("--label", default="Fresh rerun", help="Caption provenance; do not claim an unverified geometry basis")
    p.add_argument("--dry-run", action="store_true", help="Validate input selection and list outputs without writing")
    args = p.parse_args()
    roots = [getattr(args, name).resolve() for name in
             ("noise", "initialization", "trainability", "adapt", "prefix")]
    noise, init, train, adapt, prefix = roots
    jobs = []

    def add(root, folder, source, suffix="", *extra):
        output = root / f"prof_rerun_{folder}{suffix}.png"
        jobs.append((root, [sys.executable, str(HERE / folder / "generate.py"),
                           "--input", str(source), "--output", str(output),
                           "--label", args.label, *extra]))

    def table(root, filename):
        path = root / filename
        if not path.is_file():
            p.error(f"Missing input: {path}")
        frame = pd.read_csv(path)
        if frame.empty:
            p.error(f"Empty input: {path}")
        return path, frame

    def molecules(frame):
        values = sorted(frame.molecule.dropna().astype(str).unique())
        if not values or any(not re.fullmatch(r"[A-Za-z0-9_-]+", value) for value in values):
            p.error("Missing or unsafe molecule identifiers")
        return values

    source, frame = table(noise, "batch_all_results.csv")
    add(noise, "figure_17_noise_cosine_similarity", source)
    for mol in molecules(frame):
        add(noise, "additional_01_noise_energy_comparison", source, f"_{mol}", "--molecule", mol)

    # Determine the convergence figure from saved configuration, never from a folder name.
    configs = sorted(init.glob("*/run_config.json"))
    if (init / "run_config.json").is_file():
        configs.append(init / "run_config.json")
    optimizers = {str(json.loads(path.read_text()).get("optimizer", "")).upper() for path in configs}
    if len(optimizers) != 1 or not optimizers <= {"BAYESIAN", "COBYLA"}:
        p.error(f"Initialization requires consistently BAYESIAN or COBYLA run_config.json files: {optimizers}")
    trace_name = ("convergence_traces_all.csv" if (init / "convergence_traces_all.csv").is_file()
                  else "all_molecules_convergence_traces.csv")
    source, _ = table(init, trace_name)
    from figure_style import load_table
    load_table(init, "optimization_results.csv")  # Fail before creating any figures.
    folder = ("figure_14_bayesian_initialization_convergence" if "BAYESIAN" in optimizers
              else "figure_13_cobyla_initialization_convergence")
    add(init, folder, source)
    add(init, "figure_15_initialization_final_energies", init)

    source, frame = table(train, "all_molecules_results.csv")
    add(train, "figure_16_trainability_evaluation_cost", source)
    targets = sorted(pd.to_numeric(frame.target_n_params, errors="raise").dropna().unique())
    if not targets or any(value <= 0 or value != int(value) for value in targets):
        p.error("Trainability targets must be positive integers")
    for target in targets:
        for folder in ("additional_02_trainability_energy", "additional_03_trainability_runtime",
                       "additional_05_trainability_combined"):
            add(train, folder, source, f"_target_{int(target)}", "--target", str(int(target)))

    source, frame = table(adapt, "batch_all_results.csv")
    add(adapt, "additional_04_adapt_operator_cost", source)
    if "status" in frame:
        frame = frame[frame.status.astype(str).str.lower() != "failed"]
    for mol in molecules(frame):
        add(adapt, "additional_06_adapt_energy_and_cost", source, f"_{mol}", "--molecule", mol)

    source, _ = table(prefix, "summary_by_algorithm_term_count.csv")
    add(prefix, "additional_07_prefix_energy", source)
    add(prefix, "additional_07_prefix_energy", source, "_without_qaoa", "--exclude-qaoa")

    if args.dry_run:
        for _, command in jobs:
            print(command[command.index("--output") + 1])
        print(f"Validated selection: {len(jobs)} PNG/PDF pairs. No files written.")
        return 0

    reports = {root: {"created_utc": datetime.now(timezone.utc).isoformat(),
                      "label": args.label, "figures": []} for root in roots}
    failures = 0
    for root, command in jobs:
        output = Path(command[command.index("--output") + 1])
        print(f"Generating {output.name}", flush=True)
        result = subprocess.run(command, capture_output=True, text=True)
        entry = {"command": command, "returncode": result.returncode,
                 "stdout": result.stdout, "stderr": result.stderr}
        source = Path(command[command.index("--input") + 1])
        files = [source] if source.is_file() else sorted(source.glob("*/optimization_results.csv")) + sorted(source.glob("*/run_config.json"))
        entry["input_sha256"] = {str(path): hashlib.sha256(path.read_bytes()).hexdigest() for path in files}
        reports[root]["figures"].append(entry)
        if result.returncode:
            failures += 1
            print(result.stderr, file=sys.stderr)
    for root, report in reports.items():
        (root / "prof_rerun_manifest.json").write_text(json.dumps(report, indent=2) + "\n")
    print(f"Finished: {len(jobs) - failures}/{len(jobs)} PNG/PDF pairs. See prof_rerun_manifest.json in each output root.")
    return 1 if failures else 0


if __name__ == "__main__":
    sys.exit(main())
