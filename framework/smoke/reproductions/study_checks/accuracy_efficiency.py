# ABOUTME: Times the native adaptive-size batch with plots and compares every real optimizer request.
# ABOUTME: Verifies inputs, attainment, fitted gates, energies, charge, aggregate tables and image files.
import csv
from hashlib import sha256
import json
import os
from pathlib import Path
import statistics
import subprocess
import sys
from time import perf_counter


def record(path):
    from experiments import accuracy_vs_params
    from experiments._common import CostEvalCounter
    from smoke.record import record as record_energy

    attach = CostEvalCounter.attach
    with path.open("x") as stream:
        def attach_recording(counter, solver):
            fields = dict(algorithm=solver.name, molecule=solver.hamiltonian.molecule.name)
            record_energy(solver, stream, fields, lambda: counter._in_callback)
            attach(counter, solver)

        CostEvalCounter.attach = attach_recording
        accuracy_vs_params.main()


def digest(data):
    return sha256(data).hexdigest()


def fingerprint(path):
    files = {}
    for f in sorted(path.rglob("*")):
        if not f.is_file() or f.name in ("runtime.json", "framework-source.zip"):
            continue
        data = f.read_bytes()
        if f.suffix == ".csv":
            with f.open() as table:
                reader = csv.DictReader(table)
                fields = [k for k in reader.fieldnames if k != "cumulative_runtime_s"]
                rows = [[line[k] for k in fields] for line in reader]
            data = json.dumps([fields, rows]).encode()
        if f.name == "run_config.json":
            cfg = json.loads(data)
            cfg.pop("solver_runtime_s", None)
            data = json.dumps(cfg, sort_keys=True).encode()
        files[f.relative_to(path).as_posix()] = digest(data)
    return files


def scientific_record(case, events):
    files = fingerprint(case)
    files["objective_events"] = digest(events.read_bytes())
    return files


def run(control, destination):
    framework = Path(__file__).resolve().parents[3]
    launch = Path(__file__).resolve()
    destination.mkdir(parents=True, exist_ok=False)
    by_case = {}
    for cohort, methods, optimizers, molecules, checkpoints, order in (
        ("hydrogen", ("adapt_vqe", "qubit_adapt_vqe"), ("COBYLA", "BAYESIAN"), ("hydrogen",), [1, 2, 5], "ABBA"),
        ("hydrogen-glycine", ("qubit_adapt_vqe",), ("COBYLA",), ("hydrogen", "gly"), [1], "AB"),
    ):
        for method in methods:
            for opt in optimizers:
                key = f"{cohort}-{method}-{opt}"
                summaries = []
                for i, variant in enumerate(order):
                    source = control if variant == "A" else framework
                    stem = f"{key}-{i}-{variant}"
                    case = destination / stem
                    events = destination / (stem + ".jsonl")
                    args = [sys.executable, str(launch), "record", str(events), "--molecules", *molecules, "--algorithm",
                            method, "--optimizer", opt, "--cs-target-qubits", "0", "--inner-max-iter", "24",
                            "--seed", "42", "--k-list", *map(str, checkpoints), "--output-dir", str(case)]
                    env = {**os.environ, "PYTHONPATH": str(source), "QMPROT_WORKLOAD": "accuracy-vs-size",
                           "QMPROT_GEOMETRIES": str(framework / "smoke/geometries"), "OMP_NUM_THREADS": "1",
                           "OPENBLAS_NUM_THREADS": "1", "MKL_NUM_THREADS": "1", "VECLIB_MAXIMUM_THREADS": "1",
                           "MPLBACKEND": "Agg"}
                    log_path = destination / (stem + ".log")
                    with log_path.open("w") as log:
                        start = perf_counter()
                        subprocess.run(args, cwd=source, env=env, stdout=log, stderr=subprocess.STDOUT, check=True)
                        elapsed = perf_counter() - start
                    log = log_path.read_text()
                    if any(s in log for s in ("Warning:", "[WARNING]", "[ERROR]", "Traceback")):
                        raise ValueError(f"Native size study logged a warning or failure in {log_path}")
                    if json.loads((case / "batch_failures.json").read_text()) != []:
                        raise ValueError(f"Failed molecule in {case}")
                    files = scientific_record(case, events)
                    if i == 0:
                        reference = files
                    elif files != reference:
                        different = [k for k in set(files) | set(reference) if files.get(k) != reference.get(k)]
                        raise ValueError(f"Size study parity failed for {stem}: {different}")
                    row = dict(variant=variant, cli_seconds=elapsed, cost_and_callback_events=len(events.read_text().splitlines()),
                               fits={})
                    for molecule in molecules:
                        sub = case / f"{molecule}_{method}"
                        cfg = json.loads((sub / "run_config.json").read_text())
                        runtime = json.loads((sub / "runtime.json").read_text())
                        row["fits"][molecule] = dict(
                            source_sha256=runtime["source_sha256"], solver_seconds=cfg["solver_runtime_s"],
                            attained_operators=cfg["actual_max_k"], cost_evals=cfg["total_cost_evals"],
                        )
                    summaries.append(row)
                    print(stem, elapsed, flush=True)
                medians = {variant: statistics.median(x["cli_seconds"] for x in summaries if x["variant"] == variant)
                           for variant in "AB"}
                by_case[key] = dict(checkpoints=checkpoints, molecules=molecules, order=order, runs=summaries,
                                    scientific_digests=reference, end_to_end_speedup=medians["A"]/medians["B"])
                (destination / "summary.json").write_text(json.dumps(by_case, indent=2, allow_nan=False) + "\n")


def summarize(destination, output):
    cases = json.loads((destination / "summary.json").read_text())
    result = dict(
        workload="experiments.accuracy_vs_params, native batch CLI with figures, canonical active space",
        baseline="ab1954f", platform="macOS arm64", inner_max_iter=24, seed=42, threads=1,
        timing="Fresh process through plot writes, including streaming a digest of every cost request.",
        parity="Compared input snapshots, scientific CSV headers and cells, scientific JSON, PNG bytes, "
               "and all objective and callback events. Excluded runtime.json, framework-source.zip, "
               "solver_runtime_s and cumulative_runtime_s.",
        source_sha256={}, cases={},
    )
    for key, case in cases.items():
        runs = case["runs"]
        for i, row in enumerate(runs):
            variant = row["variant"]
            stem = f"{key}-{i}-{variant}"
            files = scientific_record(destination / stem, destination / (stem + ".jsonl"))
            if i == 0:
                reference = files
            elif files != reference:
                raise ValueError(f"Scientific output differs for {stem}")
            for fit in row["fits"].values():
                source = result["source_sha256"].setdefault(variant, fit["source_sha256"])
                if source != fit["source_sha256"]:
                    raise ValueError(f"Inconsistent source for {stem}")
        result["cases"][key] = dict(
            run_order=case["order"], requested_operators=case["checkpoints"],
            cli_seconds=[r["cli_seconds"] for r in runs],
            end_to_end_speedup=case["end_to_end_speedup"],
            events_per_run=runs[0]["cost_and_callback_events"],
            events_sha256=reference["objective_events"],
            files_compared=len(reference) - 1, plots_compared=sum(p.endswith(".png") for p in reference),
            manifest_sha256=digest(json.dumps(reference, sort_keys=True).encode()), molecules={},
        )
        for mol in case["molecules"]:
            fit = runs[0]["fits"][mol]
            result["cases"][key]["molecules"][mol] = dict(
                attained_operators=fit["attained_operators"], cost_evals=fit["cost_evals"],
                solver_seconds=[r["fits"][mol]["solver_seconds"] for r in runs],
            )
        case["scientific_digests"] = reference
    (destination / "summary.json").write_text(json.dumps(cases, indent=2, allow_nan=False) + "\n")
    output.write_text(json.dumps(result, indent=2, allow_nan=False) + "\n")


if __name__ == "__main__":
    if sys.argv[1] == "record":
        output = Path(sys.argv[2])
        sys.argv[:] = ["experiments.accuracy_vs_params", *sys.argv[3:]]
        record(output)
    elif sys.argv[1] == "summarize":
        summarize(Path(sys.argv[2]), Path(sys.argv[3]))
    else:
        run(Path(sys.argv[1]).resolve(), Path(sys.argv[2]).resolve())
