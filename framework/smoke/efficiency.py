# ABOUTME: Checks native before-and-after trainability and prefix runs against their frozen inputs.
# ABOUTME: Verifies complete energy and angle traces and reports whole-process timings and case costs.
import argparse
import csv
from hashlib import sha256
import json
from pathlib import Path
import re

from smoke.reproductions.study_checks.paired_prefixes import checked_run


def _rows(path):
    with path.open() as stream:
        return list(csv.DictReader(stream))


def _digest(path):
    return sha256(path.read_bytes()).hexdigest()


def _scientific_rows(rows, excluded):
    return [{k: v for k, v in row.items() if k not in excluded} for row in rows]


def _require(condition, message):
    if not condition:
        raise ValueError(message)


def _seconds(directory, workload):
    seconds = {}
    for version in ("baseline", "candidate"):
        log = (directory / f"{version}-{workload}.log").read_text()
        _require(not any(marker in log for marker in ("[WARNING]", "[ERROR]", "Traceback (most recent call last)")),
                 f"Benchmark logged a warning or failure in {version} {workload}")
        matches = re.findall(r"^real ([0-9.]+)$", log, flags=re.MULTILINE)
        _require(len(matches) == 1, f"Missing elapsed time for {version} {workload}")
        seconds[version] = float(matches[0])
    return seconds


def trainability(directory):
    versions = ("baseline", "candidate")
    runs = [directory / f"{v}-trainability" for v in versions]
    signed = []
    tables = []
    sources = {}
    for version, run in zip(versions, runs, strict=True):
        batch = json.loads((run / "batch_manifest.json").read_text())
        _require(batch["molecules_selected"] == batch["molecules_succeeded"] == ["gly"]
                 and not batch["molecules_failed"], "Incomplete glycine trainability batch")
        rows = _rows(run / "gly/results.csv")
        expected = {(a, str(t)) for a in batch["algorithms"] for t in batch["param_targets"]}
        _require(len(rows) == len(expected) and {(r["algorithm"], r["target_n_params"]) for r in rows} == expected,
                 "Duplicated or missing trainability case")
        tables.append(rows)
        runtime = json.loads((run / "gly/runtime.json").read_text())
        sources[version] = _digest(run / "gly/framework-source.zip")
        _require(runtime["source_sha256"] == sources[version], "Incorrect trainability source fingerprint")
        artifacts = {f: _digest(run / "gly" / f) for f in ("inputs/gly.npz", "run_config.json", "endpoints.jsonl",
                     "eval_traces.csv", "convergence_traces.csv", "model_reference.json")}
        artifacts.update({f: _digest(run / f) for f in ("batch_manifest.json", "all_molecules_convergence_traces.csv",
                          "all_molecules_eval_traces.csv", "all_molecules_cost_to_accuracy_summary.csv")})
        signed.append(artifacts)
    _require(signed[0] == signed[1], "Trainability input, endpoint, reference, or trace bytes differ")
    excluded = {"runtime_seconds"}
    if "n_reused_reference_probes" not in tables[0][0]:
        excluded.add("n_reused_reference_probes")
    _require(_scientific_rows(tables[0], excluded) == _scientific_rows(tables[1], excluded),
             "Trainability scientific rows differ")
    aggregates = [_scientific_rows(_rows(run / "all_molecules_results.csv"), excluded) for run in runs]
    _require(aggregates[0] == aggregates[1], "Trainability aggregate rows differ")
    events = [directory / f"{v}-events.jsonl" for v in versions]
    _require(_digest(events[0]) == _digest(events[1]), "Objective calls, parameters, energies, or operators differ")
    with events[0].open() as stream:
        count = sum(1 for _ in stream)
    _require(count == sum(int(row["n_cost_evals"]) + int(row["n_callback_evals"]) for row in tables[0]),
             "Trace length differs from logical evaluation count")
    cases = [dict(algorithm=a["algorithm"], target=int(a["target_n_params"]), evaluations=int(a["n_cost_evals"]),
                  baseline_seconds=float(a["runtime_seconds"]), candidate_seconds=float(b["runtime_seconds"]),
                  reused_reference_probes=int(b["n_reused_reference_probes"])) for a,b in zip(*tables, strict=True)]
    return dict(seconds=_seconds(directory, "trainability"), source_sha256=sources, cases=cases, events=count,
                events_sha256=_digest(events[0]), artifact_sha256=signed[0])


def prefixes(directory):
    first, second = [directory / f"{v}-prefix" for v in ("baseline", "candidate")]
    expected, receipt = checked_run(first)
    measured, candidate_receipt = checked_run(second)
    sources = dict(baseline=receipt["source_sha256"], candidate=candidate_receipt["source_sha256"])
    for path, digest in ((first, sources["baseline"]), (second, sources["candidate"])):
        _require(_digest(path / "framework-source.zip") == digest, "Incorrect prefix source fingerprint")
    _require(expected.keys() == measured.keys(), "Prefix case identities differ")
    _require((first / "run_config.json").read_bytes() == (second / "run_config.json").read_bytes(),
             "Prefix configurations differ")
    for key, value in measured.items():
        for name in ("n_materialized_commutators", "n_screen_support_zeros"):
            if name not in expected[key]["metadata"]:
                value["metadata"].pop(name, None)
        _require(json.dumps(value, sort_keys=True) == json.dumps(expected[key], sort_keys=True),
                 f"Prefix scientific case differs for {key}")
    before, after = (_rows(path / "batch_all_results.csv") for path in (first, second))
    per_method = {}
    for a, b in zip(before, after, strict=True):
        method = a["algorithm"]
        stats = per_method.setdefault(method, dict(baseline_seconds=0., candidate_seconds=0.))
        stats["baseline_seconds"] += float(a.pop("runtime_seconds"))
        stats["candidate_seconds"] += float(b.pop("runtime_seconds"))
        _require(a == b, f"Prefix CSV differs for {method}")
    aggregates = [_scientific_rows(_rows(path / "summary_by_algorithm_term_count.csv"), {"mean_runtime_seconds"})
                  for path in (first, second)]
    _require(aggregates[0] == aggregates[1], "Prefix aggregate rows differ")
    scientific_sha = sha256(json.dumps(list(expected.values()), sort_keys=True).encode()).hexdigest()
    return dict(seconds=_seconds(directory, "prefix"), source_sha256=sources,
                cases=len(expected), cases_sha256=scientific_sha, input_sha256=receipt["input_sha256"],
                solver_seconds=per_method)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("directory", type=Path)
    parser.add_argument("--trainability-only", action="store_true")
    args = parser.parse_args()
    report = {"trainability": trainability(args.directory)}
    if not args.trainability_only:
        report["prefix"] = prefixes(args.directory)
    print(json.dumps(report, indent=2, allow_nan=False))
