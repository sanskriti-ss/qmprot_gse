# ABOUTME: Rebuilds stored trainability endpoints and first accepted crossings on their frozen inputs.
# ABOUTME: Checks physical-sector measurements, accepted generators, reference energies and case coverage.
import argparse
import csv
from hashlib import sha256
import json
from pathlib import Path

import numpy as np

from algorithms import get_algorithm
from experiments._common import threshold_label
from experiments.cost_to_accuracy import summarize
from experiments.endpoint_sector import model_reference, summarize_endpoint
from experiments.provenance import load_problem


def _close(actual, expected, description):
    if expected is None or actual is None:
        if expected is not actual:
            raise ValueError(f"Unavailable value differs for {description}")
    elif not np.isclose(actual, expected, rtol=0., atol=1e-8):
        raise ValueError(f"{description} differs: {actual} vs {expected}")


def _restore_selection(solver, operators):
    if not hasattr(solver, "operator_pool"):
        if operators:
            raise ValueError("Fixed circuit endpoint contains adaptive operators")
        return []
    if any(type(i) is not int or not 0 <= i < len(solver.operator_pool) for i in operators):
        raise ValueError("Adaptive endpoint contains an unknown pool index")
    solver.selected_operators = list(operators)
    return [list(g) if isinstance(g, tuple) else g for g in (solver.operator_pool[i] for i in operators)]


def _replay_case(problem, record, cfg, csv_row):
    solver = get_algorithm(record["algorithm"])(problem.hamiltonian, **record["settings"])
    solver.build_ansatz()
    generators = _restore_selection(solver, record["selected_operators"])
    if generators != record["generators"]:
        raise ValueError("Recorded generator differs from its saved pool index")
    angles = record["parameters"]
    if len(angles) != int(csv_row["actual_n_params"]):
        raise ValueError("Recorded parameter count differs from fit")
    outcome = summarize_endpoint(problem, solver, angles)
    energy = outcome["rescored_energy"]
    _close(energy, record["reported_energy"], "terminal energy")
    _close(energy, float(csv_row["final_energy"]), "CSV energy")
    if any(outcome[key] != record[key] for key in ("sector_status", "state_kind", "expected_electron_count")):
        raise ValueError("Endpoint sector coverage differs from saved result")
    for key in ("rescored_energy", "mean_electron_count", "wrong_electron_probability", "mean_spin_squared",
                "mean_spin_z", "target_spin_z_probability"):
        _close(outcome[key], record[key], key)
    leakage = csv_row["wrong_electron_probability"]
    _close(outcome["wrong_electron_probability"], float(leakage) if leakage else None, "CSV charge leakage")
    weights = outcome["electron_count_probabilities"]
    if weights is None or record["electron_count_probabilities"] is None:
        if weights is not record["electron_count_probabilities"]:
            raise ValueError("Endpoint population availability differs")
    else:
        if weights.keys() != record["electron_count_probabilities"].keys():
            raise ValueError("Endpoint sector labels differ")
        for count, weight in weights.items():
            _close(weight, record["electron_count_probabilities"][count], f"N={count}")

    tolerances = {threshold_label(t): t for t in cfg["accuracy_thresholds_ha"]}
    crossings = dict.fromkeys(["hf", *tolerances])
    previous_index = 0
    witnesses = 0
    for event in record["accepted_states"]:
        index, value = event["eval_index"], event["energy"]
        if not previous_index <= index <= int(csv_row["n_cost_evals"]):
            raise ValueError("Accepted-state evaluation index is outside its fit")
        previous_index = index
        hits = []
        for key in crossings:
            if key == "hf":
                hit = problem.hf_energy is not None and value <= problem.hf_energy + cfg["hf_tolerance_ha"]
            else:
                hit = problem.casci_energy is not None and abs(value - problem.casci_energy) <= tolerances[key]
            if hit and crossings[key] is None:
                hits.append(key)
                crossings[key] = index
        if set(hits) != set(event.get("first_crossings", [])):
            raise ValueError("Accepted-state first crossings disagree with its energy")
        if hits and "parameters" not in event:
            raise ValueError("Accepted-state first crossing is missing its fitted parameters")
        if "parameters" in event:
            _restore_selection(solver, event.get("selected_operators", []))
            witness = summarize_endpoint(problem, solver, event["parameters"])
            _close(witness["rescored_energy"], value, "accepted-state energy")
            _close(witness["wrong_electron_probability"], event["wrong_electron_probability"], "accepted-state sector")
            witnesses += bool(hits)
    if crossings != record["accepted_evals_to"]:
        raise ValueError("Accepted-state threshold counts disagree with the retained history")
    for kind in ("accepted", "probe"):
        for key, count in record[f"{kind}_evals_to"].items():
            cell = csv_row[f"{kind}_evals_to_{key}"]
            if count != (int(cell) if cell else None):
                raise ValueError(f"CSV {kind} crossing differs for {key}")
    return witnesses


def _table_signatures(paths, table_name, result_rows=None):
    """Hash ordered rows within each fit, allowing a resumed batch to reorder fits."""
    groups = {}
    for path, molecule in paths:
        with path.open(newline="") as stream:
            for row in csv.DictReader(stream):
                if molecule is not None:
                    if table_name == "convergence_traces":
                        row["molecule"] = molecule
                    if row["molecule"] != molecule:
                        raise ValueError(f"Wrong molecule in {path}")
                    if table_name == "results":
                        result_rows.append(row)
                key = (row["molecule"], row["algorithm"], row["target_n_params"])
                if key not in groups:
                    groups[key] = sha256()
                groups[key].update(json.dumps(row, sort_keys=True).encode() + b"\n")
    return {key: digest.digest() for key, digest in groups.items()}


def _check_batch_tables(directory, molecules, batch):
    result_rows = []
    for name in ("results", "convergence_traces", "eval_traces"):
        children = ((directory / molecule / f"{name}.csv", molecule) for molecule in molecules)
        expected = _table_signatures(children, name, result_rows)
        aggregate = directory / f"all_molecules_{name}.csv"
        if _table_signatures([(aggregate, None)], name) != expected:
            raise ValueError(f"Trainability aggregate {name} differs from per-molecule tables")
    if batch["n_rows"] != len(result_rows):
        raise ValueError("Trainability batch row count differs from per-molecule tables")
    expected_summary = [{k: str(v) for k, v in row.items()} for row in summarize(result_rows)]
    with (directory / "all_molecules_cost_to_accuracy_summary.csv").open(newline="") as stream:
        if list(csv.DictReader(stream)) != expected_summary:
            raise ValueError("Trainability aggregate accuracy summary differs from per-molecule tables")


def _replay_batch(directory, batch):
    molecules = batch["molecules_selected"]
    requested = batch["molecules_requested"]
    if not molecules or len(set(molecules)) != len(molecules):
        raise ValueError("Missing or duplicate selected molecule in batch")
    if requested is not None and list(dict.fromkeys(requested)) != molecules:
        raise ValueError("Trainability batch selected a different roster than requested")
    if batch["molecules_failed"] or sorted(batch["molecules_succeeded"]) != sorted(molecules):
        raise ValueError("Trainability batch contains a failed or missing molecule")
    resumed = batch["molecules_resumed"]
    if len(set(resumed)) != len(resumed) or set(resumed) - set(molecules):
        raise ValueError("Trainability batch lists duplicate or unselected resumed molecules")
    molecule_fields = {"molecule", "geometry_sha256", "n_qubits_active", "n_qubits_final", "hf_energy", "casci_energy"}
    reports = {}
    for name in molecules:
        cfg = json.loads((directory / name / "run_config.json").read_text())
        if cfg["molecule"] != name:
            raise ValueError(f"Trainability batch directory identifies a different molecule than {name}")
        for field in cfg.keys() - molecule_fields:
            if field not in batch or batch[field] != cfg[field]:
                raise ValueError(f"Trainability batch {field} differs from {name} configuration")
        reports[name] = replay(directory / name)
    _check_batch_tables(directory, molecules, batch)
    return reports


def replay(directory):
    """Validate a native single-molecule result or every requested molecule in a batch."""
    directory = Path(directory)
    manifest = directory / "batch_manifest.json"
    if manifest.exists():
        return _replay_batch(directory, json.loads(manifest.read_text()))

    cfg = json.loads((directory / "run_config.json").read_text())
    name = cfg["molecule"]
    snapshot = directory / "inputs" / f"{name}.npz"
    digest = sha256(snapshot.read_bytes()).hexdigest()
    problem = load_problem(snapshot)
    if problem.molecule != name:
        raise ValueError("Run and input identify different molecules")
    expected = {(algo, k) for algo in cfg["algorithms"] for k in cfg["param_targets"]}
    with (directory / "results.csv").open() as table:
        rows = list(csv.DictReader(table))
    csv_by_key = {(r["algorithm"], int(r["target_n_params"])): r for r in rows}
    if len(rows) != len(expected) or set(csv_by_key) != expected:
        raise ValueError("CSV is missing, duplicating or adding trainability fits")
    reference = json.loads((directory / "model_reference.json").read_text())
    for key, value in model_reference(problem).items():
        _close(value, reference[key], key)
    seen = set()
    witnesses = 0
    charged = 0
    with (directory / "endpoints.jsonl").open() as stream:
        for line in stream:
            record = json.loads(line)
            key = (record["algorithm"], record["target_n_params"])
            if key not in expected or key in seen or record["input_sha256"] != digest or record["molecule"] != name:
                raise ValueError("Missing, duplicate, or incorrectly bound trainability endpoint")
            witnesses += _replay_case(problem, record, cfg, csv_by_key[key])
            probability = record["wrong_electron_probability"]
            charged += int(probability is not None and probability > 1e-8)
            seen.add(key)
    if seen != expected:
        raise ValueError("Incomplete trainability endpoint grid")
    return dict(endpoints=len(seen), crossing_witnesses=witnesses,
                endpoints_with_wrong_electron_probability_above_1e_8=charged, **reference)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Independently replay a saved trainability run.")
    parser.add_argument("directory", type=Path)
    print(json.dumps(replay(parser.parse_args().directory), indent=2, allow_nan=False))
