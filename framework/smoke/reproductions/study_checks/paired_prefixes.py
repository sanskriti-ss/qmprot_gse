# ABOUTME: Verifies native prefix trials, endpoint energies and physical charge probabilities on frozen molecules.
# ABOUTME: Compares paired scientific records across reordered workloads and summarizes the two optimizer requests.
import argparse
import csv
from hashlib import sha256
from itertools import product
import json
import math
from pathlib import Path
from statistics import mean


def checked_run(directory):
    config = json.loads((directory / "run_config.json").read_text())
    runtime = json.loads((directory / "runtime.json").read_text())
    progress = json.loads((directory / "progress.json").read_text())
    with (directory / "batch_all_results.csv").open() as stream:
        table = list(csv.DictReader(stream))
    cases = [json.loads(line) for line in (directory / "cases.jsonl").read_text().splitlines()]
    expected = list(product(config["molecules"], config["term_counts"], config["algorithms"], config["trial_seeds"]))
    assert len(cases) == len(table) == len(expected) == progress["successful_rows"]
    assert progress["failed_cases"] == progress["skipped_cases"] == 0
    digests = {m: sha256((directory / "inputs" / f"{m}.npz").read_bytes()).hexdigest() for m in config["molecules"]}
    errors, indexed = [], {}
    for key, case, row in zip(expected, cases, table, strict=True):
        molecule, k, method, seed = key
        assert key == (case["molecule"], case["prefix_terms"], case["algorithm"], case["random_seed"])
        assert key == (row["molecule"], int(row["n_parameters"]), row["algorithm"], int(row["random_seed"]))
        assert digests[molecule] == case["input_sha256"] and row["status"] == "ok"
        assert int(row["algorithm_n_parameters"]) == len(case["parameters"])
        assert case["optimizer_requested"] == config["optimizer"]
        policy = {"iqcc_inspired_vqe": "COBYLA", "iqcc_true_vqe": "analytic Pauli angle selection"}.get(
            method, config["optimizer"])
        assert case["parameter_update_policy"] == row["parameter_update_policy"] == policy
        sector = case["endpoint_sector"]
        assert sector["sector_status"] == "measured"
        expected_n = str(sector["expected_electron_count"])
        probabilities = sector["electron_count_probabilities"]
        assert all(0 <= p <= 1 + 1e-12 for p in probabilities.values())
        assert abs(sum(probabilities.values()) - 1) < 1e-12
        outside = sum(p for n, p in probabilities.items() if n != expected_n)
        assert abs(outside - sector["wrong_electron_probability"]) < 1e-12
        assert abs(outside - float(row["wrong_electron_probability"])) < 1e-10
        energy = case["energy"]
        assert energy >= case["prefix_reference"] - 1e-8 and abs(energy - float(row["energy"])) < 1e-10
        errors.append(abs(energy - sector["rescored_energy"]))
        indexed[key] = case
    assert max(errors) < 1e-8
    with (directory / "summary_by_algorithm_term_count.csv").open() as stream:
        summaries = list(csv.DictReader(stream))
    assert len(summaries) == len(config["term_counts"]) * len(config["algorithms"])
    for row in summaries:
        means = []
        for molecule in config["molecules"]:
            trials = [indexed[molecule, int(row["n_parameters"]), row["algorithm"], s] for s in config["trial_seeds"]]
            means.append(mean(abs(t["energy"] - t["full_casci"]) for t in trials))
        assert math.isclose(mean(means), float(row["mean_abs_error_vs_full_reference"]), abs_tol=1e-12)
        assert int(row["n_molecules"]) == len(means) and int(row["n_samples"]) == len(means) * len(config["trial_seeds"])
    summary = dict(source_sha256=runtime["source_sha256"], input_sha256=digests, term_counts=config["term_counts"],
                   n_layers=config["n_layers"], max_operators=config["max_operators"], max_iterations=config["max_iterations"],
                   optimizer_requested=config["optimizer"], seeds=config["trial_seeds"], passed=len(cases),
                   max_endpoint_energy_difference_ha=max(errors))
    return indexed, summary


def compare(directory):
    data, reports = {}, {}
    for name in ("prefix-cobyla", "prefix-bayesian", "gly-iqcc-cobyla", "gly-iqcc-bayesian", "gly-adapt-cobyla",
                 "gly-adapt-bayesian", "gly-roster-A", "gly-roster-B", "molecule-roster-A", "molecule-roster-B"):
        data[name], reports[name] = checked_run(directory / f"verified-{name}")
    replays = {}
    for name in ("gly-roster", "molecule-roster"):
        a, b = data[name + "-A"], data[name + "-B"]
        assert all(case == a[key] for key, case in b.items())
        replays[name] = len(b)
    policies, charge = {}, {}
    for optimizer in ("cobyla", "bayesian"):
        cases = data["prefix-" + optimizer]
        methods = sorted({key[2] for key in cases})
        assert len(methods) == 11 and len(cases) == 22
        for method in methods:
            trials = [cases["hydrogen", 15, method, seed] for seed in (42, 43)]
            policy = trials[0]["parameter_update_policy"]
            policies.setdefault(method, {})[optimizer.upper()] = policy
            charge.setdefault(method, {})[optimizer.upper()] = [
                t["endpoint_sector"]["wrong_electron_probability"] for t in trials]
        for key, case in data["gly-iqcc-" + optimizer].items():
            assert len(case["parameters"]) >= 2
        for key, case in data["gly-adapt-" + optimizer].items():
            assert len(case["parameters"]) == 1
    return dict(scope="Local statevector; H2 all 15 terms, contextual glycine and two-molecule subset controls. "
                      "Seeds 42 and 43 are paired trials. Stationary solvers may select zero operators.",
                runs=reports, identical_replayed_case_records=replays, h2_parameter_update_policy=policies,
                h2_wrong_electron_probability_by_seed_42_43=charge)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("directory", type=Path, help="Parent of the ten verified native experiment output directories")
    print(json.dumps(compare(parser.parse_args().directory), indent=2, allow_nan=False))
