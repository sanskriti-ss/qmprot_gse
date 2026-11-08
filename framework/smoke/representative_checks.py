# ABOUTME: Verifies the scientific artifacts of the configured trainability and hydrogen noise experiments.
# ABOUTME: Cross-checks individual cases, evaluation budgets, trajectories, optimum parameters and batch outputs.
import hashlib
import json
import math
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np
from PIL import Image

from algorithms import get_algorithm
from experiments import cost_to_accuracy
from experiments.noise_resilience import noise_strength_label
from experiments._common import (
    ACCURACY_LADDER_HA, CHEMICAL_ACCURACY_HA, HF_TOLERANCE_HA, parameter_drift_metrics, sample_initial_parameters, threshold_label,
)
from smoke.batch import FRAMEWORK, read_rows, require


# The experiment writes energies and parameter metrics to ten decimal places.
CSV_TOLERANCE = 1.1e-10

# Direct PySCF STO-3G references use full CI for H₂ and CAS(6,6) in glycine orbitals 17–22 (zero-based).
TINY_CASCI_ENERGIES = {
    "ala": -1.1297025558704,
    "arg": -1.1222508120508,
    "asp": -1.1132737792729,
    "cys": -1.1032476612245,
    "gln": -1.0925391540402,
    "glu": -1.0814385650691,
    "gly": -279.1376629146243,
    "his": -1.0589646308990,
    "hydrogen": -1.1371170673457,
    "ile": -1.0479495737708,
    "leu": -1.0372714325382,
    "lys": -1.0270393415632,
    "met": -1.0173386758917,
    "phe": -1.0082326685949,
    "pro": -0.9997641261213,
    "ser": -0.9919573337202,
    "thr": -0.9848201474809,
    "trp": -0.9783462284041,
    "tyr": -0.9725173580938,
    "val": -0.9673057691733,
}


def validate_tiny_reference(molecule: str, energy: float):
    require(molecule in TINY_CASCI_ENERGIES, f"Unknown tiny fixture {molecule}")
    expected = TINY_CASCI_ENERGIES[molecule]
    require(math.isclose(energy, expected, rel_tol=0, abs_tol=1e-9),
            f"Unexpected tiny-fixture CASCI energy for {molecule}: {energy} Ha (expected {expected} Ha)")


def _campaign(filename, args=None):
    if args is None:
        args = json.loads((FRAMEWORK / "campaigns/20261105" / filename).read_text())["args"]
    keys = {"seed": "random_seed", "max-iter": "max_iterations", "eval-budget": "he_eval_budget"}
    fields = {keys.get(k, k.replace("-", "_")): v for k, v in args.items()}
    fields["cs_target_qubits"] = fields["cs_target_qubits"] or None
    return fields


def _fields(data, expected, location):
    for key, value in expected.items():
        require(key in data and data[key] == value, f"{location}: unexpected {key}")


def _plots(folder, names):
    for name in names:
        with Image.open(folder / name) as image:
            require(image.format == "PNG", f"Invalid plot: {folder / name}")
            image.verify()


def _close(a, b):
    return math.isclose(float(a), float(b), rel_tol=0, abs_tol=CSV_TOLERANCE)


def _groups(rows, key):
    grouped = defaultdict(list)
    for row in rows:
        grouped[key(row)].append(row)
    return grouped


def _train_key(row):
    return row["algorithm"], int(row["target_n_params"])


def _iterations(trace, count, label):
    require([int(t["iteration"]) for t in trace] == list(range(count)) and
            all(math.isfinite(float(t["energy"])) for t in trace), f"Invalid iterations for {label}")


def _train_case(row, convergence, evaluations, cfg, endpoint):
    algo, target = _train_key(row)
    case = f"{cfg['molecule']}/{algo}/{target}"
    actual = int(row["actual_n_params"])
    count = int(row["n_cost_evals"])
    callbacks = int(row["n_callback_evals"])
    budget = cfg["he_eval_budget"]
    require(row["molecule"] == cfg["molecule"] and 0 <= actual <= target and
            1 < count <= budget and int(row["eval_budget"]) == budget, f"Invalid case or budget for {case}")
    require(row["param_match"] == ("exact" if actual == target else "stopped_early") and
            row["stopped_at_accuracy"] == row["bounds_only"] == "0", f"Invalid stop or size for {case}")
    require(math.isfinite(float(row["runtime_seconds"])) and float(row["runtime_seconds"]) > 0,
            f"Invalid runtime for {case}")
    ref = cfg["casci_energy"]
    energy = float(row["final_energy"])
    require(math.isfinite(energy) and _close(energy - ref, row["error_vs_casci"]) and
            _close(cfg["hf_energy"] - ref, row["hf_error_vs_casci"]), f"Inconsistent energy for {case}")
    hf_error = cfg["hf_energy"] - ref
    require(row["hf_trivial"] == str(int(abs(hf_error) <= CHEMICAL_ACCURACY_HA)) and
            row["attainment_recorded"] == "1" and bool(row["stop_reason"]) and
            0 <= float(row["wrong_electron_probability"]) <= 1 + CSV_TOLERANCE, f"Invalid endpoint for {case}")
    require(_close(energy, endpoint["reported_energy"]) and _close(energy, endpoint["rescored_energy"]) and
            _close(row["wrong_electron_probability"], endpoint["wrong_electron_probability"]) and
            len(endpoint["parameters"]) == actual, f"Invalid saved endpoint for {case}")
    if algo == "hardware_efficient_vqe":
        require(row["init_label"] == cfg["he_init_strategy"] + "+HF" and row["converged_below_threshold"] == ""
                and row["optimizer_success"] in ("", "0", "1"),
                f"Invalid HE initialization or stopping flag for {case}")
        require(actual == target and 0 < callbacks == int(row["n_outer_iters"]) <= count, f"Invalid callbacks for {case}")
        _iterations(convergence, callbacks, case)
    else:
        require(row["init_label"] == "HF reference" and actual <= int(row["n_outer_iters"]) <= target and
                row["converged_below_threshold"] in ("0", "1"),
                f"Invalid ADAPT initialization or iteration count for {case}")
        require(callbacks == 0, f"Unexpected callbacks for {case}")
        _iterations(convergence, actual, case)
        require(_close(convergence[-1]["energy"] if convergence else endpoint["accepted_states"][0]["energy"], energy),
                f"Inconsistent final ADAPT energy for {case}")

    indices = [int(t["eval_index"]) for t in evaluations]
    signed = [float(t["best_error_vs_casci"]) for t in evaluations]
    absolute = [float(t["best_abs_error_vs_casci"]) for t in evaluations]
    require(bool(indices) and indices == sorted(set(indices)) and indices[0] == 1 and indices[-1] == count and
            all(math.isfinite(s) and 0 <= a <= abs(s) + CSV_TOLERANCE for s, a in zip(signed, absolute)) and
            signed == sorted(signed, reverse=True) and absolute == sorted(absolute, reverse=True) and
            _close(abs(signed[0]), absolute[0]) and _close(signed[-1], row["best_error_vs_casci"]) and
            _close(absolute[-1], row["best_abs_error_vs_casci"]), f"Invalid evaluation trace for {case}")
    require(float(row["hf_tolerance_ha"]) == cfg["hf_tolerance_ha"] == HF_TOLERANCE_HA,
            f"Invalid HF tolerance for {case}")
    milestones = [("hf", hf_error + HF_TOLERANCE_HA, signed)]
    milestones.extend((threshold_label(t), t, absolute) for t in cfg["accuracy_thresholds_ha"])
    for label, threshold, errors in milestones:
        hit = row[f"evals_to_{label}"]
        require(row[f"reached_{label}"] == str(int(bool(hit))), f"Inconsistent threshold flag for {case}/{label}")
        if hit:
            ix = int(hit)
            require(ix in indices and errors[indices.index(ix)] <= threshold + CSV_TOLERANCE and
                    all(e >= threshold - CSV_TOLERANCE for i, e in zip(indices, errors) if i < ix),
                    f"Invalid threshold crossing for {case}/{label}")
        else:
            require(errors[-1] >= threshold - CSV_TOLERANCE, f"Missing threshold crossing for {case}/{label}")
        for kind in ("accepted", "probe"):
            idx = endpoint[f"{kind}_evals_to"][label]
            require(row[f"{kind}_evals_to_{label}"] == ("" if idx is None else str(idx)),
                    f"Inconsistent {kind} crossing for {case}/{label}")
            if idx is not None:
                require(0 <= idx <= count, f"Out-of-budget {kind} crossing for {case}/{label}")
                if kind == "probe":
                    require(algo != "hardware_efficient_vqe" and hit and 1 <= int(hit) <= idx,
                            f"Uncounted probe crossing for {case}/{label}")
                else:
                    require(any(s["eval_index"] == idx and label in s.get("first_crossings", ()) and
                                _close(s["energy"], s["rescored_energy"]) for s in endpoint["accepted_states"]),
                            f"Missing accepted state for {case}/{label}")


def validate_trainability(directory: Path, molecules: tuple[str, ...], *, batch: bool, targets=(10, 50, 100),
                          args=None, tiny_inputs=False):
    directory = Path(directory)
    require(bool(molecules) and len(set(molecules)) == len(molecules) and (batch or len(molecules) == 1),
            "Invalid trainability molecule selection")
    settings = _campaign("01_trainability.json", args)
    settings.update(param_targets=list(targets), adapt_gradient_threshold=1e-8,
                    accuracy_thresholds_ha=list(ACCURACY_LADDER_HA))
    settings.setdefault("qubit_adapt_max_gradient_candidates", None)
    settings.pop("molecule", None)
    settings.pop("molecules", None)
    stop_at_accuracy = settings.pop("he_stop_at_accuracy")
    expected = {(algo, n) for algo in settings["algorithms"] for n in targets}
    totals = defaultdict(list)
    for molecule in molecules:
        folder = directory / molecule if batch else directory
        cfg = json.loads((folder / "run_config.json").read_text())
        n_qubits = 4 if molecule == "hydrogen" or (tiny_inputs and molecule != "gly") else 12
        _fields(cfg, dict(settings, molecule=molecule, n_qubits_active=n_qubits, n_qubits_final=n_qubits), folder)
        require(all(math.isfinite(cfg[k]) for k in ("casci_energy", "hf_energy")), f"Invalid references for {molecule}")
        if tiny_inputs:
            validate_tiny_reference(molecule, cfg["casci_energy"])
        rows = read_rows(folder / "results.csv")
        require(Counter(map(_train_key, rows)) == Counter({k: 1 for k in expected}), f"Incomplete grid for {molecule}")
        require(any(r["algorithm"] == "qubit_adapt_vqe" and int(r["actual_n_params"]) > 0 for r in rows),
                f"ADAPT did not optimize an operator for {molecule}")
        raw_convergence = read_rows(folder / "convergence_traces.csv")
        raw_evals = read_rows(folder / "eval_traces.csv")
        convergence = _groups(raw_convergence, _train_key)
        evaluations = _groups(raw_evals, _train_key)
        require(set(convergence) <= expected and set(evaluations) == expected and
                all(r["molecule"] == molecule for r in raw_evals), f"Invalid trace grid for {molecule}")
        endpoint_records = [json.loads(line) for line in (folder / "endpoints.jsonl").read_text().splitlines()]
        endpoints = {_train_key(record): record for record in endpoint_records}
        require(set(endpoints) == expected and len(endpoints) == len(endpoint_records),
                f"Incomplete endpoint grid for {molecule}")
        for row in rows:
            key = _train_key(row)
            _train_case(row, convergence[key], evaluations[key], cfg, endpoints[key])
        totals["results"].extend(rows)
        totals["eval_traces"].extend(raw_evals)
        totals["convergence_traces"].extend(dict(molecule=molecule, **r) for r in raw_convergence)
        require((folder / "cost_to_accuracy_report.md").stat().st_size > 0, f"Empty accuracy report for {molecule}")
        _plots(folder, ("cost_evals_vs_params.png", "cost_to_accuracy.png", "error_vs_cost.png",
                        "final_energy_vs_params.png", "runtime_vs_params.png"))
    if batch:
        manifest = json.loads((directory / "batch_manifest.json").read_text())
        _fields(manifest, dict(settings, he_stop_at_accuracy=stop_at_accuracy, molecules_requested=list(molecules),
                               molecules_selected=list(molecules), molecules_succeeded=list(molecules),
                               molecules_failed={}, molecules_resumed=[], n_rows=len(molecules)*len(expected)), directory)
        for table, records in totals.items():
            require(records == read_rows(directory / f"all_molecules_{table}.csv"), f"Invalid aggregate {table}")
        summary = [{k: str(v) for k, v in row.items()} for row in cost_to_accuracy.summarize(totals["results"])]
        require(read_rows(directory / "all_molecules_cost_to_accuracy_summary.csv") == summary, "Invalid aggregate accuracy summary")
        require((directory / "cost_to_accuracy_report.md").stat().st_size > 0, "Empty aggregate accuracy report")
        _plots(directory, ("all_molecules_cost_to_accuracy.png", "all_molecules_cost_evals_vs_params.png"))


def validate_short_batch_objectives(directory: Path, molecules: tuple[str, ...]):
    directory = Path(directory)
    references = {key: json.loads((directory / key / "run_config.json").read_text())["hf_energy"] for key in molecules}
    # A binary64 positive zero makes the single Pauli rotation the identity on the HF state.
    zero_angle = hashlib.sha256(bytes(8)).hexdigest()
    seen = set()
    with (directory / "events.jsonl").open() as stream:
        for line in stream:
            event = json.loads(line)
            if (event["algorithm"] == "qubit_adapt_vqe" and event["target"] == 1 and not event["callback"]
                    and len(event["operators"]) == 1 and event["parameters"] == zero_angle):
                key = event["molecule"]
                energy = float.fromhex(event["energy"])
                require(key in references, f"Unexpected zero-angle ADAPT molecule {key}")
                require(math.isclose(energy, references[key], rel_tol=0, abs_tol=1e-8),
                        f"Zero-angle ADAPT energy differs from HF for {key}: {energy} vs {references[key]} Ha")
                seen.add(key)
    missing = references.keys() - seen
    require(not missing, f"Missing zero-angle ADAPT objective for {sorted(missing)}")


def validate_short_hydrogen(directory: Path):
    directory = Path(directory)
    rows = {_train_key(row): row for row in read_rows(directory / "results.csv")}
    with (directory / "events.jsonl").open() as stream:
        cases = _groups((json.loads(line) for line in stream), lambda e: (e["algorithm"], e["target"]))

    small = ("qubit_adapt_vqe", 2)
    probes = cases[small]
    row = rows[small]
    second_screen = {tuple(e["operators"]) for e in probes if len(e["operators"]) == 2}
    require(row["actual_n_params"] == "1" and row["converged_below_threshold"] == "0" and
            len(probes) == int(row["n_cost_evals"]) == int(row["eval_budget"]) and
            len(second_screen) > 1 and len({ops[0] for ops in second_screen}) == 1 and
            tuple(probes[-1]["operators"]) in second_screen,
            "Hydrogen ADAPT did not exhaust the budget during the second screen")

    large = ("qubit_adapt_vqe", 21)
    row = rows[large]
    probes = cases[large]
    # The last evaluation records the HF energy after the decision to skip the inner solve.
    after_screen = int(row["eval_budget"]) - (len(probes) - 1)
    require(row["actual_n_params"] == "0" and row["converged_below_threshold"] == "0" and
            len(probes) == int(row["n_cost_evals"]) and 1 < len(probes) and 0 < after_screen < large[1] and
            all(len(e["operators"]) == 1 for e in probes[:-1]) and probes[-1]["operators"] == [],
            "Hydrogen ADAPT did not reach the inner budget check before adding an operator")

    for target in (2, 21):
        evaluated = set()
        for event in cases["hardware_efficient_vqe", target]:
            if event["callback"]:
                if event["parameters"] in evaluated:
                    return
            else:
                evaluated.add(event["parameters"])
    raise ValueError("Hydrogen trainability has no repeated HE callback input")


def validate_noise(directory: Path, *, args=None, tiny_inputs=False):
    directory = Path(directory)
    cfg = json.loads((directory / "run_config.json").read_text())
    settings = _campaign("02_noise_resilience.json", args)
    _fields(cfg, dict(settings, n_qubits_active=4, n_qubits_final=4, n_parameters=4*(1+settings["n_layers"])), directory)
    require(all(math.isfinite(cfg[k]) for k in ("casci_energy", "hf_energy")), "Invalid noise reference energy")
    if tiny_inputs:
        validate_tiny_reference(cfg["molecule"], cfg["casci_energy"])
    grid = {(m, p) for m in cfg["noise_models"] for p in cfg["noise_strengths"]} | {("none", 0.0)}
    def key(row):
        return row["noise_model"], float(row["noise_strength"])
    results = read_rows(directory / "results.csv")
    require(Counter(map(key, results)) == Counter({k: 1 for k in grid}), "Invalid noise grid")
    rows = {key(r): r for r in results}
    traces = _groups(read_rows(directory / "convergence_traces.csv"), key)
    require(set(traces) == grid, "Invalid noise trace grid")
    params = json.loads((directory / "optimum_parameters.json").read_text())
    noisy = params["theta_noisy_star"]
    require(set(noisy) == {f"{m}@{noise_strength_label(p)}" for m, p in grid if p}, "Invalid noisy optimum parameter grid")
    count = cfg["n_parameters"]
    def vector(values):
        v = np.asarray(values, dtype=float)
        require(v.shape == (count,) and np.isfinite(v).all(), "Invalid optimum parameter vector")
        return v
    initial = vector(params["theta_init"])
    clean = vector(params["theta_clean_star"])
    expected_initial = sample_initial_parameters(count, cfg["init_strategy"], rng=np.random.default_rng(cfg["random_seed"]))
    require(np.array_equal(initial, expected_initial), "Incorrect seeded noise initialization")
    baseline = rows["none", 0.0]
    for (model, p), r in rows.items():
        label = f"{model}@{noise_strength_label(p)}"
        trace = traces[model, p]
        iterations = int(r["n_iterations"])
        require(1 < iterations <= cfg["max_iterations"], f"Invalid noise iterations for {label}")
        _iterations(trace, iterations, label)
        require(all(math.isfinite(float(r[k])) for k in ("energy_noisy", "energy_clean_at_noisy_params", "energy_clean_baseline"))
                and _close(r["energy_clean_baseline"], baseline["energy_noisy"]) and
                float(r["energy_noisy"]) <= min(float(t["energy"]) for t in trace) + CSV_TOLERANCE,
                f"Inconsistent noise energy for {label}")
        theta = vector(noisy[label]) if p else clean
        if not p:
            require(all(r[k] == baseline[k] for k in baseline if k != "noise_model") and
                    _close(r["energy_clean_at_noisy_params"], r["energy_noisy"]) and
                    [t["energy"] for t in trace] == [t["energy"] for t in traces["none", 0.0]],
                    f"Invalid clean baseline for {label}")
        metrics = parameter_drift_metrics(theta, clean, initial, period=get_algorithm(cfg["algorithm"]).parameter_period)
        denom = np.linalg.norm(theta) * np.linalg.norm(clean)
        metrics["param_drift_cosine"] = np.dot(theta, clean) / denom if denom else 1.0
        for metric, value in metrics.items():
            matches = r[metric] == "" if value is None else _close(r[metric], value)
            require(matches, f"Inconsistent {metric} for {label}")
    for model in cfg["noise_models"]:
        require(any(not _close(rows[model, p]["energy_noisy"], rows[model, p]["energy_clean_at_noisy_params"])
                    for p in cfg["noise_strengths"] if p), f"No measured noise effect for {model}")
    _plots(directory, ("energy_vs_noise.png", "param_drift_vs_noise.png", "convergence.png"))
