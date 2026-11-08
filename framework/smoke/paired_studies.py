# ABOUTME: Runs paired H₂ optimizer controls through the native prefix, adaptive accuracy and trainability batch CLIs.
# ABOUTME: Reconstructs saved endpoints from their parameters and measures charge on the actual molecular Hamiltonian.
import argparse
from hashlib import sha256
import json
import os
from pathlib import Path
import subprocess
import sys

import numpy as np
from openfermion import QubitOperator

from algorithms import ALGORITHMS
from experiments.endpoint_sector import summarize_endpoint
from experiments.provenance import load_problem
from experiments.replay_trainability import replay
from smoke.batch import FRAMEWORK, read_rows, require
from smoke.reproductions.study_checks.paired_prefixes import checked_run
from smoke.solver_studies import FROZEN, FROZEN_SHA, check_input


OPTIMIZERS = ("COBYLA", "BAYESIAN")
ADAPT = ("adapt_vqe", "qubit_adapt_vqe")
TRAINABILITY = ("hardware_efficient_vqe", *ADAPT)


def run(directory):
    directory = Path(directory).resolve()
    directory.mkdir(parents=True, exist_ok=False)
    env = {**os.environ, "QMPROT_GEOMETRIES": str(FRAMEWORK / "smoke/geometries")}
    with (directory / "campaign.log").open("w") as log:
        def launch(module, args, output, flag="--output-dir"):
            subprocess.run([sys.executable, "-m", "experiments." + module, "--molecules", "hydrogen",
                            *args, "--no-plots", flag, str(output)], cwd=FRAMEWORK, env=env,
                           stdout=log, stderr=subprocess.STDOUT, check=True)

        for optimizer in OPTIMIZERS:
            opt = ["--optimizer", optimizer]
            output = directory / optimizer
            launch("experiment5_term_count_sweep",
                   [*opt, "--prepared-inputs", str(FROZEN.parent), "--cs-target-qubits", "0", "--term-counts", "15",
                    "--seeds", "42", "43", "--n-layers", "1", "--max-operators", "2", "--max-iter", "24"],
                   output / "prefix")
            for method in ADAPT:
                launch("accuracy_vs_params", [*opt, "--algorithm", method, "--k-list", "1", "2",
                                              "--inner-max-iter", "12"], output / "accuracy" / method)
            launch("trainability", [*opt, "--algorithms", *TRAINABILITY, "--param-targets", "2", "--eval-budget",
                                    "64", "--max-iter", "12"], output / "trainability", "--output-root")


def check_prefix(directory, optimizer):
    cases, receipt = checked_run(directory)
    require(receipt["optimizer_requested"] == optimizer and receipt["seeds"] == [42, 43]
            and set(cases) == {("hydrogen", 15, alg, s) for alg in ALGORITHMS for s in (42, 43)}
            and receipt["input_sha256"] == {"hydrogen": FROZEN_SHA}, "Incorrect hydrogen prefix matrix")
    problem = load_problem(directory / "inputs/hydrogen.npz")
    leaked = 0
    for (_, _, method, seed), case in cases.items():
        solver = ALGORITHMS[method](problem.hamiltonian, n_layers=1, max_operators=2, random_seed=seed)
        solver.build_ansatz()
        meta = case["metadata"]
        if method == "adapt_vqe":
            solver.selected_operators = meta["selected_operators"]
        elif method == "qubit_adapt_vqe":
            solver.selected_operators = [solver.operator_pool.index(word) for word in meta["selected_operators_str"]]
        elif method == "iqcc_inspired_vqe":
            solver.selected_operators = [QubitOperator(word) for word in meta["selected_generators"]]
        elif method == "iqcc_true_vqe":
            solver.selected_operators = meta["selected_generators"]
        state = summarize_endpoint(problem, solver, case["parameters"])
        saved = case["endpoint_sector"]
        measured, reported = state["electron_count_probabilities"], saved["electron_count_probabilities"]
        require(measured.keys() == reported.keys(), "Incorrect H₂ electron-count labels")
        np.testing.assert_allclose([state["rescored_energy"], *measured.values()],
                                   [case["energy"], *(reported[n] for n in measured)], rtol=0., atol=1e-9)
        leaked += state["wrong_electron_probability"] > .01
    return len(cases), int(leaked)


def check_accuracy(directory, method, optimizer):
    require(json.loads((directory / "batch_failures.json").read_text()) == [], "Accuracy batch failed")
    folder = directory / ("hydrogen_" + method)
    check_input(folder)
    problem = load_problem(folder / "inputs/hydrogen.npz")
    config = json.loads((folder / "run_config.json").read_text())
    require(config["algorithm"] == method and config["optimizer"] == optimizer and config["k_list"] == [1, 2],
            "Incorrect accuracy configuration")
    solver = ALGORITHMS[method](problem.hamiltonian, random_seed=42)
    solver.build_ansatz()
    pool = [list(op) if isinstance(op, tuple) else op for op in solver.operator_pool]
    selected = []
    states = [summarize_endpoint(problem, solver, [], [])]
    trajectory = read_rows(folder / "full_convergence.csv")
    require([int(r["k"]) for r in trajectory] == list(range(1, config["actual_max_k"] + 1)), "Missing accuracy steps")
    for row in trajectory:
        selected.append(pool.index(json.loads(row["added_generator"])))
        angles = json.loads(row["parameters"])
        require(len(angles) == len(selected), "Incorrect ADAPT angles")
        state = summarize_endpoint(problem, solver, angles, selected)
        np.testing.assert_allclose([state["rescored_energy"], state["wrong_electron_probability"]],
                                   [float(row["energy"]), float(row["wrong_electron_probability"])], atol=1e-8, rtol=0.)
        states.append(state)
    native = read_rows(folder / "results.csv")
    batch = read_rows(directory / "batch_all_results.csv")
    require([int(r["k"]) for r in native] == [1, 2] and len(batch) == len(native), "Missing accuracy checkpoints")
    for row, merged in zip(native, batch, strict=True):
        k = int(row["k"])
        attained = min(k, len(selected))
        state = states[attained]
        require(all(merged[key] == value for key, value in row.items()) and int(row["attained_k"]) == attained
                and row["status"] == ("ok" if attained == k else "stopped_early"), "Incorrect accuracy milestone")
        e = state["rescored_energy"]
        np.testing.assert_allclose([float(row[key]) for key in ("energy", "rescored_energy", "error_vs_hf",
                                                                "error_vs_casci", "wrong_electron_probability")],
                                   [e, e, e - problem.hf_energy, e - problem.casci_energy,
                                    state["wrong_electron_probability"]], atol=1e-8, rtol=0.)
    return len(native)


def validate(directory):
    directory = Path(directory)
    log = (directory / "campaign.log").read_text()
    require(log and not any(token in log for token in ("[ERROR]", "[WARNING]", "Warning:", "Traceback (most recent")),
            "Paired study logged a failure or warning")
    require(sha256(FROZEN.read_bytes()).hexdigest() == FROZEN_SHA, "Incorrect frozen hydrogen snapshot")
    counts = dict(prefix=0, prefix_wrong_electron_above_1pct=0, accuracy=0, trainability=0)
    fingerprints = set()
    for optimizer in OPTIMIZERS:
        output = directory / optimizer
        n, leaked = check_prefix(output / "prefix", optimizer)
        counts["prefix"] += n
        counts["prefix_wrong_electron_above_1pct"] += leaked
        for method in ADAPT:
            counts["accuracy"] += check_accuracy(output / "accuracy" / method, method, optimizer)
        train = output / "trainability"
        check_input(train / "hydrogen")
        cfg = json.loads((train / "hydrogen/run_config.json").read_text())
        require(cfg["optimizer"] == optimizer and tuple(cfg["algorithms"]) == TRAINABILITY and cfg["param_targets"] == [2],
                "Incorrect trainability configuration")
        counts["trainability"] += replay(train)["hydrogen"]["endpoints"]
        snapshots = list(output.rglob("runtime.json"))
        require(len(snapshots) == 4, "Missing native study snapshots")
        for path in snapshots:
            sha = sha256((path.parent / "framework-source.zip").read_bytes()).hexdigest()
            require(json.loads(path.read_text())["source_sha256"] == sha, "Paired source archive differs from runtime")
            fingerprints.add(sha)
    require(len(fingerprints) == 1, "Paired scientific source changed within the campaign")
    return {**counts, "source_sha256": fingerprints.pop(), "frozen_sha256": FROZEN_SHA}


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("directory", type=Path)
    parser.add_argument("--validate", action="store_true")
    args = parser.parse_args()
    if not args.validate:
        run(args.directory)
    print(json.dumps(validate(args.directory), indent=2))
