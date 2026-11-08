# ABOUTME: Records a full native Bayesian initialization trajectory on an archived glycine Hamiltonian.
# ABOUTME: Verifies surrogate grids, predicted distributions, selected angles and physical endpoints across sources.
import argparse
from dataclasses import asdict
from hashlib import sha256
from importlib.metadata import version
from importlib.util import find_spec
import json
import os
from pathlib import Path
import platform
from time import perf_counter

import numpy as np
from skopt.learning.forest import ExtraTreesRegressor
from experiments import barren_plateau
from experiments.provenance import framework_source, load_problem


def digest(array):
    return sha256(np.ascontiguousarray(array).tobytes()).hexdigest()


def run(path, directory):
    directory.mkdir(parents=True, exist_ok=False)
    source = framework_source()
    (directory / "framework-source.zip").write_bytes(source)
    problem = load_problem(path)
    if problem.molecule != "gly" or problem.hamiltonian.n_qubits != 4:
        raise ValueError("Expected a four-qubit glycine Hamiltonian")
    source_input = sha256(path.read_bytes()).hexdigest()
    if source_input != "cbc61637c2f05242c3385dd3754c6564100c2fa347af820c355b952f0cfc7f15":
        raise ValueError("Unexpected ordered glycine input")
    forest = ExtraTreesRegressor
    if find_spec("core.forest") is not None:
        from core.forest import EnergyForest
        forest = EnergyForest
    original_build = barren_plateau.build_vqe
    original_predict = forest.predict
    calls, predictions = [], []
    grid_hash_seconds, event_trace_seconds = 0., 0.

    def build(*args, **kwargs):
        solver = original_build(*args, **kwargs)
        evaluate = solver.cost_function

        def record(theta):
            nonlocal event_trace_seconds
            energy = evaluate(theta)
            start_record = perf_counter()
            calls.append(dict(parameters=np.asarray(theta).tobytes().hex(), energy=float(energy).hex()))
            event_trace_seconds += perf_counter() - start_record
            return energy

        solver.cost_function = record
        return solver

    def predict(self, X, return_std=False):
        nonlocal grid_hash_seconds
        result = original_predict(self, X, return_std=return_std)
        if return_std:
            start_hash = perf_counter()
            mean, std = result
            predictions.append(dict(candidates=digest(X), mean=digest(mean), std=digest(std), count=len(X)))
            grid_hash_seconds += perf_counter() - start_hash
        return result

    barren_plateau.build_vqe = build
    forest.predict = predict
    try:
        start = perf_counter()
        row, = barren_plateau.run_optimization_comparison(
            problem, algorithm="hardware_efficient_vqe", n_layers=16, strategies=("small_random",), n_trials=1,
            optimizer="BAYESIAN", max_iterations=200, convergence_threshold=1e-10, learning_rate=.1,
            noise_model=None, noise_strength=0., random_seed=42)
        elapsed = perf_counter() - start
    finally:
        barren_plateau.build_vqe = original_build
        forest.predict = original_predict
    if row.n_parameters != 68 or len(calls) != 200 or len(predictions) != 168:
        raise ValueError("Incomplete Bayesian trial")
    for name, values in (("events", calls), ("forests", predictions)):
        (directory / (name + ".jsonl")).write_text("".join(json.dumps(v) + "\n" for v in values))
    (directory / "endpoint.json").write_text(json.dumps(asdict(row), indent=2, allow_nan=False) + "\n")
    libraries = ("numpy", "scipy", "scikit-optimize", "scikit-learn", "joblib", "pennylane", "pennylane-lightning")
    threads = ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "VECLIB_MAXIMUM_THREADS")
    runtime = dict(trial_seconds=elapsed, scientific_source_sha256=sha256(source).hexdigest(), input_sha256=source_input,
                   grid_hash_seconds=grid_hash_seconds, event_trace_seconds=event_trace_seconds, host=platform.platform(),
                   kernel=platform.release(), python=platform.python_version(),
                   libraries={name: version(name) for name in libraries}, threads={name: os.getenv(name) for name in threads})
    (directory / "runtime.json").write_text(json.dumps(runtime, indent=2) + "\n")


def compare(directory):
    info = {}
    files = ("events.jsonl", "forests.jsonl", "endpoint.json")
    hashes = {}
    for version in ("baseline", "candidate"):
        root = directory / version
        info[version] = json.loads((root / "runtime.json").read_text())
        if sha256((root / "framework-source.zip").read_bytes()).hexdigest() != info[version]["scientific_source_sha256"]:
            raise ValueError("Invalid scientific source hash")
        for f in files:
            current = sha256((root / f).read_bytes()).hexdigest()
            if current != hashes.setdefault(f, current):
                raise ValueError(f"Nonidentical {f}")
    endpoint = json.loads((directory / "candidate/endpoint.json").read_text())
    events = [json.loads(line) for line in (directory / "candidate/events.jsonl").read_text().splitlines()]
    if (len(events) != endpoint["n_iterations"] or len(events) != 200 or endpoint["n_parameters"] != 68
            or any(len(bytes.fromhex(e["parameters"])) != 68 * 8 for e in events)
            or [energy.hex() for energy in endpoint["convergence_history"]] != [e["energy"] for e in events]):
        raise ValueError("Incorrect objective stream")
    for event, angle_key, energy_key in ((events[0], "initial_parameters", "initial_energy"),
                                          (min(events, key=lambda e: float.fromhex(e["energy"])), "optimal_parameters", "final_energy")):
        if event != dict(parameters=np.asarray(endpoint[angle_key]).tobytes().hex(), energy=endpoint[energy_key].hex()):
            raise ValueError(f"Unmatched {angle_key} or {energy_key}")
    if info["candidate"]["input_sha256"] != info["baseline"]["input_sha256"]:
        raise ValueError("Mismatched glycine inputs")
    forests = [json.loads(s) for s in (directory / "candidate/forests.jsonl").read_text().splitlines()]
    if len(forests) != 168 or any(f["count"] != 10_000 for f in forests):
        raise ValueError("Incorrect forest candidate count")
    result = dict(runs=info, trial_speedup=info["baseline"]["trial_seconds"]/info["candidate"]["trial_seconds"],
                  scientific_outputs_sha256=hashes, objective_calls=endpoint["n_iterations"], forests=len(forests),
                  candidates_per_forest=10_000, initial_energy=endpoint["initial_energy"], final_energy=endpoint["final_energy"],
                  sector_status=endpoint["endpoint"]["sector_status"],
                  wrong_electron_probability=endpoint["endpoint"]["wrong_electron_probability"])
    (directory / "comparison.json").write_text(json.dumps(result, indent=2, allow_nan=False) + "\n")
    return result


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("output", type=Path)
    parser.add_argument("--input", type=Path)
    args = parser.parse_args()
    if args.input:
        run(args.input, args.output)
    else:
        print(json.dumps(compare(args.output), indent=2))
