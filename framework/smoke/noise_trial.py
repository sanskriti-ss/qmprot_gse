# ABOUTME: Times the native clean and depolarizing fits on the frozen eight-qubit glycine operator.
# ABOUTME: Records complete binary64 parameters, optimizer callbacks and returned energies for independent replay.
import argparse
import csv
from hashlib import sha256
import json
import math
from pathlib import Path
import sys
from time import perf_counter

import numpy as np

from experiments import noise_resilience
from experiments.provenance import framework_source


INPUT = Path(__file__).parent / "reproductions/layered_vqe/contextual_gly.npz"
INPUT_SHA256 = "8c07e44128d2bc4463a351e80c37b88d384b605a72a353bb17d9ab19517dbe85"


def run(directory):
    from threadpoolctl import threadpool_info

    directory.mkdir(parents=True, exist_ok=False)
    if sha256(INPUT.read_bytes()).hexdigest() != INPUT_SHA256:
        raise ValueError("Unexpected contextual glycine input")
    native_fit = noise_resilience._optimize_one
    timings = {}
    with (directory / "events.jsonl").open("x", buffering=1) as stream:

        def fit(solver, initial):
            name = solver.backend_config.noise_model or "clean"
            if name in timings:
                raise ValueError(f"Repeated {name} fit")
            native_callback, native_optimize = solver.callback, solver.optimize
            in_callback = False

            def callback(*args, **kwargs):
                nonlocal in_callback
                in_callback = True
                try:
                    return native_callback(*args, **kwargs)
                finally:
                    in_callback = False

            def optimize(*args, **kwargs):
                # The native fit has installed its analytic parameter cache at this point.
                evaluate = solver.cost_function

                def record(theta):
                    start = perf_counter()
                    energy = evaluate(theta)
                    event = dict(fit=name, callback=in_callback, parameters=np.asarray(theta, dtype="<f8").tobytes().hex(),
                                 energy=float(energy).hex(), seconds=perf_counter() - start)
                    stream.write(json.dumps(event, separators=(",", ":"), allow_nan=False) + "\n")
                    return energy

                solver.cost_function = record
                try:
                    return native_optimize(*args, **kwargs)
                finally:
                    solver.cost_function = evaluate

            solver.callback, solver.optimize = callback, optimize
            try:
                start = perf_counter()
                result = native_fit(solver, initial)
                timings[name] = dict(seconds=perf_counter() - start, optimizer=solver.optimizer_outcome,
                                     device=solver.device.name, optimal_energy=result[1].hex())
                return result
            finally:
                solver.callback, solver.optimize = native_callback, native_optimize

        argv = ["experiments.noise_resilience", "--molecule", "gly", "--basis", "sto-3g",
                "--cs-target-qubits", "8", "--input-snapshot", str(INPUT), "--algorithm", "hardware_efficient_vqe",
                "--n-layers", "2", "--rotation-gates", "RY", "--prepare-hf", "--optimizer", "COBYLA",
                "--max-iter", "150", "--convergence-threshold", "1e-10", "--init-strategy", "small_random",
                "--seed", "42", "--noise-models", "depolarizing", "--noise-strengths", "0", "0.01",
                "--output-dir", str(directory / "results")]
        previous = sys.argv
        noise_resilience._optimize_one = fit
        try:
            sys.argv = argv
            start = perf_counter()
            noise_resilience.main()
            elapsed = perf_counter() - start
        finally:
            noise_resilience._optimize_one = native_fit
            sys.argv = previous
    (directory / "timings.json").write_text(json.dumps(dict(native_cli_seconds=elapsed, fits=timings,
                                                            native_threads=threadpool_info()), indent=2) + "\n")


def validate(directory, expected_source=None):
    results = directory / "results"
    def read(name):
        return json.loads((results / name).read_text())
    scientific_source = sha256((results / "framework-source.zip").read_bytes()).hexdigest()
    if scientific_source != read("runtime.json")["source_sha256"] or (
            expected_source is not None and scientific_source != expected_source):
        raise ValueError("Scientific source differs from the worker record or requested fingerprint")
    archived = results / "inputs/gly.npz"
    if sha256(archived.read_bytes()).hexdigest() != INPUT_SHA256:
        raise ValueError("Molecular snapshot differs from the frozen input")
    timings = json.loads((directory / "timings.json").read_text())
    config, angles = read("run_config.json"), read("optimum_parameters.json")
    expected = dict(molecule="gly", basis="sto-3g", cs_target_qubits=8, n_qubits_active=12, n_qubits_final=8,
                    n_layers=2, n_parameters=24, optimizer="COBYLA", max_iterations=150, init_strategy="small_random",
                    algorithm="hardware_efficient_vqe", rotation_gates="RY", prepare_hf=True, convergence_threshold=1e-10,
                    random_seed=42, noise_models=["depolarizing"], noise_strengths=[0., .01])
    if any(config[k] != v for k, v in expected.items()):
        raise ValueError("Incorrect native study configuration")
    with (results / "results.csv").open() as stream:
        clean, zero, noisy = csv.DictReader(stream)
    if (clean["noise_model"], zero["noise_model"], noisy["noise_model"], clean["noise_strength"],
            zero["noise_strength"], noisy["noise_strength"]) != ("none", "depolarizing", "depolarizing", "0", "0", "0.01"):
        raise ValueError("Incorrect fitted cases")
    if any(clean[k] != zero[k] for k in clean if k != "noise_model"):
        raise ValueError("Zero strength differs from the clean baseline")
    events = [json.loads(s) for s in (directory / "events.jsonl").read_text().splitlines()]
    start = np.asarray(angles["theta_init"], dtype="<f8").tobytes().hex()
    seen = {}
    for name, optimum, row in (("clean", angles["theta_clean_star"], clean),
                               ("depolarizing", angles["theta_noisy_star"]["depolarizing@0.01"], noisy)):
        group = [e for e in events if e["fit"] == name]
        objective = [e for e in group if not e["callback"]]
        callbacks = [e for e in group if e["callback"]]
        fit = timings["fits"][name]
        if (not objective or len(objective) != fit["optimizer"]["optimizer_evaluations"] or len(objective) > 150
                or len(callbacks) != int(row["n_iterations"]) or objective[0]["parameters"] != start):
            raise ValueError(f"Inconsistent native work counts or start for {name}")
        best = np.asarray(optimum, dtype="<f8").tobytes().hex()
        if not any(e["parameters"] == best and e["energy"] == fit["optimal_energy"] for e in objective):
            raise ValueError(f"Missing optimizer-selected energy or parameters for {name}")
        for e in group:
            if (len(bytes.fromhex(e["parameters"])) != 24 * 8 or not math.isfinite(float.fromhex(e["energy"]))
                    or not math.isfinite(e["seconds"]) or e["seconds"] < 0):
                raise ValueError(f"Incorrect binary objective record for {name}")
        parameters = {e["parameters"] for e in group}
        seen[name] = dict(objective_calls=len(objective), callbacks=len(callbacks), distinct_vectors=len(parameters),
                          repeated_calls=len(group)-len(parameters), objective_seconds=sum(e["seconds"] for e in objective),
                          callback_seconds=sum(e["seconds"] for e in callbacks), **fit)
    if len(events) != sum(c["objective_calls"]+c["callbacks"] for c in seen.values()):
        raise ValueError("Unrecognized fit in objective trace")
    endpoint = read("state_endpoints.json")
    if endpoint["none@0"] != endpoint["depolarizing@0"] or any(
            r["sector_status"] != "measured" for item in endpoint.values() for r in item.values()):
        raise ValueError("Missing measured parent sectors")
    for name in ("convergence.png", "param_drift_vs_noise.png", "energy_vs_noise.png"):
        if (results / name).read_bytes()[:8] != b"\x89PNG\r\n\x1a\n":
            raise ValueError(f"Missing native figure {name}")
    return dict(fits=seen, native_cli_seconds=timings["native_cli_seconds"], input_sha256=INPUT_SHA256,
                source_sha256=scientific_source,
                trace_sha256=sha256((directory / "events.jsonl").read_bytes()).hexdigest())


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("directory", type=Path)
    parser.add_argument("--validate", action="store_true")
    parser.add_argument("--source-sha", help="Expected scientific source archive SHA-256")
    args = parser.parse_args()
    if not args.validate:
        run(args.directory)
    expected = args.source_sha if args.validate else sha256(framework_source()).hexdigest()
    print(json.dumps(validate(args.directory, expected), indent=2))
