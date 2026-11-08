# ABOUTME: Times molecular objectives and the individual search phases in the frozen glycine initialization trial.
# ABOUTME: Records candidate prediction digests and exact optimizer outputs for whichever scientific source is loaded.
import argparse
from collections import defaultdict
from functools import wraps
from hashlib import sha256
from importlib.util import find_spec
import json
from pathlib import Path
from time import perf_counter

import numpy as np
from skopt.learning.forest import ExtraTreesRegressor
from skopt.space import Space

from algorithms.vqe_hardware_efficient import HardwareEfficientVQE
from core.base_vqe import BaseVQE
from experiments.barren_plateau import run_optimization_comparison
from experiments.provenance import framework_source, load_problem


def run(path):
    samples = defaultdict(list)
    predictions = []
    angles = set()
    stream = sha256()
    originals = []
    forest = ExtraTreesRegressor
    if find_spec("core.forest") is not None:
        from core.forest import EnergyForest
        forest = EnergyForest

    def time_method(cls, method, label):
        original = getattr(cls, method)

        @wraps(original)
        def call(instance, *args, **kwargs):
            start = perf_counter()
            result = original(instance, *args, **kwargs)
            samples[label].append(perf_counter() - start)
            if label == "predict" and len(samples[label]) in (1, 84, 168):
                mean, std = result
                predictions.append(dict(forest_number=len(samples[label]), grid_shape=args[0].shape,
                                        prediction_sha256=sha256(mean.tobytes() + std.tobytes()).hexdigest()))
            if label == "energy":
                key = np.asarray(args[0], dtype=np.float64).tobytes()
                angles.add(key)
                stream.update(key)
                stream.update(np.float64(result).tobytes())
            return result

        originals.append((cls, method, original, method in cls.__dict__))
        setattr(cls, method, call)

    problem = load_problem(path)
    try:
        for cls, name, label in ((BaseVQE, "optimize", "optimizer"), (HardwareEfficientVQE, "cost_function", "energy"),
                                 (forest, "fit", "forest_fit"), (forest, "predict", "predict"),
                                 (Space, "rvs", "sample"), (Space, "transform", "transform")):
            time_method(cls, name, label)
        start = perf_counter()
        row, = run_optimization_comparison(
            problem, algorithm="hardware_efficient_vqe", n_layers=16, strategies=("small_random",), n_trials=1,
            optimizer="BAYESIAN", max_iterations=200, convergence_threshold=1e-10, learning_rate=.1,
            noise_model=None, noise_strength=0., random_seed=42)
        trial_seconds = perf_counter() - start
    finally:
        for cls, method, original, owned in reversed(originals):
            if owned:
                setattr(cls, method, original)
            else:
                delattr(cls, method)
    if (row.n_parameters != 68 or len(samples["energy"]) != 200 or len(row.convergence_history) != 200
            or len(samples["predict"]) != 168):
        raise ValueError("Incomplete 200-call forest trial")
    return dict(source_sha256=sha256(framework_source()).hexdigest(), input_sha256=sha256(path.read_bytes()).hexdigest(),
                trial_seconds=trial_seconds, phases={k: dict(calls=len(v), seconds=sum(v)) for k, v in samples.items()},
                logical_objectives=len(samples["energy"]), distinct_angle_vectors=len(angles),
                objective_stream_sha256=stream.hexdigest(), initial_energy=row.initial_energy, final_energy=row.final_energy,
                winning_angles_sha256=sha256(np.array(row.optimal_parameters).tobytes()).hexdigest(),
                candidate_predictions=predictions)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("input", type=Path)
    parser.add_argument("output", type=Path)
    args = parser.parse_args()
    args.output.write_text(json.dumps(run(args.input), indent=2, allow_nan=False) + "\n")
