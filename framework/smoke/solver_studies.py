# ABOUTME: Exercises the registered initialization and noise-study methods on the published hydrogen geometry.
# ABOUTME: Checks saved input identity, fitted energies and electron sectors with dense matrix calculations.
"""Run from framework with QMPROT_GEOMETRIES=smoke/geometries python -m smoke.solver_studies OUTPUT."""
import argparse
from hashlib import sha256
import json
from pathlib import Path

import numpy as np
from scipy.linalg import eigh

from algorithms import ALGORITHMS
from experiments._common import INIT_STRATEGIES, make_backend_config, sample_initial_parameters
from experiments.barren_plateau import INITIALIZATION_ALGORITHMS, run_barren_plateau
from experiments.noise_resilience import DEFAULT_NOISE_MODELS, LAYERED_ALGORITHMS, NOISE_ALGORITHMS, run_noise_resilience
from experiments.provenance import load_problem
from smoke.batch import read_rows, require
from smoke.physics_acceptance import circuit_state, dense_hamiltonian
from smoke.test_noise_physics import density_score

FROZEN = Path(__file__).parent / "reproductions/layered_vqe/hydrogen.npz"
FROZEN_SHA = "fd1e51fabf33165b030d0ea0f1bd9efbe015e8ab64e7a14863f3846fc19546d3"
SHIFT = ("hardware_efficient_vqe", "pennylane_vqe")
GROWING = {"adapt_vqe", "qubit_adapt_vqe", "iqcc_inspired_vqe", "iqcc_true_vqe"}
BUDGETS = {"COBYLA": 40, "BAYESIAN": 6, "GradientDescent": 3}
ATOL = 1e-9


def load_json(path):
    return json.loads(path.read_text())


def digest(path):
    return sha256(path.read_bytes()).hexdigest()


def compare_values(observed, expected, tolerance, location):
    if isinstance(expected, dict):
        require(set(observed) == set(expected), f"Unexpected fields at {location}")
        for key in expected:
            compare_values(observed[key], expected[key], tolerance, f"{location}.{key}")
    elif isinstance(expected, list):
        for i, (left, right) in enumerate(zip(observed, expected, strict=True)):
            compare_values(left, right, tolerance, f"{location}[{i}]")
    elif isinstance(expected, float):
        np.testing.assert_allclose(observed, expected, rtol=0, atol=tolerance, err_msg=location)
    else:
        require(observed == expected, f"Unexpected {location}: {observed!r} != {expected!r}")


def check_input(directory, *, legacy_snapshot=False):
    path = directory / "inputs/hydrogen.npz"
    with np.load(path, allow_pickle=False) as actual, np.load(FROZEN, allow_pickle=False) as frozen:
        require(set(actual.files) - {"encoded_reference_energy"} == set(frozen.files),
                "Hydrogen input schema differs")
        if "encoded_reference_energy" in actual.files:
            encoded = float(actual["encoded_reference_energy"])
            # Resaving a legacy prepared input encodes its absent reference as NaN.
            if not (legacy_snapshot and np.isnan(encoded)):
                np.testing.assert_allclose(encoded, frozen["molecular_hf"],
                                           atol=1e-12, rtol=0, err_msg="encoded_reference_energy")
        for name in frozen.files:
            if name in {"molecule", "preparation", "cs_metadata"}:
                observed = json.loads(str(actual[name]))
                expected = json.loads(str(frozen[name]))
                if name == "preparation":
                    # The two-orbital H₂ control retains its entire orbital space.
                    # Check the operator below even when the preparation label changes.
                    require(observed.pop("active_orbital_basis") in {"canonical", "natural"},
                            "Unknown hydrogen orbital basis")
                    expected.pop("active_orbital_basis")
                    additions = {"active_orbitals": 2, "active_orbital_indices": [0, 1],
                                 "geometry_file_sha256": expected["geometry_sha256"]}
                    for key, value in additions.items():
                        if key in observed:
                            compare_values(observed.pop(key), value, 0., key)
                compare_values(observed, expected, 1e-12, name)
            elif frozen[name].dtype.kind in "fc":
                np.testing.assert_allclose(actual[name], frozen[name], atol=1e-12, rtol=0, err_msg=name)
            else:
                np.testing.assert_array_equal(actual[name], frozen[name], err_msg=name)
    return digest(path)


def reference_scores(problem, algorithm, theta, channel=None, strength=0.):
    solver = ALGORITHMS[algorithm](problem.hamiltonian, n_layers=2, random_seed=42,
                                   backend_config=make_backend_config(4, channel, strength))
    solver.build_ansatz()
    theta = np.asarray(theta)
    h = dense_hamiltonian(problem.hamiltonian)
    wrong_count = np.diag([int(i.bit_count() != 2) for i in range(16)])
    if channel:
        layers = 2 if algorithm in LAYERED_ALGORITHMS else 1
        return (density_score(solver, theta, h, channel, strength, layers),
                density_score(solver, theta, wrong_count.astype(complex), channel, strength, layers))
    state = circuit_state(solver, theta, algorithm)
    if algorithm == "cb_vqe":
        span = np.column_stack((np.eye(16)[:, 12], state))
        basis, singular, _ = np.linalg.svd(span, full_matrices=False)
        basis = basis[:, singular > np.finfo(float).eps * max(span.shape)]
        _, vectors = eigh(basis.conj().T @ h @ basis)
        state = basis @ vectors[:, 0]
    return float(np.vdot(state, h @ state).real), float(np.vdot(state, wrong_count @ state).real)


def check_endpoint(expected, endpoint, csv_energy, csv_leakage, kind):
    require(endpoint["state_kind"] == kind and endpoint["sector_status"] == "measured", "Incorrect endpoint kind")
    distribution = endpoint["electron_count_probabilities"]
    np.testing.assert_allclose([sum(distribution.values()), 1-distribution["2"]], [1., expected[1]], atol=ATOL, rtol=0)
    reported = np.array([endpoint["rescored_energy"], endpoint["wrong_electron_probability"],
                         float(csv_energy), float(csv_leakage)])
    delta = np.abs(reported - [*expected, *expected])
    require(np.max(delta) < ATOL, f"Endpoint replay disagrees with the reported energy or sector: {delta}")
    return dict(energy=float(csv_energy), wrong_electron_probability=float(csv_leakage),
                maximum_replay_difference=float(delta.max()))


def initialization(directory, problem, algorithm, optimizer, channel, generate):
    strength = 0.01 if channel else 0.
    strategies = ("random_uniform",) if channel else INIT_STRATEGIES
    part_a = not channel and algorithm in SHIFT and optimizer == "GradientDescent"
    if generate:
        directory.mkdir(parents=True)
        run_barren_plateau(molecule="hydrogen", algorithm=algorithm, cs_target_qubits=None, optimizer=optimizer,
                           max_iterations=BUDGETS[optimizer], opt_layers=2, opt_strategies=strategies, n_trials=1,
                           skip_part_a=not part_a, layers=(1, 2), bp_strategies=("small_random",), n_samples=5,
                           max_gradient_probes=3, noise_model=channel, noise_strength=strength,
                           save_plots=False, output_dir=directory)
    cfg = load_json(directory / "run_config.json")
    rows = read_rows(directory / "optimization_results.csv")
    trials = load_json(directory / "trial_endpoints.json")
    traces = read_rows(directory / "convergence_traces.csv")
    require([r["strategy"] for r in rows] == list(strategies), "Incomplete initialization grid")
    result = dict(algorithm=algorithm, optimizer=optimizer, channel=channel, input_sha256=check_input(directory),
                  parameter_period=cfg["parameter_period"], bayes_bounds=cfg["bayes_bounds"], trials={})
    require((cfg["algorithm"], cfg["optimizer"], cfg["max_iterations"], cfg["noise_model"], cfg["random_seed"],
             cfg["trial_seed_stride"]) == (algorithm, optimizer, BUDGETS[optimizer], channel, 42, 1009),
            "Initialization config differs")
    require(cfg["endpoint_energy"] == ("two_state_ritz" if algorithm == "cb_vqe" else "variational"), "Incorrect score label")
    for row, trial in zip(rows, trials, strict=True):
        require(trial["strategy"] == row["strategy"] and int(row["trial"]) == trial["trial"] == 0, "Wrong trial")
        theta = sample_initial_parameters(int(row["n_parameters"]), row["strategy"], rng=np.random.default_rng(42))
        np.testing.assert_array_equal(theta, trial["initial_parameters"])
        expected = reference_scores(problem, algorithm, theta, channel, strength)
        np.testing.assert_allclose(float(row["initial_energy"]), expected[0], atol=ATOL, rtol=0)
        trace = [float(t["energy"]) for t in traces if t["strategy"] == row["strategy"]]
        require(len(trace) == int(row["n_iterations"]) + int(optimizer == "GradientDescent")
                and np.isfinite(trace).all(), "Incorrect initialization trace")
        if optimizer == "BAYESIAN":
            require(len(trace) == BUDGETS[optimizer], "Incorrect acquisition count")
            if algorithm == "cb_vqe":
                solver = ALGORITHMS[algorithm](problem.hamiltonian, n_layers=2)
                solver.build_ansatz()
                state = circuit_state(solver, theta, algorithm)
                initial_variational = np.vdot(state, dense_hamiltonian(problem.hamiltonian) @ state).real
            else:
                initial_variational = expected[0]
            np.testing.assert_allclose(trace[0], initial_variational, atol=ATOL, rtol=0)
        final = reference_scores(problem, algorithm, trial["optimal_parameters"], channel, strength)
        kind = "density_matrix" if channel else "ritz_state" if algorithm == "cb_vqe" else "statevector"
        case = check_endpoint(final, trial["endpoint"], row["final_energy"], row["wrong_electron_probability"], kind)
        result["trials"][row["strategy"]] = dict(**case, initial_energy=float(row["initial_energy"]),
                                                  variational_energy=trial["variational_energy"])
    if part_a:
        gradients = read_rows(directory / "gradient_variance_by_parameter.csv")
        require(len(gradients) == 6, "Invalid gradient grid")
        for depth in (1, 2):
            solver = ALGORITHMS[algorithm](problem.hamiltonian, n_layers=depth)
            solver.build_ansatz()
            selected = [r for r in gradients if int(r["n_layers"]) == depth]
            indices = np.array([int(r["gradient_index"]) for r in selected])
            require(len(np.unique(indices)) == 3, "Invalid gradient probe indices")
            deltas = np.eye(solver.n_parameters)[indices] * 1e-5
            values = []
            for sample in range(5):
                angles = np.random.default_rng(42 + 1009 * sample).uniform(-np.pi/8, np.pi/8, solver.n_parameters)
                values.append([(solver.cost_function(angles + d) - solver.cost_function(angles - d))/2e-5 for d in deltas])
            np.testing.assert_allclose([float(r["var_grad"]) for r in selected], np.var(values, axis=0), atol=ATOL, rtol=0.)
            np.testing.assert_allclose([float(r["mean_abs_grad"]) for r in selected], np.abs(values).mean(axis=0), atol=ATOL, rtol=0.)
        result["gradient_rows"] = len(gradients)
    return result


def noise(directory, problem, algorithm, optimizer, generate):
    if generate:
        directory.mkdir(parents=True)
        run_noise_resilience(molecule="hydrogen", algorithm=algorithm, cs_target_qubits=None, optimizer=optimizer,
                             max_iterations=BUDGETS[optimizer], n_layers=2, init_strategy="random_uniform",
                             noise_strengths=(0, .01), save_plots=False, output_dir=directory)
    rows = read_rows(directory / "results.csv")
    cfg = load_json(directory / "run_config.json")
    require((cfg["algorithm"], cfg["optimizer"], cfg["max_iterations"], cfg["random_seed"]) ==
            (algorithm, optimizer, BUDGETS[optimizer], 42), "Noise config differs")
    params = load_json(directory / "optimum_parameters.json")
    traces = read_rows(directory / "convergence_traces.csv")
    endpoints = load_json(directory / "state_endpoints.json")
    expected = {"none@0"} | {f"{m}@{p}" for m in DEFAULT_NOISE_MODELS for p in ("0", "0.01")}
    require(set(endpoints) == expected and len(rows) == len(expected), "Incomplete noise grid")
    theta = sample_initial_parameters(cfg["n_parameters"], "random_uniform", rng=np.random.default_rng(42))
    np.testing.assert_array_equal(theta, params["theta_init"])
    result = dict(algorithm=algorithm, optimizer=optimizer, input_sha256=check_input(directory),
                  parameter_period=cfg["parameter_period"], bayes_bounds=cfg["bayes_bounds"], endpoints={})
    for row in rows:
        model, p = row["noise_model"], float(row["noise_strength"])
        key = f"{model}@{p:g}"
        if not p and model != "none":
            require(endpoints[key] == endpoints["none@0"] and row == {**rows[0], "noise_model": model}, "Incorrect zero anchor")
            continue
        trial = [float(t["energy"]) for t in traces if t["noise_model"] == model and float(t["noise_strength"]) == p]
        require(len(trial) == int(row["n_iterations"]) and np.isfinite(trial).all(), "Incorrect noise trace")
        if optimizer == "BAYESIAN":
            initial_energy = reference_scores(problem, algorithm, theta, model if p else None, p)[0]
            np.testing.assert_allclose([trial[0], min(trial)], [initial_energy, float(row["energy_noisy"])], atol=ATOL, rtol=0.)
        angles = params["theta_noisy_star"][key] if p else params["theta_clean_star"]
        expected_mixed = reference_scores(problem, algorithm, angles, model if p else None, p)
        expected_clean = reference_scores(problem, algorithm, angles)
        mixed = check_endpoint(expected_mixed, endpoints[key]["noisy"], row["energy_noisy"],
                               row["wrong_electron_probability_noisy"], "density_matrix" if p else "statevector")
        clean = check_endpoint(expected_clean, endpoints[key]["clean_at_noisy_params"], row["energy_clean_at_noisy_params"],
                               row["wrong_electron_probability_clean_at_noisy_params"], "statevector")
        require((row["param_drift_l2_wrapped"] == "") == (algorithm == "hva_vqe"), "Incorrect wrapped-drift availability")
        result["endpoints"][key] = dict(noisy=mixed, clean_at_noisy_params=clean)
    return result


def examine(directory, generate):
    require(digest(FROZEN) == FROZEN_SHA, "Frozen control changed")
    require(set(ALGORITHMS) == set(INITIALIZATION_ALGORITHMS) | GROWING, "Unclassified registered algorithm")
    problem = load_problem(FROZEN)
    report = dict(frozen_sha256=FROZEN_SHA, geometry_sha256=problem.preparation["geometry_sha256"], budgets=BUDGETS,
                  seed=42, initialization=[], noise=[], restrictions=dict(
                      growing_algorithms=sorted(GROWING), gradient_shift_algorithms=list(SHIFT),
                      pure_ritz_only="cb_vqe", noise_optimizers=["COBYLA", "BAYESIAN"]))
    for algorithm in INITIALIZATION_ALGORITHMS:
        for optimizer in BUDGETS:
            if optimizer == "GradientDescent" and algorithm not in SHIFT:
                continue
            channels = (None, "phaseflip") if algorithm in NOISE_ALGORITHMS else (None,)
            for channel in channels:
                folder = directory / "initialization" / algorithm / optimizer / (channel or "clean")
                report["initialization"].append(initialization(folder, problem, algorithm, optimizer, channel, generate))
            if optimizer != "GradientDescent" and algorithm in NOISE_ALGORITHMS:
                report["noise"].append(noise(directory / "noise" / algorithm / optimizer, problem, algorithm, optimizer, generate))
    return report


def run(directory):
    report = examine(directory, generate=True)
    (directory / "matrix.json").write_text(json.dumps(report, indent=2, allow_nan=False) + "\n")
    return report


def validate(directory):
    report = examine(directory, generate=False)
    compare_values(report, load_json(directory / "matrix.json"), ATOL, "matrix")
    return report


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("output_directory", type=Path)
    parser.add_argument("--validate", action="store_true")
    args = parser.parse_args()
    (validate if args.validate else run)(args.output_directory)
