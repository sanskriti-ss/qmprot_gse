# ABOUTME: Validates downloaded campaign results using their frozen Hamiltonians and real-study configuration.
# ABOUTME: Independently checks H₂ initialization statistics, adaptive stopping and glycine prefix references.
from dataclasses import replace
from functools import reduce
from hashlib import sha256
import json
from pathlib import Path
from zipfile import ZipFile

import numpy as np
from openfermion import QubitOperator, get_sparse_operator
from PIL import Image
from scipy.linalg import expm

from campaigns.run_campaign import argv_from_args
from experiments.provenance import load_problem
from experiments.replay_trainability import replay
from smoke.batch import read_rows, require
from smoke.fermionic_adapt import excitation_matrix
from smoke.iqcc_control import score_rotations, validate as validate_iqcc_control
from smoke.physics_acceptance import dense_hamiltonian
from smoke.representative_checks import validate_noise, validate_trainability
from smoke.study_campaign import CONFIGURATION, CONTROL_SOURCES


PLOTS = {
    "accuracy": {"cost_evals_vs_k", "energy_vs_n_params", "energy_vs_n_params_linear"},
    "initialization": {"convergence_by_init", "final_energy_by_init", "gradient_variance_vs_layers"},
    "noise": {"convergence", "energy_vs_noise", "param_drift_vs_noise"},
    "prefix": {"mean_absolute_error_vs_term_count", "mean_absolute_error_vs_term_count_filtered"},
    "trainability": {"cost_evals_vs_params", "cost_to_accuracy", "error_vs_cost", "final_energy_vs_params",
                     "runtime_vs_params"},
}


def dense_energy(hamiltonian, angles):
    state = np.zeros(16)
    state[0b1100] = 1
    for layer, wall in enumerate(angles.reshape(-1, 4)):
        if layer:
            for control in range(4):
                target = (control + 1) % 4
                permutation = [i ^ (1 << (3 - target)) if i & (1 << (3 - control)) else i
                               for i in range(16)]
                state = state[permutation]
        matrices = []
        for angle in wall:
            c, s = np.cos(angle / 2), np.sin(angle / 2)
            matrices.append([[c, -s], [s, c]])
        state = reduce(np.kron, matrices) @ state
    return float((state @ hamiltonian @ state).real)


def _check_initialization(directory, matrix):
    gradients = read_rows(directory / "gradient_variance.csv")
    per_parameter = read_rows(directory / "gradient_variance_by_parameter.csv")
    require([(int(r["n_layers"]), r["strategy"]) for r in gradients] == [(1, "small_random"), (2, "small_random")],
            "Initialization gradient cases differ")
    require(len(per_parameter) == 6, "Missing or duplicate per-parameter variances")
    for row, indices in zip(gradients, ((0, 4, 7), (0, 6, 11)), strict=True):
        layers = int(row["n_layers"])
        n_params = (layers + 1)*4
        require((int(row["n_parameters"]), int(row["n_samples"]), int(row["n_probes"])) == (n_params, 5, 3),
                "Initialization dimensions or number of draws differ")
        by_index = [r for r in per_parameter if int(r["n_layers"]) == layers]
        require([(int(r["gradient_index"]), r["strategy"], int(r["n_samples"])) for r in by_index] ==
                [(i, "small_random", 5) for i in indices], "Incorrect per-parameter gradient cases")
        deltas = 1e-5 * np.eye(n_params)[list(indices)]
        values = []
        for sample in range(5):
            rng = np.random.default_rng(42 + 1009 * sample)
            theta = rng.uniform(-np.pi/8, np.pi/8, n_params)
            values.append([(dense_energy(matrix, theta + d) - dense_energy(matrix, theta - d))/2e-5 for d in deltas])
        variances = np.var(values, axis=0)
        magnitudes = np.abs(values).mean(axis=0)
        np.testing.assert_allclose([float(r["var_grad"]) for r in by_index], variances, atol=1e-9, rtol=0)
        np.testing.assert_allclose([float(r["mean_abs_grad"]) for r in by_index], magnitudes, atol=1e-9, rtol=0)
        np.testing.assert_allclose([float(row["var_grad"]), float(row["mean_abs_grad"])],
                                   [variances.mean(), magnitudes.mean()], atol=1e-9, rtol=0)
    results = read_rows(directory / "optimization_results.csv")
    traces = read_rows(directory / "convergence_traces.csv")
    strategies = ["small_random", "near_identity"]
    require([(r["strategy"], r["trial"], r["n_iterations"]) for r in results] ==
            [(s, "0", "14") for s in strategies] and len(traces) == 28, "Incorrect Bayesian run grid")
    for row, width in zip(results, (np.pi/8, 0.05), strict=True):
        trial = [r for r in traces if (r["strategy"], r["trial"]) == (row["strategy"], "0")]
        require([int(r["iteration"]) for r in trial] == list(range(14)), "Incomplete Bayesian acquisitions")
        energies = np.array([float(r["energy"]) for r in trial])
        angles = np.random.default_rng(42).uniform(-width, width, 8)
        np.testing.assert_allclose([energies[0], energies.min()], [dense_energy(matrix, angles), float(row["final_energy"])],
                                   atol=1e-9, rtol=0)
        np.testing.assert_allclose(float(row["initial_energy"]), energies[0], atol=1e-9, rtol=0)
        require(np.isfinite(energies).all(), "Nonfinite Bayesian acquisition")


def _check_accuracy(directory, problem):
    cfg = json.loads((directory / "run_config.json").read_text())
    rows = read_rows(directory / "results.csv")
    require([int(r["k"]) for r in rows] == [1, 2, 5], "Incorrect requested ADAPT checkpoints")
    trajectory = read_rows(directory / "full_convergence.csv")
    attained = cfg["actual_max_k"]
    require(0 < attained < 5 and [int(r["k"]) for r in trajectory] == list(range(1, attained + 1)),
            "Missing adaptive growth or early stopping")
    for row in rows:
        k = int(row["k"])
        achieved = min(k, attained)
        endpoint = trajectory[achieved - 1]
        require(int(row["attained_k"]) == achieved and row["status"] == ("ok" if k == achieved else "stopped_early")
                and int(row["cumulative_cost_evals"]) == int(endpoint["cumulative_cost_evals"]) > 0,
                f"Incorrect adaptive cost or attainment at {k}")
        energy = float(row["energy"])
        np.testing.assert_allclose([energy, float(row["error_vs_casci"]), float(row["error_vs_hf"])],
                                   [float(endpoint["energy"]), energy - problem.casci_energy,
                                    energy - problem.hf_energy], rtol=0, atol=1e-9)
        require(abs(energy - problem.casci_energy) < 1e-8, f"Adaptive H₂ endpoint incorrect at {k}")
    return attained


def _check_prefix(directory, problem, input_file, settings):
    ham = problem.hamiltonian
    rows = read_rows(directory / "batch_all_results.csv")
    cases = [json.loads(line) for line in (directory / "cases.jsonl").read_text().splitlines()]
    expected = [(n, a, s) for n in settings["term-counts"] for a in settings["algorithms"]
                for s in settings["seeds"]]
    require([(int(r["n_parameters"]), r["algorithm"], int(r["random_seed"])) for r in rows] == expected,
            "Incorrect paired prefix result grid")
    require([(r["prefix_terms"], r["algorithm"], r["random_seed"]) for r in cases] == expected,
            "Incorrect paired prefix case records")
    digest = sha256(input_file.read_bytes()).hexdigest()
    max_rotations = 0
    for row, case in zip(rows, cases, strict=True):
        n = case["prefix_terms"]
        prefix = replace(ham, coefficients=ham.coefficients[:n], pauli_strings=ham.pauli_strings[:n], n_terms=n)
        matrix = dense_hamiltonian(prefix)
        reference = float(np.linalg.eigvalsh(matrix)[0])
        energy = float(row["energy"])
        require(case["input_sha256"] == digest and case["molecule"] == row["molecule"] == "gly"
                and row["status"] == "ok" and int(row["n_qubits"]) == 4 and int(row["n_cost_evals"]) > 0
                and energy >= reference - 1e-8, f"Invalid {row['algorithm']} glycine prefix {n}")
        np.testing.assert_allclose([float(row["prefix_reference_energy"]), case["prefix_reference"],
                                    float(row["full_reference_energy"]), case["energy"], float(row["error_vs_prefix_reference"])],
                                   [reference, reference, problem.casci_energy, energy, energy - reference], atol=1e-8, rtol=0)
        sector = case["endpoint_sector"]
        probability = float(row["wrong_electron_probability"])
        require(0 <= probability <= 1, "Invalid physical electron probability")
        np.testing.assert_allclose([sector["rescored_energy"], sector["wrong_electron_probability"]],
                                   [energy, probability], atol=1e-8, rtol=0)
        if case["algorithm"] == "iqcc_true_vqe":
            state = np.asarray(ham.cs_initial_state, dtype=complex)
            generators = case["metadata"]["selected_generators"]
            angles = case["parameters"]
            require(len(generators) == len(angles) == int(row["algorithm_n_parameters"]) and
                    case["metadata"]["n_molecular_energy_evals"] == 1 + len(angles), "Incomplete iQCC history")
            np.testing.assert_allclose(score_rotations(state, matrix, generators, angles), case["energy"], rtol=0, atol=1e-8)
            max_rotations = max(max_rotations, len(angles))
    progress = json.loads((directory / "progress.json").read_text())
    require(progress["status"] == "finished" and progress["successful_rows"] == len(expected) and
            progress["failed_cases"] == progress["skipped_cases"] == 0, "Incomplete prefix sweep")
    return max_rotations


def _check_fermionic_control(directory):
    control = json.loads((directory / "fermionic-control.json").read_text())
    archive = directory / "fermionic-control.zip"
    require(sha256(archive.read_bytes()).hexdigest() == control["control_archive_sha256"],
            "Damaged fermionic control archive")
    references = {
        "hydrogen": ([0, 1], -1.117505884204331, -1.1368465754720538,
                     "44b816528c2cd1296765d5497e6d7040e63a6eae377c98da63c3079982cf9357"),
        "gly": ([19, 20], -279.1115147283362, -279.1115600159873,
                "d6c06b5e1faec00a1d5d3ea6ff47232a9b5f049ffc375ac502f404179704cb8f"),
    }
    require(set(control["molecules"]) == set(references) and control["energy_evaluation_limit"] == 300 and
            control["solver_settings"] == dict(max_operators=3, max_iterations=100, convergence_threshold=1e-9,
                                               gradient_threshold=1e-8, random_seed=42),
            "Incorrect molecular or solver grid in fermionic control")
    with ZipFile(archive) as source:
        geometries = [f"smoke/geometries/{m}/{m}.h5" for m in references]
        require(source.testzip() is None and sorted(source.namelist()) == sorted([*CONTROL_SOURCES, *geometries]),
                "Incomplete fermionic control archive")
        for name, path in zip(references, geometries, strict=True):
            m = control["molecules"][name]
            orbitals, hf, casci, digest = references[name]
            require(sha256(source.read(path)).hexdigest() == m["geometry_sha256"] == digest and
                    m["n_active_electrons"] == 2 and m["active_orbitals"] == orbitals,
                    f"Incorrect {name} fermionic geometry or active space")
            words = m["pauli_strings"]
            require(len(words) == len(set(words)) == len(m["coefficients"]) == m["n_terms"] and
                    all(len(word) == 4 and set(word) <= set("IXYZ") for word in words), f"Invalid {name} control input")
            matrix = np.zeros((16, 16), dtype=complex)
            for word, coefficient in zip(words, m["coefficients"], strict=True):
                term = tuple((wire, p) for wire, p in enumerate(word) if p != "I")
                matrix += coefficient * get_sparse_operator(QubitOperator(term), n_qubits=4).toarray()
            require(sha256(matrix.astype("<c16").tobytes()).hexdigest() == m["hamiltonian_sha256"],
                    f"Incorrect {name} control Hamiltonian fingerprint")
            determinants = [3, 6, 9, 12]  # One spin-up and one spin-down electron.
            sector = np.linalg.eigvalsh(matrix[np.ix_(determinants, determinants)])[0]
            np.testing.assert_allclose([matrix[12, 12], m["hf"], m["casci"], m["sector_energy"], sector],
                                       [hf, hf, casci, casci, casci], atol=1e-8, rtol=0)
            rows = m["runs"]
            pools = {"registered": 12, "approximate_matched_pool": 3, "fermionic_matched_pool": 3}
            require(set(rows) == set(pools) and rows["approximate_matched_pool"]["pool"] == rows["fermionic_matched_pool"]["pool"],
                    f"Incorrect {name} ADAPT comparison pools")
            for method, row in rows.items():
                require(len(row["pool"]) == pools[method] and 0 < row["cost_evals"] <= 300 and
                        all(op in row["pool"] for op in row["selected"]) and
                        len(row["selected"]) == len(row["angles"]) <= 3 and np.isfinite(row["angles"]).all(),
                        f"Invalid {name} {method} operator history")
                np.testing.assert_allclose(row["error_mha"], 1000*(row["energy"]-m["casci"]), atol=1e-7, rtol=0)
            physical = rows["fermionic_matched_pool"]
            require(physical["selected"] and 0 <= physical["off_particle_sector"] < 1e-12,
                    f"Wrong particle sector for {name} fermionic endpoint")
            state = np.eye(16)[:, 12]
            for (_, *sites), angle in zip(physical["selected"], physical["angles"], strict=True):
                state = expm(angle/2 * excitation_matrix(sites, 4)) @ state
            np.testing.assert_allclose([physical["energy"], np.vdot(state, matrix @ state), np.linalg.norm(state[determinants])],
                                       [casci, physical["energy"], 1], atol=1e-8, rtol=0)
            expected = [(method, op, angle) for method in ("approximate_matched_pool", "fermionic_matched_pool")
                        for op in (["single", 0, 2], ["double", 0, 1, 2, 3]) for angle in (0., 0.4)]
            gates = m["matched_rotations"]
            require([(g["circuit"], g["operator"], g["angle"]) for g in gates] == expected,
                    f"Missing {name} gate comparisons")
            for gate in gates:
                if gate["circuit"] == "fermionic_matched_pool":
                    np.testing.assert_allclose([gate["fidelity_to_jw"], gate["off_particle_sector"]],
                                               [1, 0], atol=1e-12, rtol=0)
                elif gate["operator"][0] == "double" and gate["angle"] == 0:
                    np.testing.assert_allclose([gate["fidelity_to_jw"], gate["off_particle_sector"]],
                                               [0, 1], atol=1e-12, rtol=0)
    return {name: m["runs"]["fermionic_matched_pool"]["error_mha"] for name, m in control["molecules"].items()}


def validate(directory):
    directory = Path(directory)
    configs = {f.stem: json.loads(f.read_text()) for f in CONFIGURATION.glob("*.json")}
    require(set(configs) == {"accuracy", "initialization", "noise", "prefix", "trainability"}, "Missing campaign specs")
    manifest = json.loads((directory / "manifest.json").read_text())["runs"]
    require(len(manifest) == 1 and [r["file"] for r in manifest[0]["experiments"]] ==
            [f"{name}.json" for name in sorted(configs)], "Incomplete or duplicate campaign launches")
    for command in manifest[0]["experiments"]:
        name = Path(command["file"]).stem
        require(command["argv"][1:-2] == argv_from_args(configs[name])[1:] and
                command["argv"][-2:] == ["--output-dir", command["results"]] and
                Path(command["results"]).parts[-2:] == ("results", name), f"Incorrect {name} command")
    log = (directory / "campaign.log").read_text()
    require(log.count("Bayesian optimizer backend=gp, n_params=8, n_calls=14, n_initial_points=11") == 2,
            "Did not fit both configured Gaussian-process surrogates")
    require(not any(s in log for s in ("[ERROR]", "[WARNING]", "Warning:", "Traceback (most recent call last)")),
            "Campaign logged a warning or failure")
    report = {}
    source_hashes = set()
    for name, spec in configs.items():
        require(json.loads((directory / f"{name}.json").read_text()) == spec, f"Incorrect saved {name} spec")
        folder = directory / "results" / name
        data = "gly" if name == "prefix" else "hydrogen"
        runtime = json.loads((folder / "runtime.json").read_text())
        archive = folder / "framework-source.zip"
        require(sha256(archive.read_bytes()).hexdigest() == runtime["source_sha256"], f"Damaged {name} source snapshot")
        source_hashes.add(runtime["source_sha256"])
        with ZipFile(archive) as source:
            require(source.testzip() is None and spec["module"].replace(".", "/") + ".py" in source.namelist() and
                    "contextual_subspace/core.py" in source.namelist() and
                    "drivers/run_barren_plateau_part_b_all_molecules.py" in source.namelist(),
                    f"Incomplete {name} source archive")
        require(runtime["packages"]["scikit-optimize"] == "0.10.2", f"Missing pinned optimizer in {name}")
        input_file = folder / "inputs" / f"{data}.npz"
        problem = load_problem(input_file)
        require(problem.molecule == data and problem.basis == "sto-3g" and problem.n_qubits_final == 4
                and problem.cs_reduced == (name == "prefix"), f"Incorrect input for {name}")
        if name == "prefix":
            require(problem.n_qubits_active == 12 and np.isclose(np.linalg.norm(problem.hamiltonian.cs_initial_state), 1),
                    "Expected a projected glycine reference state")
            reduction = problem.cs_metadata
            require(reduction["original_qubits"] == 12 and reduction["reduced_qubits"] == 4 and
                    len(reduction["stabilizers"]) == reduction["n_generators_stabilized"] == 8 and
                    all(s["eigenvalue"] in {-1, 1} and len(s["pauli"]) == 12 for s in reduction["stabilizers"]) and
                    reduction["rotations"] and all(np.isfinite(r["angle_rad"]) and len(r["pauli"]) == 12
                                                   for r in reduction["rotations"]),
                    "Incomplete glycine transformation or sector provenance")
            np.testing.assert_allclose(problem.casci_energy, -279.1376629146243, rtol=0, atol=1e-7)
            report["max_iqcc_rotations"] = _check_prefix(folder, problem, input_file, spec["args"])
        else:
            matrix = dense_hamiltonian(problem.hamiltonian)
            require(problem.n_qubits_active == 4 and problem.hamiltonian.cs_initial_state is None,
                    f"Unexpected H₂ input for {name}")
            np.testing.assert_allclose([problem.casci_energy, np.linalg.eigvalsh(matrix)[0], problem.hf_energy,
                                        matrix[0b1100, 0b1100].real],
                                       [-1.1368465754720538]*2 + [-1.117505884204331]*2, rtol=0, atol=1e-9)
            if name == "initialization":
                _check_initialization(folder, matrix)
            elif name == "accuracy":
                report["attained_k"] = _check_accuracy(folder, problem)
            elif name == "noise":
                validate_noise(folder, args=spec["args"])
            elif name == "trainability":
                validate_trainability(folder, ("hydrogen",), batch=False, targets=(2, 21), args=spec["args"])
                report["trainability_replay"] = replay(folder)
        figures = tuple(folder.glob("*.png"))
        require({figure.stem for figure in figures} == PLOTS[name], f"Incomplete plot grid in {name}")
        for figure in figures:
            with Image.open(figure) as image:
                image.verify()
        report[name] = {"molecule": data, "qubits": 4, "plots": len(figures), "source_sha256": runtime["source_sha256"]}
    require(len(source_hashes) == 1, "Studies ran from different source snapshots")
    report["fermionic_error_mha"] = _check_fermionic_control(directory)
    report["fixed_iqcc"] = validate_iqcc_control(directory)
    from smoke.solver_studies import validate as validate_solver_studies
    matrix = validate_solver_studies(directory / "solver-studies")
    report["solver_studies"] = dict(initialization_configurations=len(matrix["initialization"]),
                                     noise_configurations=len(matrix["noise"]),
                                     frozen_sha256=matrix["frozen_sha256"])
    matrix_log = (directory / "solver-studies.log").read_text()
    require(not any(s in matrix_log for s in ("[ERROR]", "[WARNING]", "Warning:", "Traceback (most recent call last)")),
            "Solver matrix logged a warning or failure")
    receipts = list((directory / "solver-studies").rglob("runtime.json"))
    require(len(receipts) == len(matrix["initialization"]) + len(matrix["noise"]), "Incomplete solver matrix provenance")
    for path in receipts:
        digest = sha256((path.parent / "framework-source.zip").read_bytes()).hexdigest()
        require(digest in source_hashes and digest == json.loads(path.read_text())["source_sha256"],
                f"Unexpected scientific source in {path.parent}")
    from smoke.paired_studies import validate as validate_paired_studies
    report["paired_studies"] = validate_paired_studies(directory / "paired-studies")
    require(report["paired_studies"]["source_sha256"] in source_hashes,
            "Paired study source differs from the campaign")
    return report
