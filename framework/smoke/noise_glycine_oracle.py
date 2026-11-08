# ABOUTME: Independently propagates the frozen glycine HE circuit and lifts its states into the parent orbitals.
# ABOUTME: Replays native fit events, physical endpoints, channel placement, drift metrics and run provenance.
"""From framework: python -m smoke.noise_glycine_oracle RUN [--output receipt.json]."""
import argparse
import csv
from functools import reduce
from hashlib import sha256
import json
from pathlib import Path
import shutil
from tempfile import TemporaryDirectory

import numpy as np

FROZEN = Path(__file__).parent / "reproductions/layered_vqe/contextual_gly.npz"
INPUT_SHA = "8c07e44128d2bc4463a351e80c37b88d384b605a72a353bb17d9ab19517dbe85"
SOURCE_SHA = "1d680189899cd170d88fa35c9407740f7b6d7d2c82e92bb0c5155aa38a021d1a"
PAULI = dict(I=np.eye(2), X=np.array([[0, 1], [1, 0]]), Y=np.array([[0, -1j], [1j, 0]]), Z=np.diag([1, -1]))


def require(condition, message):
    if not condition:
        raise ValueError(message)


def digest(path):
    return sha256(path.read_bytes()).hexdigest()


def read(path):
    return json.loads(path.read_text())


def table(path):
    with path.open(newline="") as stream:
        return list(csv.DictReader(stream))


def close(actual, expected, tolerance, name):
    difference = np.max(np.abs(np.asarray(actual, dtype=float) - np.asarray(expected, dtype=float)))
    require(difference <= tolerance, f"{name} differs by {difference}")
    return float(difference)


def rotation_layer(angles):
    rotations = [np.array([[c, -s], [s, c]]) for c, s in zip(np.cos(angles / 2), np.sin(angles / 2))]
    return reduce(np.kron, rotations)


def ring_permutation(n):
    destinations = np.arange(2**n)
    for control in range(n):
        target = (control + 1) % n
        destinations ^= ((destinations >> (n - 1 - control)) & 1) << (n - 1 - target)
    return np.argsort(destinations)


def depolarize(density, strength):
    """Apply independent Pauli errors per wire using the one-wire reduced trace channel."""
    n = density.shape[0].bit_length() - 1
    for wire in range(n):
        tensor = density.reshape((2,) * (2 * n))
        blocks = np.moveaxis(tensor, [wire, n + wire], [0, 1])
        partial = blocks[0, 0] + blocks[1, 1]
        blocks *= 1 - 4 * strength / 3
        blocks[0, 0] += 2 * strength / 3 * partial
        blocks[1, 1] += 2 * strength / 3 * partial
    return density


class GlycineCircuit:
    def __init__(self, fixture):
        with np.load(fixture, allow_pickle=False) as data:
            self.start = data["initial_state"]
            self.h = sum(c * reduce(np.kron, [PAULI[p] for p in word])
                         for c, word in zip(data["coefficients"], data["pauli_strings"], strict=True))
            self.meta = json.loads(str(data["cs_metadata"]))
            self.preparation = json.loads(str(data["preparation"]))
            self.casci = float(data["casci"])
            self.hf = float(data["molecular_hf"])
            require(len(data["coefficients"]) == 361 and int(data["n_qubits"]) == 8, "Wrong frozen operator")
        require(np.array_equal(self.h, self.h.conj().T), "Non-Hermitian Hamiltonian")
        self.ring = ring_permutation(8)
        self.bounds = np.linalg.eigvalsh(self.h)[[0, -1]]

    def state(self, theta, strength):
        rotations = [rotation_layer(layer) for layer in np.asarray(theta).reshape(3, 8)]
        ket = rotations[0] @ self.start
        ket = rotations[1] @ ket[self.ring]
        if strength == 0:
            return rotations[2] @ ket[self.ring]
        density = depolarize(np.outer(ket, ket.conj()), strength)
        unitary = rotations[2]
        density = unitary @ density[np.ix_(self.ring, self.ring)] @ unitary.T
        return depolarize(density, strength)

    def energy(self, state):
        value = np.vdot(state, self.h @ state) if state.ndim == 1 else np.einsum("ji,ij->", self.h, state)
        require(abs(value.imag) < 1e-10 and self.bounds[0]-1e-9 <= value.real <= self.bounds[1]+1e-9,
                "Energy outside operator spectrum")
        return float(value.real)


def parent_isometry(circuit):
    """Insert fixed stabilizer bits and apply the inverse rotations to every embedded basis column."""
    meta = circuit.meta
    n, reduced = meta["original_qubits"], meta["reduced_qubits"]
    require((n, reduced) == (12, 8), "Wrong parent or contextual register")
    fixed = {item["pauli"].index("Z"): "0" if item["eigenvalue"] == 1 else "1" for item in meta["stabilizers"]}
    require(len(fixed) == n - reduced, "Wrong number of occupied stabilizer wires")
    vectors = np.zeros((2**n, 2**reduced), dtype=complex)
    for col in range(2**reduced):
        bits = iter(format(col, f"0{reduced}b"))
        row = "".join(fixed[i] if i in fixed else next(bits) for i in range(n))
        vectors[int(row, 2), col] = 1
    for rotation in reversed(meta["rotations"]):
        tensor = vectors.reshape((2,) * n + (2**reduced,))
        for wire, letter in enumerate(rotation["pauli"]):
            if letter != "I":
                tensor = np.moveaxis(np.tensordot(PAULI[letter], tensor, axes=(1, wire)), 0, wire)
        angle = rotation["angle_rad"] / 2
        vectors = np.cos(angle) * vectors - 1j * np.sin(angle) * tensor.reshape(vectors.shape)
    np.testing.assert_allclose(vectors.conj().T @ vectors, np.eye(2**reduced), atol=1e-13, rtol=0)
    electrons = circuit.preparation["active_electrons"]
    hf_index = int("1" * electrons + "0" * (n - electrons), 2)
    projected_hf = vectors[hf_index].conj()
    overlap = abs(np.vdot(circuit.start, projected_hf)) ** 2
    close(overlap, meta["hf_projection_probability"], 1e-12, "Parent HF projection")
    return vectors


def physical_endpoint(circuit, isometry, state):
    if state.ndim == 1:
        eigenvalues = np.ones(1)
        ensemble = (isometry @ state)[:, None]
    else:
        np.testing.assert_allclose(state, state.conj().T, atol=1e-12, rtol=0)
        eigenvalues, eigenvectors = np.linalg.eigh(state)
        require(eigenvalues.min() > -1e-12, "Negative density eigenvalue")
        ensemble = isometry @ eigenvectors
    population = (np.abs(ensemble) ** 2) @ eigenvalues
    close(population.sum(), 1., 1e-11, "Parent populations")
    n = circuit.meta["original_qubits"]
    basis = np.arange(2**n)
    bits = np.array([list(map(int, format(i, f"0{n}b"))) for i in basis])
    number = bits.sum(axis=1)
    sz = (bits[:, ::2].sum(axis=1) - bits[:, 1::2].sum(axis=1)) / 2
    weights = np.bincount(number, weights=population, minlength=n+1)
    spin_squared = np.dot(sz**2, population)
    # S² = Sz² + (S+ S- + S- S+)/2 in paired alpha/beta orbitals.
    for source, target in ((0, 1), (1, 0)):
        flipped = np.zeros_like(ensemble)
        for start in range(0, n, 2):
            occupied = basis[(bits[:, start+source] == 1) & (bits[:, start+target] == 0)]
            swapped = occupied ^ (3 << (n-2-start))
            flipped[swapped] += ensemble[occupied]
        spin_squared += .5 * ((np.abs(flipped)**2).sum(axis=0) @ eigenvalues)
    target_count = circuit.preparation["active_electrons"]
    target_spin = circuit.preparation["spin"]
    return dict(state_kind="statevector" if state.ndim == 1 else "density_matrix", expected_electron_count=target_count,
                electron_count_probabilities={str(i): p for i, p in enumerate(weights)},
                wrong_electron_probability=1 - weights[target_count], mean_electron_count=population @ number,
                mean_spin_z=population @ sz, mean_spin_squared=spin_squared,
                target_spin_z_probability=population[2*sz == target_spin].sum(), rescored_energy=circuit.energy(state),
                sector_status="measured")


def check_endpoint(expected, actual):
    errors = {}
    require(actual.keys() == expected.keys(), "Endpoint fields differ")
    for key, value in expected.items():
        if key == "electron_count_probabilities":
            require(value.keys() == actual[key].keys(), "Incomplete electron distribution")
            errors[key] = close([actual[key][i] for i in value], list(value.values()), 1e-10, key)
        elif key in ("state_kind", "sector_status", "expected_electron_count"):
            require(value == actual[key], f"Incorrect {key}")
        else:
            errors[key] = close(actual[key], value, 5e-10 if key == "rescored_energy" else 1e-10, key)
    return errors


def check_config(config, runtime, archive, source_sha, circuit):
    expected = dict(molecule="gly", algorithm="hardware_efficient_vqe", basis="sto-3g", cs_target_qubits=8,
                    n_layers=2, max_iterations=150, convergence_threshold=1e-10, optimizer="COBYLA",
                    init_strategy="small_random", prepare_hf=True, random_seed=42, noise_models=["depolarizing"],
                    noise_strengths=[0., .01], n_qubits_active=12, n_qubits_final=8, n_parameters=24,
                    casci_energy=circuit.casci, hf_energy=circuit.hf, rotation_gates="RY", parameter_period=2*np.pi)
    for key, value in expected.items():
        require(config[key] == value, f"Incorrect run_config.{key}")
    require(digest(archive) == runtime["source_sha256"] == source_sha, "Scientific source fingerprint differs")
    for pkg, pinned in (("numpy", "1.26.4"), ("pennylane", "0.38.0"), ("pennylane-lightning", "0.38.0"),
                        ("scipy", "1.16.3")):
        require(runtime["packages"][pkg] == pinned, f"Wrong runtime {pkg}")
    argv = runtime["argv"]
    require("--input-snapshot" in argv and str(runtime["platform"]) and str(runtime["python"]),
            "Missing frozen-input invocation or runtime details")


def check_events(events, fit, traces, row, initial, winner, circuit, strength):
    records = [entry for entry in events if entry["fit"] == fit]
    independent = {}
    worst_error = 0.
    for record in records:
        key = record["parameters"]
        theta = np.frombuffer(bytes.fromhex(key), dtype="<f8")
        require(theta.size == 24 and np.isfinite(theta).all(), "Wrong recorded angle vector")
        if key not in independent:
            independent[key] = circuit.energy(circuit.state(theta, strength))
        actual = float.fromhex(record["energy"])
        worst_error = max(worst_error, close(actual, independent[key], 5e-10, f"{fit} recorded energy"))
    queries = [e for e in records if e["callback"] is False]
    callbacks = [e for e in records if e["callback"] is True]
    require(len(queries) + len(callbacks) == len(records) and 0 < len(queries) <= 150,
            f"{fit} invalid record type or COBYLA budget")
    model = "depolarizing" if strength else "none"
    rows = [t for t in traces if (t["noise_model"], float(t["noise_strength"])) == (model, strength)]
    require(len(rows) == len(callbacks) == int(row["n_iterations"]) and len(callbacks) > 0, "Wrong callback count")
    require([int(t["iteration"]) for t in rows] == list(range(len(rows))), "Wrong convergence trace indices")
    close([t["energy"] for t in rows], [float.fromhex(e["energy"]) for e in callbacks], 5.1e-11, "Callback trace")
    require(queries[0]["parameters"] == initial.astype("<f8").tobytes().hex(), f"{fit} seeded start")
    winner_hex = winner.astype("<f8").tobytes().hex()
    require(any(e["parameters"] == winner_hex for e in queries), f"{fit} saved winner absent from queries")
    minimum = min(float.fromhex(e["energy"]) for e in queries)
    close(float(row["energy_noisy"]), minimum, 5.1e-11, f"{fit} best measured objective")
    close(independent[winner_hex], minimum, 5e-10, f"{fit} winning angles")
    return dict(objective_calls=len(queries), callback_calls=len(callbacks), distinct_angles=len(independent),
                maximum_oracle_error=worst_error, initial_energy=independent[queries[0]["parameters"]],
                best_energy=independent[winner_hex])


def audit(directory, source_sha=SOURCE_SHA):
    folder = directory / "results"
    saved = folder / "inputs/gly.npz"
    require(digest(FROZEN) == digest(saved) == INPUT_SHA, "Frozen glycine input changed")
    circuit = GlycineCircuit(saved)
    config, runtime = read(folder / "run_config.json"), read(folder / "runtime.json")
    check_config(config, runtime, folder / "framework-source.zip", source_sha, circuit)
    rows, traces = table(folder / "results.csv"), table(folder / "convergence_traces.csv")
    require([(r["noise_model"], float(r["noise_strength"])) for r in rows] ==
            [("none", 0.), ("depolarizing", 0.), ("depolarizing", .01)], "Wrong noise grid")
    clean, zero, noisy = rows
    require(clean == {**zero, "noise_model": "none"}, "Incorrect zero-noise row")
    zero_trace = [r for r in traces if r["noise_model"] == "depolarizing" and float(r["noise_strength"]) == 0]
    clean_trace = [r for r in traces if r["noise_model"] == "none" and float(r["noise_strength"]) == 0]
    require(clean_trace == [{**r, "noise_model": "none"} for r in zero_trace] and
            all((r["noise_model"], float(r["noise_strength"])) in {("none", 0.), ("depolarizing", 0.), ("depolarizing", .01)}
                for r in traces), "Incorrect zero-noise or unexpected convergence trace")
    params = read(folder / "optimum_parameters.json")
    require(set(params["theta_noisy_star"]) == {"depolarizing@0.01"}, "Unexpected optimized noise vectors")
    initial, clean_star, noisy_star = (np.array(v, dtype=float) for v in
                                      (params["theta_init"], params["theta_clean_star"], params["theta_noisy_star"]["depolarizing@0.01"]))
    require(all(v.shape == (24,) and np.isfinite(v).all() for v in (initial, clean_star, noisy_star)), "Invalid fitted angles")
    require(np.array_equal(initial, np.random.default_rng(42).uniform(-np.pi/8, np.pi/8, 24)), "Incorrect sampled start")
    events = [json.loads(line) for line in (directory / "events.jsonl").read_text().splitlines()]
    require(set(e["fit"] for e in events) == {"clean", "depolarizing"}, "Unexpected recorded fits")
    fits = {fit: check_events(events, fit, traces, row, initial, angles, circuit, p)
            for fit, row, angles, p in (("clean", clean, clean_star, 0.), ("depolarizing", noisy, noisy_star, .01))}
    timing = read(directory / "timings.json")
    require(set(timing["fits"]) == set(fits) and
            0 < sum(v["seconds"] for v in timing["fits"].values()) <= timing["native_cli_seconds"], "Invalid fit timings")
    for name, fit in fits.items():
        native = timing["fits"][name]
        require(native["device"] == ("lightning.qubit" if name == "clean" else "default.mixed")
                and fit["objective_calls"] == native["optimizer"]["optimizer_evaluations"]
                and native["seconds"] > 0 and isinstance(native["optimizer"]["optimizer_success"], bool)
                and native["optimizer"]["optimizer_message"], "Wrong device, termination or evaluation count")
        close(float.fromhex(native["optimal_energy"]), fit["best_energy"], 5e-10, "Native optimizer return")
        fit.update(device=native["device"], optimizer=native["optimizer"], seconds=native["seconds"])
    endpoints = read(folder / "state_endpoints.json")
    require(set(endpoints) == {"none@0", "depolarizing@0", "depolarizing@0.01"}
            and endpoints["none@0"] == endpoints["depolarizing@0"], "Incorrect zero-noise endpoint")
    embedding = parent_isometry(circuit)
    endpoint_report = {}
    for label, p, angles, row, theta_clean in (("none@0", 0., clean_star, clean, clean_star),
                                              ("depolarizing@0.01", .01, noisy_star, noisy, clean_star)):
        drift = angles - theta_clean
        norm = np.linalg.norm(drift)
        drift_values = dict(param_drift_l2=norm, param_drift_rms=np.sqrt(np.mean(drift**2)),
                            param_drift_l2_wrapped=np.linalg.norm(np.angle(np.exp(1j * drift))),
                            param_drift_cosine=np.dot(angles, theta_clean)/(np.linalg.norm(angles)*np.linalg.norm(theta_clean)),
                            clean_displacement_l2=np.linalg.norm(theta_clean-initial), noisy_displacement_l2=np.linalg.norm(angles-initial))
        for key, value in drift_values.items():
            close(float(row[key]), value, 5.1e-11, key)
        close(row["energy_clean_baseline"], fits["clean"]["best_energy"], 1e-10, "Shared baseline")
        results = {}
        require(set(endpoints[label]) == {"noisy", "clean_at_noisy_params"}, "Incorrect endpoint pair")
        for name, strength, energy_key, probability_key in (
            ("noisy", p, "energy_noisy", "wrong_electron_probability_noisy"),
            ("clean_at_noisy_params", 0., "energy_clean_at_noisy_params", "wrong_electron_probability_clean_at_noisy_params"),
        ):
            physical = physical_endpoint(circuit, embedding, circuit.state(angles, strength))
            errors = check_endpoint(physical, endpoints[label][name])
            close(row[energy_key], physical["rescored_energy"], 5e-10, energy_key)
            close(row[probability_key], physical["wrong_electron_probability"], 1e-10, probability_key)
            results[name] = dict(physical=physical, maximum_energy_difference=errors.pop("rescored_energy"),
                                 maximum_sector_difference=max(errors.values()))
        endpoint_report[label] = dict(endpoints=results, drift=drift_values)
    return dict(run=directory.name, input_sha256=INPUT_SHA, source_sha256=source_sha, runtime_platform=runtime["platform"],
                ordered_trace_sha256=digest(directory / "events.jsonl"), native_cli_seconds=timing["native_cli_seconds"],
                ansatz=dict(rotation_gates=24, cnot_gates=16,
                depolarizing_channels=16, layers=2, strength=.01), fits=fits, endpoints=endpoint_report,
                operator_bounds=circuit.bounds.tolist(), parent_hf=circuit.hf, parent_casci=circuit.casci,
                operator_ground_minus_casci=circuit.bounds[0]-circuit.casci,
                mixed_endpoint_tolerance="PennyLane 0.38.0 adds its 1e-14 epsilon to each squared Kraus weight. "
                                         "The ideal trace-preserving NumPy map omits it; the native raw density "
                                         "energy can differ by roughly abs(energy) × 4 × 1e-14 × 16 channels.")


def check_corrupted_artifacts(directory, source_sha):
    """Replay isolated, altered copies of a successful native run and record each rejection."""
    checks = []
    with TemporaryDirectory() as temp:
        copy = Path(temp) / "run"
        shutil.copytree(directory, copy)
        fields = (
            ("results/run_config.json", ["noise_strengths", 1], .02, "Incorrect run_config.noise_strengths"),
            ("results/state_endpoints.json", ["depolarizing@0.01", "noisy", "mean_spin_squared"], -1., "mean_spin_squared"),
            ("results/optimum_parameters.json", ["theta_noisy_star", "depolarizing@0.01", 0], 9., "winner absent"),
        )
        for filename, keys, replacement, expected in fields:
            path = copy / filename
            original = path.read_bytes()
            model = json.loads(original)
            item = model
            for key in keys[:-1]:
                item = item[key]
            item[keys[-1]] = replacement
            path.write_text(json.dumps(model))
            try:
                audit(copy, source_sha)
            except ValueError as exc:
                require(expected in str(exc), f"Unexpected rejection for {filename}: {exc}")
                checks.append(dict(file=filename, field=keys, rejected=str(exc)))
            else:
                raise ValueError(f"Undetected change in {filename}")
            path.write_bytes(original)
        path = copy / "events.jsonl"
        events = [json.loads(line) for line in path.read_text().splitlines()]
        event = next(e for e in events if e["fit"] == "depolarizing" and not e["callback"])
        event["energy"] = (float.fromhex(event["energy"]) + .001).hex()
        path.write_text("".join(json.dumps(e) + "\n" for e in events))
        try:
            audit(copy, source_sha)
        except ValueError as exc:
            require("depolarizing recorded energy" in str(exc), f"Unexpected trace rejection: {exc}")
            checks.append(dict(file="events.jsonl", field=["depolarizing", "first objective", "energy"], rejected=str(exc)))
        else:
            raise ValueError("Undetected change to a noisy objective")
    return checks


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("run", type=Path)
    parser.add_argument("--source-sha", default=SOURCE_SHA)
    parser.add_argument("--check-corruption", action="store_true")
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    report = audit(args.run, args.source_sha)
    if args.check_corruption:
        report["corruption_checks"] = check_corrupted_artifacts(args.run, args.source_sha)
    result = json.dumps(report, indent=2, allow_nan=False) + "\n"
    if args.output:
        args.output.write_text(result)
    else:
        print(result, end="")
