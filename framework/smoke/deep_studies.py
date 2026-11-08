# ABOUTME: Measures analytic noisy readout on frozen hydrogen and glycine with native molecule solvers.
# ABOUTME: Verifies study artifacts and scalar energy bits for paired deep-efficiency benchmarks.
from hashlib import sha256
import json
from pathlib import Path
from statistics import median
from time import perf_counter

import numpy as np

from smoke.efficiency import _digest, _require, _seconds


def readouts(directory):
    from algorithms import get_algorithm
    from core.backend_manager import BackendConfig
    from experiments.provenance import load_problem

    fixtures = Path(__file__).parent / "reproductions/layered_vqe"
    for name in ("hydrogen", "contextual_gly"):
        path = fixtures / f"{name}.npz"
        ham = load_problem(path).hamiltonian
        cases = {}
        for model in ("phaseflip", "amplitude_damping", "depolarizing"):
            backend = BackendConfig.noisy(ham.n_qubits, model, .027)
            solver = get_algorithm("hardware_efficient_vqe")(ham, n_layers=2, backend_config=backend, random_seed=42)
            solver.build_ansatz()
            values, seconds = [], []
            for vector in np.random.default_rng(392).uniform(-.8, .8, (6, solver.n_parameters)):
                start = perf_counter()
                value = solver.cost_function(vector)
                seconds.append(perf_counter() - start)
                values.append(value.hex())
            cases[model] = dict(energies=values, seconds=seconds)
        record = dict(input_sha256=_digest(path), qubits=ham.n_qubits, terms=ham.n_terms, cases=cases)
        (directory / f"{name}.json").write_text(json.dumps(record, indent=2, allow_nan=False) + "\n")


def compare_readouts(directory):
    report = {}
    for name in ("hydrogen", "contextual_gly"):
        a, b = [json.loads((directory / f"{v}-readouts" / f"{name}.json").read_text())
                for v in ("baseline", "candidate")]
        _require({k: v for k, v in a.items() if k != "cases"} == {k: v for k, v in b.items() if k != "cases"}
                 and a["cases"].keys() == b["cases"].keys(), "Noisy scalar cases differ")
        cases = {}
        for model, first in a["cases"].items():
            second = b["cases"][model]
            _require(first["energies"] == second["energies"] and len(first["energies"]) == 6,
                     f"Noisy measurement bits differ for {name}/{model}")
            cases[model] = dict(baseline_seconds=median(first["seconds"][1:]),
                                candidate_seconds=median(second["seconds"][1:]), energies=first["energies"])
        report[name] = dict(input_sha256=a["input_sha256"], qubits=a["qubits"], terms=a["terms"], cases=cases)
    return report


def study_files(directory, workload):
    """Compare every native scientific table, figure and prepared input in a fixed study."""
    inventories, sources = [], {}
    for version in ("baseline", "candidate"):
        folder = directory / f"{version}-{workload}"
        sources[version] = _digest(folder / "framework-source.zip")
        _require(sources[version] == json.loads((folder / "runtime.json").read_text())["source_sha256"],
                 f"Invalid source for {version}/{workload}")
        inventories.append({str(p.relative_to(folder)): _digest(p) for p in folder.rglob("*")
                            if p.is_file() and p.name not in ("runtime.json", "framework-source.zip")})
    required = ({"optimum_parameters.json", "results.csv", "state_endpoints.json"} if workload == "noise"
                else {"gradient_variance_by_parameter.csv", "optimization_results.csv", "trial_endpoints.json"})
    _require(required.issubset(inventories[0]), f"Missing {workload} outputs")
    differences = [k for k in inventories[0].keys() | inventories[1].keys()
                   if inventories[0].get(k) != inventories[1].get(k)]
    _require(not differences, f"{workload} scientific outputs differ: {differences}")
    result = dict(files=len(inventories[0]), artifact_sha256=inventories[0],
                  source_sha256=sources, seconds=_seconds(directory, workload))
    if workload == "noise":
        streams = [directory / f"{v}-noise-events.jsonl" for v in ("baseline", "candidate")]
        raw = streams[0].read_bytes()
        _require(bool(raw) and raw == streams[1].read_bytes(), "Noise optimizer streams differ")
        result.update(events=len(raw.splitlines()), events_sha256=sha256(raw).hexdigest())
    return result


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Measure frozen analytic density-matrix readouts.")
    parser.add_argument("directory", type=Path)
    directory = parser.parse_args().directory
    directory.mkdir(exist_ok=False)
    readouts(directory)
