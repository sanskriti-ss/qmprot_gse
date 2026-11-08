# ABOUTME: Compares every scientific artifact of the paired hydrogen initialization and noise grids.
# ABOUTME: Reports verified input, source and output digests alongside full study times.
import argparse
from collections import Counter
from hashlib import sha256
import json
from pathlib import Path

from smoke.efficiency import _digest, _require, _seconds


def solver_files(baseline, candidate):
    inventories = []
    sources = {}
    for version, root in (("baseline", baseline), ("candidate", candidate)):
        inventory = {p.relative_to(root).as_posix(): _digest(p) for p in root.rglob("*") if p.is_file()}
        configs = sorted(p.parent for p in root.rglob("run_config.json"))
        _require(len(configs) == 42, f"Incomplete {version} hydrogen grid")
        source_hashes = set()
        for path in configs:
            runtime = json.loads((path / "runtime.json").read_text())
            source = inventory.pop((path / "framework-source.zip").relative_to(root).as_posix())
            inventory.pop((path / "runtime.json").relative_to(root).as_posix())
            _require(runtime["source_sha256"] == source, f"Incorrect source fingerprint at {path}")
            source_hashes.add(source)
        _require(len(source_hashes) == 1, f"Source changed within the {version} hydrogen grid")
        sources[version] = source_hashes.pop()
        inventories.append(inventory)
    first, second = inventories
    differences = sorted(name for name in first.keys() | second.keys() if first.get(name) != second.get(name))
    _require(not differences, f"Paired scientific artifacts differ: {differences}")
    counts = Counter(name.rsplit("/", 1)[-1] for name in first)
    expected = {"trial_endpoints.json": 30, "state_endpoints.json": 12, "optimum_parameters.json": 12,
                "run_config.json": 42, "matrix.json": 1, "convergence_traces.csv": 42, "hydrogen.npz": 42,
                "optimization_results.csv": 30, "results.csv": 12, "gradient_variance.csv": 2,
                "gradient_variance_by_parameter.csv": 2}
    _require(counts == expected, "Incomplete paired solver outputs")
    return dict(source_sha256=sources, artifacts=len(first),
                artifact_counts=dict(sorted(counts.items())), matrix_sha256=first["matrix.json"],
                inventory_sha256=sha256(json.dumps(first, sort_keys=True).encode()).hexdigest())


def solver_studies(directory):
    return dict(seconds=_seconds(directory, "solver"),
                **solver_files(directory / "baseline-solver", directory / "candidate-solver"))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Verify two native hydrogen study matrices and their timings.")
    parser.add_argument("directory", type=Path)
    print(json.dumps(solver_studies(parser.parse_args().directory), indent=2, allow_nan=False))
