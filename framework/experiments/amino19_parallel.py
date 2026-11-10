# ABOUTME: Schedules independent 14-qubit amino-acid fits and resumes from completed results.
"""Run the pinned 14-qubit, non-HVA comparison with one process per active molecule."""

import argparse
from collections import deque
from datetime import datetime
import fcntl
from hashlib import sha256
from importlib.metadata import version
import json
import os
from pathlib import Path
import signal
import subprocess
import sys
import time
import traceback


FRAMEWORK = Path(__file__).resolve().parents[1]
GEOMETRIES = FRAMEWORK / "smoke/geometries"
MOLECULES = "ala arg asp cys gln glu gly his ile leu lys met phe pro ser thr trp tyr val".split()
ALGORITHMS = (
    "pennylane_vqe", "adapt_vqe", "qubit_adapt_vqe", "hardware_efficient_vqe",
    "qaoa_inspired_vqe", "iqcc_inspired_vqe", "iqcc_true_vqe", "cb_vqe", "ucc_vqe", "nn_ae_vqe",
)
FIT = dict(max_active_orbitals=7, expected_n_qubits=14, truncation_mode="active_space",
           active_space_basis="sto-3g", contextual_subspace=False, backend_type="statevector",
           n_shots=0, optimizer="COBYLA", n_layers=2, max_iterations=200, random_seed=42)
ENV = dict(OMP_NUM_THREADS="1", OPENBLAS_NUM_THREADS="1", MKL_NUM_THREADS="1",
           MPLBACKEND="Agg", LOG_FILE="")


def write_json(path, data):
    """Publish a checkpoint only after the whole result has been written."""
    temporary = path.with_suffix(".tmp")
    with temporary.open("w", encoding="utf-8") as handle:
        json.dump(data, handle, indent=2)
        handle.write("\n")
        handle.flush()
        os.fsync(handle.fileno())
    temporary.replace(path)


def configuration(molecules, algorithms):
    from config import CONVERGENCE_THRESHOLD
    from experiments.provenance import framework_source

    return dict(
        format_version=1, molecules=molecules, algorithms=list(algorithms),
        fit=dict(FIT, convergence_threshold=CONVERGENCE_THRESHOLD), threads=1,
        geometry_sha256={mol: sha256((GEOMETRIES / mol / f"{mol}.h5").read_bytes()).hexdigest()
                         for mol in molecules},
        source_sha256=sha256(framework_source()).hexdigest(),
        main_sha256=sha256((FRAMEWORK / "main.py").read_bytes()).hexdigest(),
        python=sys.version,
        packages={name: version(name) for name in ("numpy", "scipy", "pyscf", "h5py", "openfermion",
                                                    "pennylane", "pennylane-lightning")},
    )


def checkpoint(root, molecule, algorithm):
    return root / "molecules" / molecule / f"{algorithm}.json"


def read_checkpoint(path, molecule, algorithm):
    data = json.loads(path.read_text(encoding="utf-8"))
    identity = data.get("molecule_abbrev"), data.get("algorithm_name"), data.get("n_qubits")
    if identity != (molecule, algorithm, FIT["expected_n_qubits"]):
        raise ValueError(f"Unexpected result identity in {path}: {identity}")
    return data


def worker(root, molecule):
    manifest = json.loads((root / "run_config.json").read_text(encoding="utf-8"))
    if configuration(manifest["molecules"], manifest["algorithms"]) != manifest:
        raise ValueError("Run inputs changed before worker startup")

    from main import VQEFramework

    directory = root / "molecules" / molecule
    runner = VQEFramework(hamiltonians_dir=GEOMETRIES, results_dir=directory, plots_dir=directory)
    failed = False
    for algorithm in manifest["algorithms"]:
        path = checkpoint(root, molecule, algorithm)
        if path.exists():
            continue
        try:
            data = runner.run_single(molecule, algorithm, **manifest["fit"])
            if (data["molecule_abbrev"], data["algorithm_name"]) != (molecule, algorithm):
                raise ValueError(f"Result did not identify {molecule}/{algorithm}")
            write_json(path, data)
        except Exception:
            traceback.print_exc()
            failed = True
    return int(failed)


def schedule(root, molecules, workers, lock_fd):
    pending = deque(molecules)
    active = {}
    try:
        while pending or active:
            while pending and len(active) < workers:
                molecule = pending.popleft()
                directory = root / "molecules" / molecule
                directory.mkdir(parents=True, exist_ok=True)
                command = [sys.executable, "-u", "-m", "experiments.amino19_parallel",
                           "--worker", str(root), molecule]
                with (directory / "worker.log").open("a", encoding="utf-8") as log:
                    # Keep the run lock alive until the last worker exits, even if the parent dies.
                    active[molecule] = subprocess.Popen(command, stdout=log, stderr=subprocess.STDOUT,
                                                         cwd=FRAMEWORK, start_new_session=True,
                                                         pass_fds=(lock_fd,))
                print(f"Started {molecule}", flush=True)
            for molecule, process in list(active.items()):
                status = process.poll()
                if status is not None:
                    del active[molecule]
                    if status:
                        print(f"{molecule} exited with status {status}; see molecules/{molecule}/worker.log",
                              flush=True)
                    else:
                        print(f"Finished {molecule}", flush=True)
            if active:
                time.sleep(1)
    finally:
        for process in active.values():
            process.terminate()
        for process in active.values():
            try:
                process.wait(timeout=5)
            except subprocess.TimeoutExpired:
                process.kill()
                process.wait()


def collect(root, molecules, algorithms, plots):
    from main import VQEFramework

    runner = VQEFramework(hamiltonians_dir=GEOMETRIES, results_dir=root / "results", plots_dir=root / "plots")
    for molecule in molecules:
        for algorithm in algorithms:
            path = checkpoint(root, molecule, algorithm)
            if path.exists():
                runner.results_manager.add_results(runner.results_manager.load_results(path))
    count = len(runner.results_manager.results)
    expected = len(molecules) * len(algorithms)
    json_summary = runner.results_manager.save_all_results("all_results.tmp")
    json_summary.replace(json_summary.with_suffix(".json"))
    csv_summary = runner.results_manager.save_to_csv("results_summary.tmp")
    csv_summary.replace(csv_summary.with_suffix(".csv"))
    print(f"Saved {count}/{expected} fits under {root}", flush=True)
    if count != expected:
        print("Use --resume with this directory to retry missing fits.", file=sys.stderr)
        return 1
    if plots:
        runner.generate_plots()
    return 0


def main():
    os.environ.update(ENV)
    parser = argparse.ArgumentParser(description=__doc__)
    destination = parser.add_mutually_exclusive_group()
    destination.add_argument("--output-dir", type=Path, help="Directory for a new run")
    destination.add_argument("--resume", type=Path, help="Resume a run created by this launcher")
    parser.add_argument("--workers", type=int, default=2, help="Concurrent molecules (default: 2)")
    parser.add_argument("--plots", action="store_true", help="Generate the standard framework plots after all fits finish")
    parser.add_argument("--dry-run", action="store_true", help="Show pending fits without starting workers")
    parser.add_argument("--molecules", choices=MOLECULES, nargs="+", help="Select a subset for a new run")
    parser.add_argument("--algorithms", choices=ALGORITHMS, nargs="+", help="Select a subset for a new run")
    parser.add_argument("--worker", nargs=2, help=argparse.SUPPRESS)
    args = parser.parse_args()
    if args.worker:
        return worker(Path(args.worker[0]), args.worker[1])
    if args.workers < 1:
        parser.error("--workers must be positive")
    if args.resume and (args.molecules or args.algorithms):
        parser.error("--resume uses the molecule and algorithm lists saved in the run")

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    root = (args.resume or args.output_dir or FRAMEWORK / f"results/amino19_14q_parallel_{timestamp}").resolve()
    config_path = root / "run_config.json"
    if args.resume:
        if not config_path.is_file():
            parser.error(f"No parallel run configuration at {config_path}")
        saved = json.loads(config_path.read_text(encoding="utf-8"))
        molecules, algorithms = saved["molecules"], saved["algorithms"]
    else:
        molecules = list(dict.fromkeys(args.molecules or MOLECULES))
        algorithms = list(dict.fromkeys(args.algorithms or ALGORITHMS))
    if not set(molecules) <= set(MOLECULES) or not set(algorithms) <= set(ALGORITHMS):
        parser.error("Unknown molecule or algorithm in saved configuration")
    manifest = configuration(molecules, algorithms)
    if args.resume and saved != manifest:
        changed = sorted(key for key in saved.keys() | manifest.keys() if saved.get(key) != manifest.get(key))
        parser.error(f"Resume inputs differ from the saved run: {', '.join(changed)}")

    def pending_molecules():
        pending = []
        count = 0
        for molecule in molecules:
            for algorithm in algorithms:
                path = checkpoint(root, molecule, algorithm)
                if path.exists():
                    read_checkpoint(path, molecule, algorithm)
                else:
                    pending.append(molecule)
                    count += 1
        print(f"{count}/{len(molecules) * len(algorithms)} fits pending, up to {args.workers} workers\n{root}",
              flush=True)
        return list(dict.fromkeys(pending))

    if args.dry_run:
        pending_molecules()
        return 0
    if not args.resume:
        root.mkdir(parents=True, exist_ok=False)
    with (root / ".run.lock").open("a") as lock:
        try:
            fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError:
            parser.error(f"Another launcher or worker is using {root}")
        if not args.resume:
            from experiments.provenance import record_runtime

            record_runtime(root)
            write_json(config_path, manifest)
        schedule(root, pending_molecules(), args.workers, lock.fileno())
        return collect(root, molecules, algorithms, args.plots)


def interrupt(*_):
    raise KeyboardInterrupt


if __name__ == "__main__":
    signal.signal(signal.SIGTERM, interrupt)
    try:
        sys.exit(main())
    except KeyboardInterrupt:
        print("Interrupted. Completed fit checkpoints are available for --resume.", file=sys.stderr)
        sys.exit(130)
