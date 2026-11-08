# ABOUTME: Sweeps ordered prefixes of each molecular Hamiltonian across the selected VQE solvers.
# ABOUTME: Compares each solver with the full-molecule reference and the exact prefix ground energy.
"""
Experiment 5 -- Hamiltonian term-count sweep
===========================================

Question
--------
How does the achieved energy change when the Hamiltonian itself is revealed
incrementally, i.e. when we run the same algorithm on the first term, first two
terms, first three terms, and so on?

This experiment treats the Hamiltonian term count as the "parameter count"
on the x-axis.  For each molecule we build prefix Hamiltonians from the first
N terms, run every implemented VQE algorithm, and compare the result against
the full-molecule CASCI reference energy.

What we measure
---------------
For each molecule, algorithm, term count and trial seed we record:

* ``random_seed``            -- the same trial seed at every molecule, algorithm and prefix.
* ``wrong_electron_probability`` -- probability outside the specified physical electron sector, when known.
* ``n_parameters``           -- number of Hamiltonian terms kept in the prefix.
* ``algorithm_n_parameters``  -- number of variational parameters used by the
  VQE algorithm itself.
* ``energy``                 -- final energy returned by the algorithm.
* ``error_vs_full_reference`` -- energy minus the full-molecule CASCI energy.
* ``prefix_reference_energy`` -- exact ground-state energy of the prefix
  Hamiltonian used in that run.
* ``error_vs_prefix_reference`` -- energy minus the prefix-Hamiltonian
  ground-state energy.
* ``n_cost_evals``           -- logical molecular objective calls. Adaptive screening and angle-observable costs are separate.
* ``parameter_update_policy`` -- optimizer or analytic angle selection specified by the solver.
* ``runtime_seconds``        -- fit wall time. Endpoint replay is timed outside the fit.

Summary means weight each contributing molecule equally after averaging its available trial seeds.
The population standard deviation is taken across those molecule means. Completion counts and the
individual contributing seeds are included for each point.

Outputs
-------
``framework/experiments/results/experiment5_term_count_sweep/<timestamp>_batch/``

* ``run_config.json``
* ``batch_all_results.csv``
* ``summary_by_algorithm_term_count.csv``
* ``mean_absolute_error_vs_term_count.png``
* ``batch_failures.json``     -- only if some runs fail

Run as a script
---------------
::

    python -m experiments.experiment5_term_count_sweep \
        --batch-amino-acids --term-counts 1 2 3 5 10 20 50
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import logging
import os
import sys
import time
from collections import defaultdict
from dataclasses import asdict, dataclass, field, fields
from datetime import datetime
from itertools import product
from operator import index
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np

_FW_DIR = Path(__file__).resolve().parent.parent
if str(_FW_DIR) not in sys.path:
    sys.path.insert(0, str(_FW_DIR))

from algorithms import get_algorithm, list_algorithms  # noqa: E402
from core.base_vqe import VQEResult  # noqa: E402
from core.hamiltonian_loader import Molecule, QubitHamiltonian  # noqa: E402
from experiments.provenance import load_problem, record_runtime, save_problem
from experiments._common import (  # noqa: E402
    CostEvalCounter,
    LoadedHamiltonian,
    load_small_hamiltonian,
    make_backend_config,
    validate_study_axes,
)
from experiments.noise_resilience import DATASETS2_AMINO_ACIDS  # noqa: E402

logger = logging.getLogger(__name__)


DEFAULT_MOLECULES: Tuple[str, ...] = DATASETS2_AMINO_ACIDS
DEFAULT_ALGORITHMS: Tuple[str, ...] = tuple(dict.fromkeys(list_algorithms()))
DEFAULT_TERM_COUNTS: Tuple[int, ...] = (1, 2, 3, 5, 10, 20, 50)
DEFAULT_N_LAYERS: int = 2
DEFAULT_MAX_OPERATORS: int = 20
DEFAULT_GRADIENT_THRESHOLD: float = 1e-8


# ──────────────────────────────────────────────────────────────────────────
# Result containers
# ──────────────────────────────────────────────────────────────────────────

@dataclass
class TermCountRow:
    molecule: str
    algorithm: str
    n_parameters: int                  # Hamiltonian prefix length
    algorithm_n_parameters: int        # variational parameter count
    n_qubits: int
    n_terms_available: int
    energy: float
    full_reference_energy: float
    prefix_reference_energy: float
    error_vs_full_reference: float
    abs_error_vs_full_reference: float
    error_vs_prefix_reference: float
    abs_error_vs_prefix_reference: float
    n_iterations: int
    n_cost_evals: int
    runtime_seconds: float
    converged: bool
    status: str = "ok"
    notes: str = ""
    random_seed: int = 42
    parameter_update_policy: str = ""
    wrong_electron_probability: Optional[float] = None


@dataclass
class TermCountConfig:
    molecules: Tuple[str, ...]
    algorithms: Tuple[str, ...]
    term_counts: Tuple[int, ...]
    basis: str
    cs_target_qubits: Optional[int]
    n_layers: int
    max_operators: int
    gradient_threshold: float
    optimizer: str
    max_iterations: int
    convergence_threshold: float
    trial_seeds: Tuple[int, ...]
    models: Dict[str, Dict[str, int]] = field(default_factory=dict)


# ──────────────────────────────────────────────────────────────────────────
# Hamiltonian helpers
# ──────────────────────────────────────────────────────────────────────────

class _HamiltonianPrefixes:
    """Build successive prefixes for one molecule, retaining only its latest sparse sum."""

    def __init__(self, full_hamiltonian: QubitHamiltonian):
        self.full_hamiltonian = full_hamiltonian
        self.term_count = 0
        self.matrix = None

    def at(self, n_terms: int) -> QubitHamiltonian:
        full_hamiltonian = self.full_hamiltonian
        if n_terms < 1:
            raise ValueError("n_terms must be at least 1")
        n_terms = min(int(n_terms), int(full_hamiltonian.n_terms))
        if n_terms < self.term_count:
            raise ValueError("Hamiltonian prefixes must be requested in increasing order")
        n_qubits = full_hamiltonian.n_qubits
        if n_qubits > 18:
            raise ValueError(f"Exact prefix references support at most 18 qubits (requested {n_qubits})")

        added = slice(self.term_count, n_terms)
        self.matrix = full_hamiltonian._extend_sparse_matrix(
            self.matrix, full_hamiltonian.coefficients[added], full_hamiltonian.pauli_strings[added], n_qubits
        )
        self.term_count = n_terms
        energy = full_hamiltonian._sparse_ground_state_energy(self.matrix, n_qubits)
        if not np.isfinite(energy):
            raise ValueError(f"Nonfinite reference energy for the {n_terms}-term prefix")
        coeffs = np.array(full_hamiltonian.coefficients[:n_terms], copy=True)
        pauli_strings = list(full_hamiltonian.pauli_strings[:n_terms])

        prefix_molecule = Molecule(
            abbreviation=full_hamiltonian.molecule.abbreviation,
            name=full_hamiltonian.molecule.name,
            n_qubits=full_hamiltonian.n_qubits,
            n_coefficients=n_terms,
            reference_energy=energy,
            truncated_ground_state_energy=energy,
            hamiltonian_file=full_hamiltonian.molecule.hamiltonian_file,
            n_electrons=full_hamiltonian.molecule.n_electrons,
            n_orbitals=full_hamiltonian.molecule.n_orbitals,
            charge=full_hamiltonian.molecule.charge,
            spin=full_hamiltonian.molecule.spin,
            basis=full_hamiltonian.molecule.basis,
            coordinates=full_hamiltonian.molecule.coordinates,
            molecular_formula=full_hamiltonian.molecule.molecular_formula,
        )

        prefix = QubitHamiltonian(
            molecule=prefix_molecule,
            coefficients=coeffs,
            pauli_strings=pauli_strings,
            n_qubits=full_hamiltonian.n_qubits,
            n_terms=n_terms,
            cs_initial_state=full_hamiltonian.cs_initial_state,
        )
        prefix._retain_ordered_matrix(self.matrix, tuple(pauli_strings), coeffs.copy())
        return prefix


def _prefix_hamiltonian(full_hamiltonian: QubitHamiltonian, n_terms: int) -> QubitHamiltonian:
    """Build a standalone prefix including its ground-state reference."""
    return _HamiltonianPrefixes(full_hamiltonian).at(n_terms)


# ──────────────────────────────────────────────────────────────────────────
# Algorithm execution
# ──────────────────────────────────────────────────────────────────────────


def _run_algorithm_on_hamiltonian(
    hamiltonian: QubitHamiltonian,
    *,
    algorithm: str,
    n_layers: int,
    max_operators: int,
    gradient_threshold: float,
    optimizer: str,
    max_iterations: int,
    convergence_threshold: float,
    random_seed: int,
    loaded: Optional[LoadedHamiltonian] = None,
) -> Tuple[VQEResult, int, float]:
    """Instantiate and run one algorithm on one Hamiltonian."""
    AlgorithmCls = get_algorithm(algorithm)
    backend_config = make_backend_config(hamiltonian.n_qubits)

    vqe = AlgorithmCls(
        hamiltonian,
        n_layers=n_layers,
        max_operators=max_operators,
        gradient_threshold=gradient_threshold,
        optimizer=optimizer,
        max_iterations=max_iterations,
        convergence_threshold=convergence_threshold,
        random_seed=random_seed,
        backend_config=backend_config,
    )

    counter = CostEvalCounter()
    counter.attach(vqe)
    t0 = time.time()
    try:
        result = vqe.run()
    finally:
        counter.detach(vqe)
    runtime = time.time() - t0
    if loaded is not None:
        from experiments.endpoint_sector import summarize_endpoint

        sector = summarize_endpoint(loaded, vqe, result.optimal_parameters)
        if not np.isclose(sector["rescored_energy"], result.calculated_energy, rtol=0., atol=1e-8):
            raise ValueError(f"{algorithm} endpoint disagrees with the returned energy: "
                             f'{sector["rescored_energy"]} vs {result.calculated_energy}')
        result.metadata["endpoint_sector"] = sector
    return result, int(counter.count), runtime


# ──────────────────────────────────────────────────────────────────────────
# Public API
# ──────────────────────────────────────────────────────────────────────────


def run_term_count_sweep(
    molecules: Sequence[str] = DEFAULT_MOLECULES,
    algorithms: Sequence[str] = DEFAULT_ALGORITHMS,
    term_counts: Sequence[int] = DEFAULT_TERM_COUNTS,
    basis: str = "sto-3g",
    cs_target_qubits: Optional[int] = 4,
    n_layers: int = DEFAULT_N_LAYERS,
    max_operators: int = DEFAULT_MAX_OPERATORS,
    gradient_threshold: float = DEFAULT_GRADIENT_THRESHOLD,
    optimizer: str = "COBYLA",
    max_iterations: int = 200,
    convergence_threshold: float = 1e-10,
    random_seed: Optional[int] = None,
    output_dir: Optional[Path] = None,
    save_plots: bool = True,
    prepared_inputs: Optional[Path] = None,
    trial_seeds: Optional[Sequence[int]] = None,
) -> Dict[str, object]:
    """Run the batch experiment across many molecules and algorithms."""
    if random_seed is not None and trial_seeds is not None:
        raise ValueError("Choose a single random_seed or a sequence of trial_seeds")
    if trial_seeds is None:
        trial_seeds = (42 if random_seed is None else random_seed,)
    seeds = tuple(index(seed) for seed in trial_seeds)
    term_counts_sorted = tuple(sorted(index(k) for k in term_counts))
    molecule_list = tuple(m.strip().lower() for m in molecules)
    algorithm_list = tuple(algorithms)
    validate_study_axes(seeds=seeds, term_counts=term_counts_sorted,
                        molecules=molecule_list, algorithms=algorithm_list)
    if any(seed < 0 or seed >= 2**32 for seed in seeds):
        raise ValueError("Trial seeds must be integers from 0 to 2**32 - 1")
    if term_counts_sorted[0] < 1:
        raise ValueError("Each term count must be positive")

    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    if output_dir is None:
        output_dir = (
            _FW_DIR / "experiments" / "results" / "experiment5_term_count_sweep"
            / f"{ts}_batch"
        )
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    batch_csv = output_dir / "batch_all_results.csv"
    summary_csv = output_dir / "summary_by_algorithm_term_count.csv"
    progress_json = output_dir / "progress.json"
    failures_json = output_dir / "batch_failures.json"
    config_json = output_dir / "run_config.json"

    total_rows_estimate = len(molecule_list) * len(term_counts_sorted) * len(algorithm_list) * len(seeds)

    cfg = TermCountConfig(
        molecules=molecule_list,
        algorithms=algorithm_list,
        term_counts=term_counts_sorted,
        basis=basis,
        cs_target_qubits=cs_target_qubits,
        n_layers=n_layers,
        max_operators=max_operators,
        gradient_threshold=gradient_threshold,
        optimizer=optimizer,
        max_iterations=max_iterations,
        convergence_threshold=convergence_threshold,
        trial_seeds=seeds,
    )

    rows: List[TermCountRow] = []
    failures: List[Dict[str, object]] = []
    skipped = []

    if batch_csv.exists():
        raise FileExistsError(f"Prefix results already exist in {output_dir}; use a fresh directory")
    with open(config_json, "x") as f:
        json.dump(asdict(cfg), f, indent=2)

    record_runtime(output_dir)

    def _write_progress_tracker(*, completed: int, status: str, current: Optional[Dict[str, object]] = None) -> None:
        tracker = {
            "timestamp": datetime.now().isoformat(timespec="seconds"),
            "completed_rows": completed,
            "successful_rows": completed - len(failures) - len(skipped),
            "failed_cases": len(failures),
            "skipped_cases": len(skipped),
            "estimated_total_rows": total_rows_estimate,
            "remaining_rows_estimate": max(total_rows_estimate - completed, 0),
            "status": status,
            "output_dir": str(output_dir),
            "batch_csv": str(batch_csv),
            "summary_csv": str(summary_csv),
            "failures_json": str(failures_json),
        }
        if current is not None:
            tracker["current"] = current
        with open(progress_json, "w") as f:
            json.dump(tracker, f, indent=2)
            f.flush()
            os.fsync(f.fileno())

    try:
        from tqdm import tqdm as _tqdm
    except ImportError:
        _tqdm = None  # type: ignore[misc,assignment]

    mol_iter: Iterable[str] = molecule_list
    if _tqdm is not None:
        mol_iter = _tqdm(list(molecule_list), desc="experiment5 batch")

    completed_rows = 0
    with open(batch_csv, "x", newline="", buffering=1) as batch_file, (output_dir / "cases.jsonl").open("x", buffering=1) as cases:
        writer = csv.DictWriter(batch_file, fieldnames=[f.name for f in fields(TermCountRow)])
        writer.writeheader()
        batch_file.flush()
        os.fsync(batch_file.fileno())
        _write_progress_tracker(completed=0, status="starting")

        try:
            for mol in mol_iter:
                try:
                    if prepared_inputs is None:
                        loaded = load_small_hamiltonian(molecule=mol, basis=basis, cs_target_qubits=cs_target_qubits)
                    else:
                        loaded = load_problem(Path(prepared_inputs) / f"{mol}.npz")
                        if (loaded.molecule != mol or loaded.basis != basis or
                                loaded.preparation["cs_target_qubits"] != cs_target_qubits):
                            raise ValueError(f"Prepared model does not match the requested settings for {mol}")
                except Exception as exc:
                    logger.exception("Failed to load molecule %s: %s", mol, exc)
                    failures.extend({"molecule": mol, "algorithm": algo, "term_count": str(k),
                                     "random_seed": seed, "error": str(exc)}
                                    for k, algo, seed in product(term_counts_sorted, algorithm_list, seeds))
                    completed_rows += len(term_counts_sorted) * len(algorithm_list) * len(seeds)
                    _write_progress_tracker(
                        completed=completed_rows,
                        status="load_failed",
                        current={"molecule": mol},
                    )
                    continue

                snapshot = save_problem(loaded, output_dir / "inputs")
                cfg.models[mol] = dict(n_qubits_active=int(loaded.n_qubits_active),
                                       n_qubits_final=int(loaded.n_qubits_final),
                                       n_terms_available=int(loaded.hamiltonian.n_terms))
                input_sha256 = hashlib.sha256(snapshot.read_bytes()).hexdigest()
                prefixes = _HamiltonianPrefixes(loaded.hamiltonian)
                for term_count in term_counts_sorted:
                    if term_count > loaded.hamiltonian.n_terms:
                        logger.warning(
                            "Skipping %s at n_terms=%d because only %d terms are available",
                            mol, term_count, loaded.hamiltonian.n_terms,
                        )
                        skipped.extend({"molecule": mol, "algorithm": algo, "term_count": term_count,
                                        "n_terms_available": loaded.hamiltonian.n_terms, "random_seed": seed}
                                       for algo, seed in product(algorithm_list, seeds))
                        completed_rows += len(algorithm_list) * len(seeds)
                        _write_progress_tracker(
                            completed=completed_rows,
                            status="skipped_term_count",
                            current={"molecule": mol, "n_parameters": term_count},
                        )
                        continue

                    try:
                        prefix_ham = prefixes.at(term_count)
                    except Exception as exc:
                        _write_progress_tracker(completed=completed_rows, status="prefix_reference_failed",
                                                current={"molecule": mol, "n_parameters": term_count})
                        raise RuntimeError(f"Exact reference failed for {mol} at {term_count} terms: {exc}") from exc
                    prefix_reference = float(prefix_ham.molecule.reference_energy)
                    full_reference = float(loaded.casci_energy) if loaded.casci_energy is not None else float("nan")

                    for algo, run_seed in product(algorithm_list, seeds):
                        current_meta = {
                            "molecule": mol,
                            "algorithm": algo,
                            "n_parameters": term_count,
                            "random_seed": run_seed,
                        }
                        try:
                            result, n_cost_evals, runtime = _run_algorithm_on_hamiltonian(
                                prefix_ham,
                                algorithm=algo,
                                n_layers=n_layers,
                                max_operators=max_operators,
                                gradient_threshold=gradient_threshold,
                                optimizer=optimizer,
                                max_iterations=max_iterations,
                                convergence_threshold=convergence_threshold,
                                random_seed=run_seed,
                                loaded=loaded,
                            )
                            sector = result.metadata.pop("endpoint_sector")
                            if algo == "iqcc_true_vqe":
                                update_policy = "analytic Pauli angle selection"
                            else:
                                update_policy = result.metadata.get("inner_optimizer", result.metadata.get("optimizer"))
                            energy = float(result.calculated_energy)
                            error_vs_full = energy - full_reference
                            error_vs_prefix = energy - prefix_reference
                            row = TermCountRow(
                                molecule=mol,
                                algorithm=algo,
                                random_seed=run_seed,
                                n_parameters=term_count,
                                algorithm_n_parameters=int(result.n_parameters),
                                n_qubits=int(result.n_qubits),
                                n_terms_available=int(loaded.hamiltonian.n_terms),
                                energy=energy,
                                full_reference_energy=full_reference,
                                prefix_reference_energy=prefix_reference,
                                error_vs_full_reference=error_vs_full,
                                abs_error_vs_full_reference=abs(error_vs_full),
                                error_vs_prefix_reference=error_vs_prefix,
                                abs_error_vs_prefix_reference=abs(error_vs_prefix),
                                n_iterations=int(result.n_iterations),
                                n_cost_evals=int(n_cost_evals),
                                runtime_seconds=float(runtime),
                                converged=bool(result.converged),
                                parameter_update_policy=update_policy,
                                wrong_electron_probability=sector["wrong_electron_probability"],
                                status="ok",
                                notes=(result.metadata or {}).get("excitation_implementation", ""),
                            )
                            cases.write(json.dumps(dict(
                                molecule=mol, prefix_terms=term_count, algorithm=algo, random_seed=run_seed,
                                input_sha256=input_sha256, endpoint_sector=sector,
                                optimizer_requested=optimizer, parameter_update_policy=update_policy,
                                backend=result.backend_type, energy=energy, prefix_reference=prefix_reference,
                                full_casci=full_reference, solver_name=result.algorithm_name,
                                parameters=np.asarray(result.optimal_parameters).tolist() if result.optimal_parameters is not None else None,
                                history=[float(e) for e in result.convergence_history], cost_evaluations=n_cost_evals,
                                notes=row.notes, metadata=result.metadata,
                            ), default=lambda obj: obj.tolist() if isinstance(obj, (np.generic, np.ndarray)) else str(obj)) + "\n")
                        except Exception as exc:
                            logger.exception(
                                "Experiment 5 failed for molecule=%s algorithm=%s term_count=%d",
                                mol, algo, term_count,
                            )
                            failures.append({
                                "molecule": mol,
                                "algorithm": algo,
                                "term_count": str(term_count),
                                "random_seed": run_seed,
                                "error": str(exc),
                            })
                            row = TermCountRow(
                                molecule=mol,
                                algorithm=algo,
                                random_seed=run_seed,
                                n_parameters=term_count,
                                algorithm_n_parameters=0,
                                n_qubits=int(loaded.hamiltonian.n_qubits),
                                n_terms_available=int(loaded.hamiltonian.n_terms),
                                energy=float("nan"),
                                full_reference_energy=full_reference,
                                prefix_reference_energy=prefix_reference,
                                error_vs_full_reference=float("nan"),
                                abs_error_vs_full_reference=float("nan"),
                                error_vs_prefix_reference=float("nan"),
                                abs_error_vs_prefix_reference=float("nan"),
                                n_iterations=0,
                                n_cost_evals=0,
                                runtime_seconds=0.0,
                                converged=False,
                                status="failed",
                                notes=str(exc),
                            )
                        rows.append(row)
                        writer.writerow(_csv_row(row))
                        batch_file.flush()
                        os.fsync(batch_file.fileno())
                        completed_rows += 1
                        _write_progress_tracker(
                            completed=completed_rows,
                            status="running",
                            current=current_meta,
                        )

        finally:
            batch_file.flush()
            os.fsync(batch_file.fileno())

    config_json.write_text(json.dumps(asdict(cfg), indent=2) + "\n")
    _write_summary_csv(summary_csv, rows)
    if failures:
        with open(failures_json, "w") as f:
            json.dump(failures, f, indent=2)

    if skipped:
        (output_dir / "skipped_cases.json").write_text(json.dumps(skipped, indent=2) + "\n")
    _write_progress_tracker(completed=completed_rows, status="finished" if rows and not failures else "failed")

    if save_plots:
        _plot_aggregate_trend(summary_csv, output_dir)
        _plot_aggregate_trend_filtered(summary_csv, output_dir)

    logger.info("Wrote Experiment 5 outputs to %s", output_dir)
    return {
        "config": cfg,
        "rows": rows,
        "output_dir": output_dir,
        "batch_csv": batch_csv,
        "summary_csv": summary_csv,
        "failures": failures,
        "n_ok": sum(1 for row in rows if row.status == "ok"),
        "n_fail": len(failures),
        "n_skipped": len(skipped),
    }


# ──────────────────────────────────────────────────────────────────────────
# Output helpers
# ──────────────────────────────────────────────────────────────────────────


def _csv_row(row: TermCountRow) -> dict:
    values = asdict(row)
    for key, value in values.items():
        if isinstance(value, float):
            precision = 4 if key == "runtime_seconds" else 10
            values[key] = f"{value:.{precision}f}" if np.isfinite(value) else ""
    values["converged"] = int(row.converged)
    return values


def _write_long_csv(path: Path, rows: Sequence[TermCountRow]) -> None:
    with open(path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=[f.name for f in fields(TermCountRow)])
        writer.writeheader()
        writer.writerows(_csv_row(row) for row in rows)


def _write_summary_csv(path: Path, rows: Sequence[TermCountRow]) -> None:
    grouped: Dict[Tuple[str, int], List[TermCountRow]] = defaultdict(list)
    for row in rows:
        if row.status != "ok" or not np.isfinite(row.abs_error_vs_full_reference):
            continue
        grouped[(row.algorithm, row.n_parameters)].append(row)

    summary_rows = []
    for (algorithm, n_parameters), group in sorted(grouped.items()):
        by_molecule = defaultdict(list)
        for row in group:
            by_molecule[row.molecule].append(row)
        observables = ("abs_error_vs_full_reference", "error_vs_full_reference",
                       "abs_error_vs_prefix_reference", "error_vs_prefix_reference",
                       "runtime_seconds", "n_cost_evals", "n_iterations")
        seed_map = {mol: sorted(r.random_seed for r in trials) for mol, trials in sorted(by_molecule.items())}
        summary = {
            "algorithm": algorithm, "n_parameters": n_parameters, "n_samples": len(group),
            "n_molecules": len(by_molecule), "molecules": json.dumps(sorted(by_molecule)),
            "trial_seeds_by_molecule": json.dumps(seed_map),
        }
        for name in observables:
            means = [np.mean([getattr(r, name) for r in trials]) for trials in by_molecule.values()]
            label = "cost_evals" if name == "n_cost_evals" else name
            summary[f"mean_{label}"] = float(np.mean(means))
            if "error" in name:
                summary[f"std_{label}"] = float(np.std(means))
        summary_rows.append(summary)

    with open(path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=[
            "algorithm",
            "n_parameters",
            "n_samples",
            "n_molecules",
            "molecules",
            "trial_seeds_by_molecule",
            "mean_abs_error_vs_full_reference",
            "std_abs_error_vs_full_reference",
            "mean_error_vs_full_reference",
            "std_error_vs_full_reference",
            "mean_abs_error_vs_prefix_reference",
            "std_abs_error_vs_prefix_reference",
            "mean_error_vs_prefix_reference",
            "std_error_vs_prefix_reference",
            "mean_runtime_seconds",
            "mean_cost_evals",
            "mean_n_iterations",
        ])
        writer.writeheader()
        for row in summary_rows:
            writer.writerow(row)


# ──────────────────────────────────────────────────────────────────────────
# Plotting
# ──────────────────────────────────────────────────────────────────────────


def _prefix_figure(rows, *, exclude_qaoa):
    import matplotlib.pyplot as plt
    from matplotlib.patches import Patch
    from matplotlib.ticker import MaxNLocator

    by_algo = defaultdict(list)
    for row in rows:
        by_algo[row["algorithm"]].append(row)

    fig, ax = plt.subplots(figsize=(14, 7) if exclude_qaoa else (9.5, 5.0))
    colors = plt.get_cmap("tab20")
    for idx, (algorithm, algo_rows) in enumerate(sorted(by_algo.items())):
        if exclude_qaoa and algorithm == "qaoa_inspired_vqe":
            continue
        algo_rows.sort(key=lambda r: int(r["n_parameters"]))
        xs = [int(r["n_parameters"]) for r in algo_rows]
        ys = np.array([float(r["mean_abs_error_vs_full_reference"]) for r in algo_rows])
        stds = np.array([float(r["std_abs_error_vs_full_reference"]) for r in algo_rows])
        color = colors(idx % colors.N)
        ax.plot(xs, ys, marker="o", linewidth=1.8, label=algorithm, color=color)
        ax.fill_between(xs, np.maximum(ys - stds, 0.), ys + stds, color=color, alpha=0.15)

    title = "Exp 5 - Energy error vs Hamiltonian term count"
    if exclude_qaoa:
        title += "\nQAOA-inspired VQE excluded"
    ax.set_title(title)
    ax.set_xlabel("Number of retained Hamiltonian terms")
    ax.set_ylabel("Mean |energy − full CASCI| (Ha)")
    ax.xaxis.set_major_locator(MaxNLocator(integer=True))
    ax.set_yscale("symlog", linthresh=1e-12)
    if ax.get_ylim()[0] < 0.:
        ax.set_ylim(bottom=0.)
    ax.grid(True, which="both", alpha=0.3)
    handles, _ = ax.get_legend_handles_labels()
    if handles:
        handles.append(Patch(facecolor="gray", alpha=0.15, label="±1 population SD of molecule means, clipped at zero"))
        ax.legend(handles=handles, fontsize=8, ncol=2)
    else:
        ax.text(0.5, 0.5, "No contributing algorithms in this view",
                transform=ax.transAxes, ha="center", va="center")
    fig.tight_layout()
    return fig


def _plot_aggregate_trend(summary_csv: Path, output_dir: Path, *, exclude_qaoa=False) -> None:
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except Exception as exc:  # pragma: no cover
        logger.warning("matplotlib unavailable: %s", exc)
        return

    with open(summary_csv, newline="") as f:
        rows = list(csv.DictReader(f))
    if not rows:
        logger.warning("No summary rows available for plotting")
        return
    fig = _prefix_figure(rows, exclude_qaoa=exclude_qaoa)
    suffix = "_filtered" if exclude_qaoa else ""
    fig.savefig(output_dir / f"mean_absolute_error_vs_term_count{suffix}.png", dpi=150)
    plt.close(fig)


def _plot_aggregate_trend_filtered(summary_csv: Path, output_dir: Path) -> None:
    _plot_aggregate_trend(summary_csv, output_dir, exclude_qaoa=True)


# ──────────────────────────────────────────────────────────────────────────
# CLI
# ──────────────────────────────────────────────────────────────────────────


def _build_argparser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description="Experiment 5 -- energy error as a function of Hamiltonian term count.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument("--batch-amino-acids", action="store_true",
                   help="Run all amino acids in datasets2/.")
    p.add_argument("--molecules", nargs="+", default=None,
                   help="Explicit molecule list from datasets2/ (overrides --batch-amino-acids).")
    p.add_argument("--algorithms", nargs="+", default=list(DEFAULT_ALGORITHMS),
                   help="Algorithms to run.")
    p.add_argument("--term-counts", nargs="+", type=int, default=list(DEFAULT_TERM_COUNTS),
                   help="Hamiltonian prefix lengths to evaluate.")
    p.add_argument("--basis", default="sto-3g")
    p.add_argument("--cs-target-qubits", type=int, default=4,
                   help="<=0 to skip contextual-subspace reduction.")
    p.add_argument("--n-layers", type=int, default=DEFAULT_N_LAYERS)
    p.add_argument("--max-operators", type=int, default=DEFAULT_MAX_OPERATORS)
    p.add_argument("--gradient-threshold", type=float, default=DEFAULT_GRADIENT_THRESHOLD)
    p.add_argument("--optimizer", default="COBYLA")
    p.add_argument("--max-iter", type=int, default=200)
    p.add_argument("--convergence-threshold", type=float, default=1e-10)
    trials = p.add_mutually_exclusive_group()
    trials.add_argument("--seed", type=int, help="One matched trial seed (default 42).")
    trials.add_argument("--seeds", type=int, nargs="+",
                        help="Distinct trial seeds shared across every molecule, prefix and algorithm.")
    p.add_argument("--no-plots", action="store_true")
    p.add_argument("--output-dir", type=Path, default=None)
    p.add_argument("--prepared-inputs", type=Path, help="Replay the saved inputs/ directory from a sweep")
    p.add_argument("-v", "--verbose", action="store_true")
    return p


def main() -> None:
    args = _build_argparser().parse_args()

    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(asctime)s %(name)s [%(levelname)s] %(message)s",
    )
    for noisy_logger in ("pyscf", "pennylane", "matplotlib"):
        logging.getLogger(noisy_logger).setLevel(logging.WARNING)

    cs_target = args.cs_target_qubits if args.cs_target_qubits > 0 else None

    if args.molecules:
        molecules = tuple(m.strip().lower() for m in args.molecules)
    elif args.batch_amino_acids:
        molecules = DEFAULT_MOLECULES
    else:
        parser = _build_argparser()
        parser.print_help()
        return

    out = run_term_count_sweep(
        molecules=molecules,
        algorithms=tuple(args.algorithms),
        term_counts=tuple(args.term_counts),
        basis=args.basis,
        cs_target_qubits=cs_target,
        n_layers=args.n_layers,
        max_operators=args.max_operators,
        gradient_threshold=args.gradient_threshold,
        optimizer=args.optimizer,
        max_iterations=args.max_iter,
        convergence_threshold=args.convergence_threshold,
        random_seed=args.seed,
        trial_seeds=args.seeds,
        output_dir=args.output_dir,
        save_plots=not args.no_plots,
        prepared_inputs=args.prepared_inputs,
    )

    cfg: TermCountConfig = out["config"]
    print("\n" + "=" * 60)
    print("EXPERIMENT 5 -- HAMILTONIAN TERM-COUNT SWEEP")
    print("=" * 60)
    print(f"Molecules: {len(cfg.molecules)}")
    print(f"Algorithms: {len(cfg.algorithms)}")
    print(f"Term counts: {list(cfg.term_counts)}")
    print(f"OK rows: {out['n_ok']}  failed rows: {out['n_fail']}")
    print(f"Batch CSV: {out['batch_csv']}")
    print(f"Summary CSV: {out['summary_csv']}")
    print(f"Output folder: {out['output_dir']}")
    if out["n_fail"] or not out["n_ok"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
