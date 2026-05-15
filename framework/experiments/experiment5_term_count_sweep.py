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
For each (molecule, algorithm, term-count) triple we record:

* ``n_parameters``           -- number of Hamiltonian terms kept in the prefix.
* ``algorithm_n_parameters``  -- number of variational parameters used by the
  VQE algorithm itself.
* ``energy``                 -- final energy returned by the algorithm.
* ``error_vs_full_reference`` -- energy minus the full-molecule CASCI energy.
* ``prefix_reference_energy`` -- exact ground-state energy of the prefix
  Hamiltonian used in that run.
* ``error_vs_prefix_reference`` -- energy minus the prefix-Hamiltonian
  ground-state energy.
* ``n_cost_evals``           -- actual cost-function evaluations consumed.
* ``runtime_seconds``        -- wall-clock runtime.

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
import json
import logging
import os
import sys
import time
from collections import defaultdict
from dataclasses import asdict, dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np

_FW_DIR = Path(__file__).resolve().parent.parent
if str(_FW_DIR) not in sys.path:
    sys.path.insert(0, str(_FW_DIR))

from algorithms import get_algorithm, list_algorithms  # noqa: E402
from core.base_vqe import VQEResult  # noqa: E402
from core.hamiltonian_loader import Molecule, QubitHamiltonian  # noqa: E402
from experiments._common import (  # noqa: E402
    CostEvalCounter,
    LoadedHamiltonian,
    load_small_hamiltonian,
    make_backend_config,
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
    random_seed: int
    n_qubits_active: int = 0
    n_qubits_final: int = 0
    n_terms_available: int = 0


# ──────────────────────────────────────────────────────────────────────────
# Hamiltonian helpers
# ──────────────────────────────────────────────────────────────────────────

def _prefix_hamiltonian(full_hamiltonian: QubitHamiltonian, n_terms: int) -> QubitHamiltonian:
    """Return a Hamiltonian built from the first ``n_terms`` terms.

    The returned object preserves the original qubit count and carries the
    exact ground-state energy of the prefix Hamiltonian in the molecule's
    ``reference_energy`` field.
    """
    if n_terms < 1:
        raise ValueError("n_terms must be at least 1")

    n_terms = min(int(n_terms), int(full_hamiltonian.n_terms))
    coeffs = np.array(full_hamiltonian.coefficients[:n_terms], copy=True)
    pauli_strings = list(full_hamiltonian.pauli_strings[:n_terms])

    prefix_molecule = Molecule(
        abbreviation=full_hamiltonian.molecule.abbreviation,
        name=full_hamiltonian.molecule.name,
        n_qubits=full_hamiltonian.n_qubits,
        n_coefficients=n_terms,
        reference_energy=full_hamiltonian.molecule.reference_energy,
        hamiltonian_file=full_hamiltonian.molecule.hamiltonian_file,
        n_electrons=full_hamiltonian.molecule.n_electrons,
        n_orbitals=full_hamiltonian.molecule.n_orbitals,
        charge=full_hamiltonian.molecule.charge,
        spin=full_hamiltonian.molecule.spin,
        basis=full_hamiltonian.molecule.basis,
        coordinates=full_hamiltonian.molecule.coordinates,
        molecular_formula=full_hamiltonian.molecule.molecular_formula,
    )

    prefix_ham = QubitHamiltonian(
        molecule=prefix_molecule,
        coefficients=coeffs,
        pauli_strings=pauli_strings,
        n_qubits=full_hamiltonian.n_qubits,
        n_terms=n_terms,
        cs_initial_state=full_hamiltonian.cs_initial_state,
    )

    prefix_energy = prefix_ham._calculate_ground_state_energy(
        coeffs, pauli_strings, full_hamiltonian.n_qubits
    )
    prefix_ham.molecule.reference_energy = prefix_energy
    prefix_ham.molecule.truncated_ground_state_energy = prefix_energy
    return prefix_ham


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
    random_seed: int = 42,
    output_dir: Optional[Path] = None,
    save_plots: bool = True,
) -> Dict[str, object]:
    """Run the batch experiment across many molecules and algorithms."""
    term_counts_sorted = tuple(sorted({int(t) for t in term_counts if int(t) >= 1}))
    if not term_counts_sorted:
        raise ValueError("term_counts must contain at least one positive integer")

    molecule_list = tuple(m.strip().lower() for m in molecules)
    algorithm_list = tuple(dict.fromkeys(algorithms))

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

    total_rows_estimate = len(molecule_list) * len(term_counts_sorted) * len(algorithm_list)

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
        random_seed=random_seed,
    )

    rows: List[TermCountRow] = []
    failures: List[Dict[str, str]] = []

    with open(config_json, "w") as f:
        json.dump(asdict(cfg), f, indent=2)

    def _write_progress_tracker(*, completed: int, status: str, current: Optional[Dict[str, object]] = None) -> None:
        tracker = {
            "timestamp": datetime.now().isoformat(timespec="seconds"),
            "completed_rows": completed,
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
    batch_exists = batch_csv.exists() and batch_csv.stat().st_size > 0
    with open(batch_csv, "a", newline="", buffering=1) as batch_file:
        writer = csv.writer(batch_file)
        if not batch_exists:
            writer.writerow([
                "molecule",
                "algorithm",
                "n_parameters",
                "algorithm_n_parameters",
                "n_qubits",
                "n_terms_available",
                "energy",
                "full_reference_energy",
                "prefix_reference_energy",
                "error_vs_full_reference",
                "abs_error_vs_full_reference",
                "error_vs_prefix_reference",
                "abs_error_vs_prefix_reference",
                "n_iterations",
                "n_cost_evals",
                "runtime_seconds",
                "converged",
                "status",
                "notes",
            ])
            batch_file.flush()
            os.fsync(batch_file.fileno())
        _write_progress_tracker(completed=0, status="starting")

        try:
            for mol_index, mol in enumerate(mol_iter):
                try:
                    loaded = load_small_hamiltonian(
                        molecule=mol,
                        basis=basis,
                        cs_target_qubits=cs_target_qubits,
                    )
                except Exception as exc:
                    logger.exception("Failed to load molecule %s: %s", mol, exc)
                    failures.append({"molecule": mol, "algorithm": "<load>", "error": str(exc)})
                    completed_rows += len(term_counts_sorted) * len(algorithm_list)
                    _write_progress_tracker(
                        completed=completed_rows,
                        status="load_failed",
                        current={"molecule": mol},
                    )
                    continue

                for term_count in term_counts_sorted:
                    if term_count > loaded.hamiltonian.n_terms:
                        logger.warning(
                            "Skipping %s at n_terms=%d because only %d terms are available",
                            mol, term_count, loaded.hamiltonian.n_terms,
                        )
                        completed_rows += len(algorithm_list)
                        _write_progress_tracker(
                            completed=completed_rows,
                            status="skipped_term_count",
                            current={"molecule": mol, "n_parameters": term_count},
                        )
                        continue

                    prefix_ham = _prefix_hamiltonian(loaded.hamiltonian, term_count)
                    prefix_reference = float(prefix_ham.molecule.reference_energy)
                    full_reference = float(loaded.casci_energy) if loaded.casci_energy is not None else float("nan")

                    for algo_index, algo in enumerate(algorithm_list):
                        run_seed = random_seed + 10007 * mol_index + 101 * term_count + 17 * algo_index
                        current_meta = {
                            "molecule": mol,
                            "algorithm": algo,
                            "n_parameters": term_count,
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
                            )
                            energy = float(result.calculated_energy)
                            error_vs_full = energy - full_reference
                            error_vs_prefix = energy - prefix_reference
                            row = TermCountRow(
                                molecule=mol,
                                algorithm=algo,
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
                                status="ok",
                                notes="",
                            )
                            rows.append(row)
                            writer.writerow([
                                row.molecule,
                                row.algorithm,
                                row.n_parameters,
                                row.algorithm_n_parameters,
                                row.n_qubits,
                                row.n_terms_available,
                                f"{row.energy:.10f}",
                                f"{row.full_reference_energy:.10f}",
                                f"{row.prefix_reference_energy:.10f}",
                                f"{row.error_vs_full_reference:.10f}",
                                f"{row.abs_error_vs_full_reference:.10f}",
                                f"{row.error_vs_prefix_reference:.10f}",
                                f"{row.abs_error_vs_prefix_reference:.10f}",
                                row.n_iterations,
                                row.n_cost_evals,
                                f"{row.runtime_seconds:.4f}",
                                int(row.converged),
                                row.status,
                                row.notes,
                            ])
                            batch_file.flush()
                            os.fsync(batch_file.fileno())
                        except Exception as exc:
                            logger.exception(
                                "Experiment 5 failed for molecule=%s algorithm=%s term_count=%d",
                                mol, algo, term_count,
                            )
                            failures.append({
                                "molecule": mol,
                                "algorithm": algo,
                                "term_count": str(term_count),
                                "error": str(exc),
                            })
                            row = TermCountRow(
                                molecule=mol,
                                algorithm=algo,
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
                            writer.writerow([
                                row.molecule,
                                row.algorithm,
                                row.n_parameters,
                                row.algorithm_n_parameters,
                                row.n_qubits,
                                row.n_terms_available,
                                "",
                                f"{row.full_reference_energy:.10f}",
                                f"{row.prefix_reference_energy:.10f}",
                                "",
                                "",
                                "",
                                "",
                                row.n_iterations,
                                row.n_cost_evals,
                                f"{row.runtime_seconds:.4f}",
                                int(row.converged),
                                row.status,
                                row.notes,
                            ])
                            batch_file.flush()
                            os.fsync(batch_file.fileno())
                        completed_rows += 1
                        _write_progress_tracker(
                            completed=completed_rows,
                            status="running",
                            current=current_meta,
                        )

                cfg.n_qubits_active = int(loaded.n_qubits_active)
                cfg.n_qubits_final = int(loaded.n_qubits_final)
                cfg.n_terms_available = int(loaded.hamiltonian.n_terms)
        finally:
            batch_file.flush()
            os.fsync(batch_file.fileno())

    _write_summary_csv(summary_csv, rows)
    if failures:
        with open(failures_json, "w") as f:
            json.dump(failures, f, indent=2)

    _write_progress_tracker(completed=completed_rows, status="finished")

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
        "n_fail": sum(1 for row in rows if row.status != "ok"),
    }


# ──────────────────────────────────────────────────────────────────────────
# Output helpers
# ──────────────────────────────────────────────────────────────────────────


def _write_long_csv(path: Path, rows: Sequence[TermCountRow]) -> None:
    with open(path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow([
            "molecule",
            "algorithm",
            "n_parameters",
            "algorithm_n_parameters",
            "n_qubits",
            "n_terms_available",
            "energy",
            "full_reference_energy",
            "prefix_reference_energy",
            "error_vs_full_reference",
            "abs_error_vs_full_reference",
            "error_vs_prefix_reference",
            "abs_error_vs_prefix_reference",
            "n_iterations",
            "n_cost_evals",
            "runtime_seconds",
            "converged",
            "status",
            "notes",
        ])
        for row in rows:
            writer.writerow([
                row.molecule,
                row.algorithm,
                row.n_parameters,
                row.algorithm_n_parameters,
                row.n_qubits,
                row.n_terms_available,
                f"{row.energy:.10f}" if np.isfinite(row.energy) else "",
                f"{row.full_reference_energy:.10f}" if np.isfinite(row.full_reference_energy) else "",
                f"{row.prefix_reference_energy:.10f}" if np.isfinite(row.prefix_reference_energy) else "",
                f"{row.error_vs_full_reference:.10f}" if np.isfinite(row.error_vs_full_reference) else "",
                f"{row.abs_error_vs_full_reference:.10f}" if np.isfinite(row.abs_error_vs_full_reference) else "",
                f"{row.error_vs_prefix_reference:.10f}" if np.isfinite(row.error_vs_prefix_reference) else "",
                f"{row.abs_error_vs_prefix_reference:.10f}" if np.isfinite(row.abs_error_vs_prefix_reference) else "",
                row.n_iterations,
                row.n_cost_evals,
                f"{row.runtime_seconds:.4f}",
                int(row.converged),
                row.status,
                row.notes,
            ])


def _write_summary_csv(path: Path, rows: Sequence[TermCountRow]) -> None:
    grouped: Dict[Tuple[str, int], List[TermCountRow]] = defaultdict(list)
    for row in rows:
        if row.status != "ok" or not np.isfinite(row.abs_error_vs_full_reference):
            continue
        grouped[(row.algorithm, row.n_parameters)].append(row)

    summary_rows = []
    for (algorithm, n_parameters), group in sorted(grouped.items()):
        abs_errors = np.array([r.abs_error_vs_full_reference for r in group], dtype=float)
        signed_errors = np.array([r.error_vs_full_reference for r in group], dtype=float)
        runtimes = np.array([r.runtime_seconds for r in group], dtype=float)
        cost_evals = np.array([r.n_cost_evals for r in group], dtype=float)
        iterations = np.array([r.n_iterations for r in group], dtype=float)
        summary_rows.append({
            "algorithm": algorithm,
            "n_parameters": n_parameters,
            "n_samples": int(len(group)),
            "mean_abs_error_vs_full_reference": float(np.mean(abs_errors)),
            "std_abs_error_vs_full_reference": float(np.std(abs_errors)),
            "mean_error_vs_full_reference": float(np.mean(signed_errors)),
            "std_error_vs_full_reference": float(np.std(signed_errors)),
            "mean_runtime_seconds": float(np.mean(runtimes)),
            "mean_cost_evals": float(np.mean(cost_evals)),
            "mean_n_iterations": float(np.mean(iterations)),
        })

    with open(path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=[
            "algorithm",
            "n_parameters",
            "n_samples",
            "mean_abs_error_vs_full_reference",
            "std_abs_error_vs_full_reference",
            "mean_error_vs_full_reference",
            "std_error_vs_full_reference",
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


def _plot_aggregate_trend(summary_csv: Path, output_dir: Path) -> None:
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except Exception as exc:  # pragma: no cover
        logger.warning("matplotlib unavailable: %s", exc)
        return

    import csv as _csv

    by_algo: Dict[str, List[Dict[str, str]]] = defaultdict(list)
    with open(summary_csv, newline="") as f:
        reader = _csv.DictReader(f)
        for row in reader:
            by_algo[row["algorithm"]].append(row)

    if not by_algo:
        logger.warning("No summary rows available for plotting")
        return

    fig, ax = plt.subplots(figsize=(9.5, 5.0))
    colors = plt.get_cmap("tab20")

    for idx, (algorithm, algo_rows) in enumerate(sorted(by_algo.items())):
        algo_rows.sort(key=lambda r: int(r["n_parameters"]))
        xs = np.array([int(r["n_parameters"]) for r in algo_rows], dtype=float)
        ys = np.array([float(r["mean_abs_error_vs_full_reference"]) for r in algo_rows], dtype=float)
        stds = np.array([float(r["std_abs_error_vs_full_reference"]) for r in algo_rows], dtype=float)
        color = colors(idx % colors.N)
        ax.plot(xs, ys, marker="o", linewidth=1.8, label=algorithm, color=color)
        ax.fill_between(xs, np.maximum(ys - stds, 1e-12), ys + stds, color=color, alpha=0.15)

    ax.set_yscale("log")
    ax.set_xlabel("n_parameters = number of Hamiltonian prefix terms")
    ax.set_ylabel("mean |energy error| vs full CASCI (Ha)")
    ax.set_title("Experiment 5: mean energy error vs Hamiltonian term count")
    ax.grid(True, which="both", alpha=0.3)
    ax.legend(fontsize=8, ncol=2)
    fig.tight_layout()
    fig.savefig(output_dir / "mean_absolute_error_vs_term_count.png", dpi=150)
    plt.close(fig)


def _plot_aggregate_trend_filtered(summary_csv: Path, output_dir: Path) -> None:
    """
    Plot aggregate trend excluding qaoa_inspired_vqe (has anomalous spike at 4 prefix terms),
    with better scaling and larger figure for visibility of data points.
    """
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except Exception as exc:  # pragma: no cover
        logger.warning("matplotlib unavailable: %s", exc)
        return

    import csv as _csv

    by_algo: Dict[str, List[Dict[str, str]]] = defaultdict(list)
    with open(summary_csv, newline="") as f:
        reader = _csv.DictReader(f)
        for row in reader:
            # Filter out qaoa_inspired_vqe (has anomaly at n_parameters=4)
            if row["algorithm"] == "qaoa_inspired_vqe":
                continue
            by_algo[row["algorithm"]].append(row)

    if not by_algo:
        logger.warning("No summary rows available for filtered plotting")
        return

    fig, ax = plt.subplots(figsize=(14, 7))
    colors = plt.get_cmap("tab20")

    for idx, (algorithm, algo_rows) in enumerate(sorted(by_algo.items())):
        algo_rows.sort(key=lambda r: int(r["n_parameters"]))
        xs = np.array([int(r["n_parameters"]) for r in algo_rows], dtype=float)
        ys = np.array([float(r["mean_abs_error_vs_full_reference"]) for r in algo_rows], dtype=float)
        stds = np.array([float(r["std_abs_error_vs_full_reference"]) for r in algo_rows], dtype=float)
        color = colors(idx % colors.N)
        ax.plot(xs, ys, marker="o", linewidth=2.0, markersize=8, label=algorithm, color=color)
        ax.fill_between(xs, np.maximum(ys - stds, 1e-12), ys + stds, color=color, alpha=0.15)

    ax.set_yscale("log")
    ax.set_xlabel("n_parameters = number of Hamiltonian prefix terms", fontsize=12)
    ax.set_ylabel("mean |energy error| vs full CASCI (Ha)", fontsize=12)
    ax.set_title("Experiment 5: mean energy error vs Hamiltonian term count (outlier filtered)", fontsize=14)
    ax.grid(True, which="both", alpha=0.3)
    ax.legend(fontsize=10, ncol=2, loc="best")
    ax.set_xlim(0.5, 10.5)  # Better x-axis scaling
    fig.tight_layout()
    fig.savefig(output_dir / "mean_absolute_error_vs_term_count_filtered.png", dpi=150)
    plt.close(fig)


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
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--no-plots", action="store_true")
    p.add_argument("--output-dir", type=Path, default=None)
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
        output_dir=args.output_dir,
        save_plots=not args.no_plots,
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


if __name__ == "__main__":
    main()
