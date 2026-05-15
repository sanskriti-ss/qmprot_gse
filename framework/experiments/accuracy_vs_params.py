"""
Experiment 4 -- Accuracy as a function of the number of parameters
==================================================================

Question
--------
For a fixed VQE problem, how does the achievable energy improve as we
allow the ansatz to use more variational parameters?

Method
------
The natural way to ask this for a chemistry-inspired ansatz is to use
an **adaptive VQE** with an operator pool: at each step it greedily
picks the gradient-largest operator, so after ``k`` steps the ansatz
contains the (greedy) "top-k" operators for this Hamiltonian.

We default to ``qubit_adapt_vqe`` (Pauli-string operator pool with
``PauliRot(2*theta, ...)``) rather than ``adapt_vqe`` (RY+CNOT
"single/double" approximations) because the simplified gates in
``adapt_vqe`` give zero finite-difference gradient at theta=0 by
parameterisation symmetry on these small Hamiltonians.

We also default to ``cs_target_qubits=None`` (no contextual-subspace
reduction).  CS is excellent for cheap statevector verification, but
it places the initial state at the noncontextual ground state -- which
is by construction a stationary point of the noncontextual sub-
Hamiltonian, so all greedy operators look like zero-gradient additions
and ADAPT exits at k=0.  Leaving CS off means the ansatz starts from
HF and ADAPT can grow towards CASCI.

Key efficiency trick
~~~~~~~~~~~~~~~~~~~~
A single ADAPT run with ``max_operators = max(k_list)`` automatically
records the energy after every operator addition in
``convergence_history``.  We therefore only need *one* ADAPT run to
recover energies at every requested k value, regardless of how many
points the user asks for.

Modular k-list
--------------
The default sweep is ``1, 2, 3, 5, 10`` (cheap, ~30s on 4 qubits).
Pass ``--k-list`` to evaluate any list -- including ``30 100 300``
which the user explicitly mentioned.  The script will run ADAPT once
up to ``max(k_list)`` operators (potentially slow for large values)
and slice the result.

What we measure (per k)
-----------------------

* ``energy``                 -- ADAPT energy after exactly ``k`` operators.
* ``error_vs_casci``         -- ``energy - CASCI`` (positive when above CI).
* ``error_vs_hf``            -- ``energy - HF``     (negative when below HF,
  which is the variational improvement over Hartree-Fock).
* ``cumulative_cost_evals``  -- total cost-fn evals consumed by ADAPT to
  reach this k (so we get a free trainability signal too).
* ``cumulative_runtime``     -- approximate runtime to reach this k.

Outputs
-------
``framework/experiments/results/accuracy_vs_params/<timestamp>_<molecule>_adapt_vqe/``

* ``run_config.json``
* ``results.csv``                    -- one row per k.
* ``full_convergence.csv``           -- the entire ADAPT trace (one row per op).
* ``energy_vs_n_params.png``         -- log-y absolute |error vs CASCI|.
* ``energy_vs_n_params_linear.png``  -- linear energy axis with HF/CASCI lines.

Batch Outputs
-------------
``framework/experiments/results/accuracy_vs_params_batch/<timestamp>_batch/``

* ``run_config.json``
* ``batch_all_results.csv``
* ``summary_by_k.csv``
* ``mean_abs_error_vs_k.png``
* ``mean_cost_evals_vs_k.png``

Run as a script
---------------
::

    python -m experiments.accuracy_vs_params \\
        --molecule water --k-list 1 2 3 5 10

    python -m experiments.accuracy_vs_params \\
        --batch-amino-acids --k-list 1 2 3
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np

_FW_DIR = Path(__file__).resolve().parent.parent
if str(_FW_DIR) not in sys.path:
    sys.path.insert(0, str(_FW_DIR))

from algorithms import get_algorithm  # noqa: E402

from experiments._common import (  # noqa: E402
    CostEvalCounter,
    LoadedHamiltonian,
    load_small_hamiltonian,
    make_backend_config,
    make_run_dir,
)
from experiments.noise_resilience import DATASETS2_AMINO_ACIDS  # noqa: E402

logger = logging.getLogger(__name__)


def _display_lower_band(mean: np.ndarray, std: np.ndarray, min_fraction: float = 0.55) -> np.ndarray:
    """Return a positive lower envelope suitable for log-scale plotting.

    The batch plots are meant to show uncertainty visually, not collapse
    to a near-zero band when mean - std becomes very small. This keeps
    the band anchored to a reasonable fraction of the mean while still
    reflecting the variability in the data.
    """
    mean = np.asarray(mean, dtype=float)
    std = np.asarray(std, dtype=float)
    raw_lower = mean - std
    return np.maximum(raw_lower, mean * float(min_fraction))


DEFAULT_K_LIST: Tuple[int, ...] = (1, 2, 3, 5, 10)
DEFAULT_ALGORITHM: str = "qubit_adapt_vqe"
DEFAULT_GRADIENT_THRESHOLD: float = 1e-8
DEFAULT_MOLECULES: Tuple[str, ...] = DATASETS2_AMINO_ACIDS


# ──────────────────────────────────────────────────────────────────────────
# Result containers
# ──────────────────────────────────────────────────────────────────────────

@dataclass
class AccuracyRow:
    k: int                         # requested operator count
    energy: float                  # ADAPT energy after k operators
    error_vs_casci: float          # energy - CASCI
    error_vs_hf: float             # energy - HF (negative => improvement)
    cumulative_cost_evals: int     # total cost-fn calls to reach this k
    cumulative_runtime_s: float    # approximate runtime to reach this k


@dataclass
class AccuracyConfig:
    molecule: str
    basis: str
    cs_target_qubits: Optional[int]
    algorithm: str
    gradient_threshold: float
    k_list: Tuple[int, ...]
    optimizer: str
    inner_max_iter: int
    convergence_threshold: float
    random_seed: int
    n_qubits_active: int
    n_qubits_final: int
    casci_energy: Optional[float]
    hf_energy: Optional[float]
    n_pool_operators: int          # populated after the ADAPT run
    actual_max_k: int              # ADAPT may exit early if pool exhausted


@dataclass
class AccuracyBatchRow:
    molecule: str
    k: int
    energy: float
    error_vs_casci: float
    error_vs_hf: float
    cumulative_cost_evals: int
    cumulative_runtime_s: float
    n_qubits_final: int
    n_pool_operators: int
    actual_max_k: int
    status: str = "ok"
    notes: str = ""


@dataclass
class AccuracyBatchConfig:
    molecules: Tuple[str, ...]
    basis: str
    cs_target_qubits: Optional[int]
    algorithm: str
    gradient_threshold: float
    k_list: Tuple[int, ...]
    optimizer: str
    inner_max_iter: int
    convergence_threshold: float
    random_seed: int


# ──────────────────────────────────────────────────────────────────────────
# Public API
# ──────────────────────────────────────────────────────────────────────────

def run_accuracy_vs_params(
    molecule: str = "water",
    basis: str = "sto-3g",
    cs_target_qubits: Optional[int] = None,
    algorithm: str = DEFAULT_ALGORITHM,
    k_list: Tuple[int, ...] = DEFAULT_K_LIST,
    gradient_threshold: float = DEFAULT_GRADIENT_THRESHOLD,
    optimizer: str = "COBYLA",
    inner_max_iter: int = 200,
    convergence_threshold: float = 1e-10,
    random_seed: int = 42,
    output_dir: Optional[Path] = None,
    save_plots: bool = True,
) -> Dict[str, object]:
    """Run the accuracy-vs-parameter-count study.

    A single ADAPT-VQE run with ``max_operators = max(k_list)`` records
    the energy after every operator addition.  We slice that trace at
    the requested ``k`` values to produce one :class:`AccuracyRow` per k.

    Returns a dict with keys ``"config"``, ``"rows"``, ``"full_history"``,
    ``"per_step_runtime"`` and ``"output_dir"``.
    """
    if not k_list:
        raise ValueError("k_list must be non-empty")
    k_sorted = tuple(sorted(set(int(k) for k in k_list if k >= 1)))
    if not k_sorted:
        raise ValueError("k_list must contain at least one k >= 1")
    max_k = k_sorted[-1]

    loaded = load_small_hamiltonian(
        molecule=molecule, basis=basis, cs_target_qubits=cs_target_qubits,
    )

    # ── Build adaptive VQE capped at max_k operators ────────────────────
    if algorithm not in ("adapt_vqe", "qubit_adapt_vqe"):
        raise ValueError(
            f"algorithm={algorithm!r} not supported by accuracy_vs_params; "
            "must be one of {'adapt_vqe', 'qubit_adapt_vqe'}.  "
            "Both algorithms expose ``max_operators`` and emit one energy "
            "per operator addition in ``convergence_history``."
        )
    AlgorithmCls = get_algorithm(algorithm)
    bc = make_backend_config(loaded.hamiltonian.n_qubits)
    vqe = AlgorithmCls(
        loaded.hamiltonian,
        max_operators=max_k,
        gradient_threshold=gradient_threshold,
        optimizer=optimizer,
        max_iterations=inner_max_iter,
        convergence_threshold=convergence_threshold,
        random_seed=random_seed,
        backend_config=bc,
    )

    # ── Wrap with cost-eval counter and a per-iteration timestamp hook ──
    counter = CostEvalCounter()
    counter.attach(vqe)

    # We piggy-back on ADAPT's append-after-each-operator pattern to
    # capture the cumulative cost-eval count at the moment each operator
    # is added.  Two complications:
    #
    #   1.  ``list.append`` is a slot and cannot be monkey-patched on an
    #       instance.  We therefore use a ``list`` subclass with an
    #       overridden ``append``.
    #   2.  Both ADAPT implementations begin ``run()`` with
    #       ``self.convergence_history = []``, which would *replace* our
    #       wrapped list with a plain one.  We install a class-level
    #       property whose setter rewraps any assignment in our snapshot
    #       list, so the hook survives that reset.
    per_op_cost_evals: List[int] = []
    per_op_timestamps: List[float] = []
    t_start = time.time()

    class _SnapshottingList(list):
        def append(self, value):
            per_op_cost_evals.append(int(counter.count))
            per_op_timestamps.append(time.time() - t_start)
            super().append(value)

    # Build a subclass of the algorithm class with a property that re-wraps
    # any assignment to ``convergence_history``.
    OriginalCls = vqe.__class__

    class _PatchedCls(OriginalCls):
        @property
        def convergence_history(self):
            return self._snapshotting_history

        @convergence_history.setter
        def convergence_history(self, value):
            self._snapshotting_history = _SnapshottingList(value)

    # Seed the underlying storage and switch class.
    vqe._snapshotting_history = _SnapshottingList(
        getattr(vqe, "convergence_history", [])
    )
    vqe.__class__ = _PatchedCls

    try:
        result = vqe.run()
    finally:
        counter.detach(vqe)
        # Restore original class so the algorithm instance is "clean" if
        # the caller wants to inspect it after the run.
        vqe.__class__ = OriginalCls

    full_history = list(result.convergence_history)
    n_pool = len(vqe.operator_pool)
    actual_max_k = result.n_parameters

    if actual_max_k == 0:
        raise RuntimeError(
            "ADAPT added no operators -- is the operator pool empty or did "
            "the gradient threshold trigger immediately?"
        )

    cfg = AccuracyConfig(
        molecule=molecule, basis=basis,
        cs_target_qubits=cs_target_qubits,
        algorithm=algorithm,
        gradient_threshold=gradient_threshold,
        k_list=k_sorted,
        optimizer=optimizer,
        inner_max_iter=inner_max_iter,
        convergence_threshold=convergence_threshold,
        random_seed=random_seed,
        n_qubits_active=loaded.n_qubits_active,
        n_qubits_final=loaded.n_qubits_final,
        casci_energy=loaded.casci_energy,
        hf_energy=loaded.hf_energy,
        n_pool_operators=n_pool,
        actual_max_k=actual_max_k,
    )

    out_dir = output_dir or make_run_dir(
        "accuracy_vs_params", molecule, algorithm,
    )
    logger.info("Output dir: %s", out_dir)

    casci = loaded.casci_energy if loaded.casci_energy is not None else float("nan")
    hf = loaded.hf_energy if loaded.hf_energy is not None else float("nan")

    rows: List[AccuracyRow] = []
    for k in k_sorted:
        if k > actual_max_k:
            logger.warning(
                "Requested k=%d exceeds actual_max_k=%d (ADAPT ended early). "
                "Reporting energy at actual_max_k.", k, actual_max_k,
            )
            idx = actual_max_k - 1
            k_use = actual_max_k
        else:
            idx = k - 1
            k_use = k
        e = float(full_history[idx])
        rows.append(AccuracyRow(
            k=k_use,
            energy=e,
            error_vs_casci=e - casci,
            error_vs_hf=e - hf,
            cumulative_cost_evals=int(per_op_cost_evals[idx])
                if idx < len(per_op_cost_evals) else int(counter.count),
            cumulative_runtime_s=float(per_op_timestamps[idx])
                if idx < len(per_op_timestamps) else float(time.time() - t_start),
        ))

    _save_outputs(
        cfg, rows, full_history, per_op_cost_evals, per_op_timestamps,
        out_dir, save_plots=save_plots,
    )

    return {
        "config": cfg,
        "rows": rows,
        "full_history": full_history,
        "per_step_cost_evals": per_op_cost_evals,
        "per_step_runtime": per_op_timestamps,
        "output_dir": out_dir,
    }


def run_accuracy_vs_params_batch(
    molecules: Sequence[str] = DEFAULT_MOLECULES,
    basis: str = "sto-3g",
    cs_target_qubits: Optional[int] = None,
    algorithm: str = DEFAULT_ALGORITHM,
    k_list: Tuple[int, ...] = DEFAULT_K_LIST,
    gradient_threshold: float = DEFAULT_GRADIENT_THRESHOLD,
    optimizer: str = "COBYLA",
    inner_max_iter: int = 200,
    convergence_threshold: float = 1e-10,
    random_seed: int = 42,
    output_dir: Optional[Path] = None,
    save_plots: bool = True,
) -> Dict[str, object]:
    """Run accuracy-vs-params across multiple molecules and aggregate outputs."""
    if not k_list:
        raise ValueError("k_list must be non-empty")
    k_sorted = tuple(sorted(set(int(k) for k in k_list if k >= 1)))
    if not k_sorted:
        raise ValueError("k_list must contain at least one k >= 1")

    molecule_list = tuple(m.strip().lower() for m in molecules)

    ts = time.strftime("%Y%m%d_%H%M%S")
    if output_dir is None:
        output_dir = (
            _FW_DIR / "experiments" / "results" / "accuracy_vs_params_batch"
            / f"{ts}_batch"
        )
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    batch_csv = output_dir / "batch_all_results.csv"
    summary_csv = output_dir / "summary_by_k.csv"
    config_json = output_dir / "run_config.json"
    failures_json = output_dir / "batch_failures.json"

    cfg = AccuracyBatchConfig(
        molecules=tuple(molecule_list),
        basis=basis,
        cs_target_qubits=cs_target_qubits,
        algorithm=algorithm,
        gradient_threshold=gradient_threshold,
        k_list=k_sorted,
        optimizer=optimizer,
        inner_max_iter=inner_max_iter,
        convergence_threshold=convergence_threshold,
        random_seed=random_seed,
    )

    rows: List[AccuracyBatchRow] = []
    failures: List[Dict[str, str]] = []

    with open(config_json, "w") as f:
        json.dump(asdict(cfg), f, indent=2)

    import csv

    with open(batch_csv, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow([
            "molecule",
            "k",
            "energy",
            "error_vs_casci",
            "error_vs_hf",
            "cumulative_cost_evals",
            "cumulative_runtime_s",
            "n_qubits_final",
            "n_pool_operators",
            "actual_max_k",
            "status",
            "notes",
        ])

        for mol_index, mol in enumerate(molecule_list):
            subdir = output_dir / f"{mol}_{algorithm}"
            try:
                out = run_accuracy_vs_params(
                    molecule=mol,
                    basis=basis,
                    cs_target_qubits=cs_target_qubits,
                    algorithm=algorithm,
                    k_list=k_sorted,
                    gradient_threshold=gradient_threshold,
                    optimizer=optimizer,
                    inner_max_iter=inner_max_iter,
                    convergence_threshold=convergence_threshold,
                    random_seed=random_seed + 31 * mol_index,
                    output_dir=subdir,
                    save_plots=save_plots,
                )
                mol_cfg: AccuracyConfig = out["config"]
                mol_rows: List[AccuracyRow] = out["rows"]
                for r in mol_rows:
                    row = AccuracyBatchRow(
                        molecule=mol,
                        k=int(r.k),
                        energy=float(r.energy),
                        error_vs_casci=float(r.error_vs_casci),
                        error_vs_hf=float(r.error_vs_hf),
                        cumulative_cost_evals=int(r.cumulative_cost_evals),
                        cumulative_runtime_s=float(r.cumulative_runtime_s),
                        n_qubits_final=int(mol_cfg.n_qubits_final),
                        n_pool_operators=int(mol_cfg.n_pool_operators),
                        actual_max_k=int(mol_cfg.actual_max_k),
                        status="ok",
                        notes="",
                    )
                    rows.append(row)
                    w.writerow([
                        row.molecule,
                        row.k,
                        f"{row.energy:.10f}",
                        f"{row.error_vs_casci:.10f}",
                        f"{row.error_vs_hf:.10f}",
                        row.cumulative_cost_evals,
                        f"{row.cumulative_runtime_s:.4f}",
                        row.n_qubits_final,
                        row.n_pool_operators,
                        row.actual_max_k,
                        row.status,
                        row.notes,
                    ])
            except Exception as exc:
                logger.exception("Batch accuracy_vs_params failed for %s", mol)
                failures.append({"molecule": mol, "error": str(exc)})
                for k in k_sorted:
                    row = AccuracyBatchRow(
                        molecule=mol,
                        k=int(k),
                        energy=float("nan"),
                        error_vs_casci=float("nan"),
                        error_vs_hf=float("nan"),
                        cumulative_cost_evals=0,
                        cumulative_runtime_s=0.0,
                        n_qubits_final=0,
                        n_pool_operators=0,
                        actual_max_k=0,
                        status="failed",
                        notes=str(exc),
                    )
                    rows.append(row)
                    w.writerow([
                        row.molecule,
                        row.k,
                        "",
                        "",
                        "",
                        row.cumulative_cost_evals,
                        f"{row.cumulative_runtime_s:.4f}",
                        row.n_qubits_final,
                        row.n_pool_operators,
                        row.actual_max_k,
                        row.status,
                        row.notes,
                    ])

    if failures:
        with open(failures_json, "w") as f:
            json.dump(failures, f, indent=2)

    _write_accuracy_batch_summary(summary_csv, rows)
    if save_plots:
        _plot_accuracy_batch_summary(summary_csv, output_dir)
        _plot_accuracy_batch_overview(batch_csv, summary_csv, output_dir)

    logger.info("Wrote batch outputs to %s", output_dir)
    return {
        "config": cfg,
        "rows": rows,
        "output_dir": output_dir,
        "batch_csv": batch_csv,
        "summary_csv": summary_csv,
        "failures": failures,
    }


# ──────────────────────────────────────────────────────────────────────────
# Output: CSV / JSON / plots
# ──────────────────────────────────────────────────────────────────────────

def _save_outputs(
    cfg: AccuracyConfig,
    rows: List[AccuracyRow],
    full_history: List[float],
    per_op_cost_evals: List[int],
    per_op_timestamps: List[float],
    out_dir: Path,
    save_plots: bool = True,
) -> None:
    import csv

    out_dir.mkdir(parents=True, exist_ok=True)

    cfg_dict = asdict(cfg)
    cfg_dict["k_list"] = list(cfg.k_list)
    with open(out_dir / "run_config.json", "w") as f:
        json.dump(cfg_dict, f, indent=2)

    with open(out_dir / "results.csv", "w", newline="") as f:
        w = csv.writer(f)
        w.writerow([
            "k", "energy", "error_vs_casci", "error_vs_hf",
            "cumulative_cost_evals", "cumulative_runtime_s",
        ])
        for r in rows:
            w.writerow([
                r.k, f"{r.energy:.10f}", f"{r.error_vs_casci:.10f}",
                f"{r.error_vs_hf:.10f}", r.cumulative_cost_evals,
                f"{r.cumulative_runtime_s:.4f}",
            ])

    # Save the full ADAPT trace at every operator addition (not just at k_list)
    with open(out_dir / "full_convergence.csv", "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["k", "energy", "cumulative_cost_evals", "cumulative_runtime_s"])
        for i, e in enumerate(full_history):
            k = i + 1
            cev = per_op_cost_evals[i] if i < len(per_op_cost_evals) else ""
            ts = per_op_timestamps[i] if i < len(per_op_timestamps) else ""
            w.writerow([k, f"{e:.10f}", cev,
                        f"{ts:.4f}" if isinstance(ts, float) else ""])

    if not save_plots:
        return

    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except Exception as exc:  # pragma: no cover
        logger.warning("matplotlib unavailable: %s", exc)
        return

    casci = cfg.casci_energy
    hf = cfg.hf_energy

    # ── Plot 1: linear energy vs n_params ────────────────────────────────
    fig, ax = plt.subplots(figsize=(8, 4.5))
    full_xs = np.arange(1, len(full_history) + 1)
    ax.plot(full_xs, full_history, color="tab:blue", linewidth=1, alpha=0.5,
            label="ADAPT (every operator)")
    ax.plot(
        [r.k for r in rows], [r.energy for r in rows],
        marker="o", linestyle="", color="tab:blue",
        label="reported k values",
    )
    if hf is not None:
        ax.axhline(hf, color="grey", linestyle="--", linewidth=1,
                   label=f"HF = {hf:.5f}")
    if casci is not None:
        ax.axhline(casci, color="black", linestyle=":", linewidth=1,
                   label=f"CASCI = {casci:.5f}")
    ax.set_xlabel("k = # ADAPT-selected operators (= # parameters)")
    ax.set_ylabel("energy (Ha)")
    ax.set_title(
        f"Accuracy vs n_parameters (ADAPT-VQE)\n"
        f"{cfg.molecule}, {cfg.n_qubits_final} qubits, pool={cfg.n_pool_operators}"
    )
    ax.grid(True, alpha=0.3)
    ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(out_dir / "energy_vs_n_params_linear.png", dpi=150)
    plt.close(fig)

    # ── Plot 2: |error vs CASCI| on log y ────────────────────────────────
    if casci is not None:
        fig, ax = plt.subplots(figsize=(8, 4.5))
        full_err = np.abs(np.asarray(full_history) - casci)
        # Use a tiny floor so log scale doesn't blow up at zero
        full_err = np.where(full_err < 1e-12, 1e-12, full_err)
        ax.plot(full_xs, full_err, color="tab:red", linewidth=1, alpha=0.5,
                label="ADAPT (every operator)")
        ax.plot(
            [r.k for r in rows], [abs(r.error_vs_casci) for r in rows],
            marker="o", linestyle="", color="tab:red",
            label="reported k values",
        )
        # Chemical accuracy line: 1.6 mHa
        ax.axhline(1.6e-3, color="green", linestyle="--", linewidth=1,
                   label="chemical accuracy = 1.6 mHa")
        ax.set_yscale("log")
        ax.set_xlabel("k = # ADAPT-selected operators (= # parameters)")
        ax.set_ylabel(r"$|E_\mathrm{ADAPT}(k) - E_\mathrm{CASCI}|$ (Ha)")
        ax.set_title(
            f"Energy error vs n_parameters (ADAPT-VQE)\n"
            f"{cfg.molecule}, {cfg.n_qubits_final} qubits, pool={cfg.n_pool_operators}"
        )
        ax.grid(True, which="both", alpha=0.3)
        ax.legend(fontsize=8)
        fig.tight_layout()
        fig.savefig(out_dir / "energy_vs_n_params.png", dpi=150)
        plt.close(fig)

    # ── Plot 3: cumulative cost evals vs k ───────────────────────────────
    if per_op_cost_evals:
        fig, ax = plt.subplots(figsize=(8, 4.5))
        ax.plot(np.arange(1, len(per_op_cost_evals) + 1),
                per_op_cost_evals, marker="o", color="tab:purple")
        ax.set_xlabel("k = # ADAPT-selected operators")
        ax.set_ylabel("cumulative # cost-fn evaluations")
        ax.set_yscale("log")
        ax.set_title(
            f"ADAPT cost-fn evaluations to reach k operators\n"
            f"{cfg.molecule}, {cfg.n_qubits_final} qubits"
        )
        ax.grid(True, which="both", alpha=0.3)
        fig.tight_layout()
        fig.savefig(out_dir / "cost_evals_vs_k.png", dpi=150)
        plt.close(fig)

    logger.info("Wrote outputs to %s", out_dir)


def _write_accuracy_batch_summary(path: Path, rows: Sequence[AccuracyBatchRow]) -> None:
    import csv

    grouped: Dict[int, List[AccuracyBatchRow]] = {}
    for row in rows:
        if row.status != "ok" or not np.isfinite(row.error_vs_casci):
            continue
        grouped.setdefault(row.k, []).append(row)

    with open(path, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow([
            "k",
            "n_molecules",
            "mean_abs_error_vs_casci",
            "std_abs_error_vs_casci",
            "mean_cost_evals",
            "std_cost_evals",
        ])
        for k in sorted(grouped.keys()):
            group = grouped[k]
            abs_err = np.array([abs(r.error_vs_casci) for r in group], dtype=float)
            cost_evals = np.array([r.cumulative_cost_evals for r in group], dtype=float)
            w.writerow([
                k,
                len(group),
                f"{float(np.mean(abs_err)):.10f}",
                f"{float(np.std(abs_err)):.10f}",
                f"{float(np.mean(cost_evals)):.4f}",
                f"{float(np.std(cost_evals)):.4f}",
            ])


def _plot_accuracy_batch_summary(summary_csv: Path, output_dir: Path) -> None:
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except Exception as exc:  # pragma: no cover
        logger.warning("matplotlib unavailable: %s", exc)
        return

    data = np.genfromtxt(summary_csv, delimiter=",", names=True, dtype=None, encoding=None)
    if data.size == 0:
        return

    if data.ndim == 0:
        data = np.array([data])

    ks = np.array(data["k"], dtype=int)
    mean_err = np.array(data["mean_abs_error_vs_casci"], dtype=float)
    std_err = np.array(data["std_abs_error_vs_casci"], dtype=float)
    mean_cost = np.array(data["mean_cost_evals"], dtype=float)
    std_cost = np.array(data["std_cost_evals"], dtype=float)

    fig, ax = plt.subplots(figsize=(8, 4.5))
    ax.plot(ks, mean_err, marker="o", linewidth=2.0, color="tab:red")
    lower_err = _display_lower_band(mean_err, std_err)
    upper_err = mean_err + std_err
    ax.fill_between(ks, lower_err, upper_err,
                    color="tab:red", alpha=0.15)
    ax.set_yscale("log")
    y_low = max(float(np.min(lower_err)) * 0.85, 1e-12)
    y_high = float(np.max(upper_err)) * 1.30
    ax.set_ylim(y_low, y_high)
    ax.set_xlabel("k = # ADAPT-selected operators")
    ax.set_ylabel("mean |error vs CASCI| (Ha)")
    ax.set_title("Accuracy vs parameters (batch mean)")
    ax.grid(True, which="both", alpha=0.3)
    fig.tight_layout()
    fig.savefig(output_dir / "mean_abs_error_vs_k.png", dpi=150)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(8, 4.5))
    ax.plot(ks, mean_cost, marker="o", linewidth=2.0, color="tab:blue")
    lower_cost = _display_lower_band(mean_cost, std_cost)
    upper_cost = mean_cost + std_cost
    ax.fill_between(ks, lower_cost, upper_cost,
                    color="tab:blue", alpha=0.15)
    ax.set_yscale("log")
    ax.set_xlabel("k = # ADAPT-selected operators")
    ax.set_ylabel("mean cumulative cost evals")
    ax.set_title("ADAPT cost evaluations vs k (batch mean)")
    ax.grid(True, which="both", alpha=0.3)
    fig.tight_layout()
    fig.savefig(output_dir / "mean_cost_evals_vs_k.png", dpi=150)
    plt.close(fig)


def _plot_accuracy_batch_overview(
    batch_csv: Path,
    summary_csv: Path,
    output_dir: Path,
) -> None:
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except Exception as exc:  # pragma: no cover
        logger.warning("matplotlib unavailable: %s", exc)
        return

    batch = np.genfromtxt(batch_csv, delimiter=",", names=True, dtype=None, encoding=None)
    summary = np.genfromtxt(summary_csv, delimiter=",", names=True, dtype=None, encoding=None)
    if batch.size == 0 or summary.size == 0:
        return

    if batch.ndim == 0:
        batch = np.array([batch])
    if summary.ndim == 0:
        summary = np.array([summary])

    # Explicitly filter out failed runs so no numeric points from failed
    # rows are plotted. Some CSVs contain empty fields for failures which
    # can produce NaNs; build a mask that requires finite error and
    # status == 'ok' when available.
    ks = np.array(batch["k"], dtype=int)

    # Extract error_vs_casci safely from the structured array if present,
    # otherwise produce an empty array.
    if "error_vs_casci" in batch.dtype.names:
        with np.errstate(invalid="ignore"):
            abs_err = np.abs(np.array(batch["error_vs_casci"], dtype=float))
    else:
        abs_err = np.array([])

    status = None
    if "status" in batch.dtype.names:
        status = np.array(batch["status"]).astype(str)

    finite_mask = np.isfinite(abs_err) if abs_err.size else np.zeros(ks.shape, dtype=bool)
    if status is not None:
        ok_mask = finite_mask & (status == "ok")
    else:
        ok_mask = finite_mask

    if not np.any(ok_mask):
        return

    rng = np.random.default_rng(123)
    jitter = rng.uniform(-0.12, 0.12, size=np.count_nonzero(ok_mask))
    ks_ok = ks[ok_mask]
    err_ok = abs_err[ok_mask]

    ks_summary = np.array(summary["k"], dtype=int)
    mean_err = np.array(summary["mean_abs_error_vs_casci"], dtype=float)
    std_err = np.array(summary["std_abs_error_vs_casci"], dtype=float)

    fig, ax = plt.subplots(figsize=(8.5, 5.0))
    ax.scatter(
        ks_ok + jitter,
        np.where(err_ok < 1e-12, 1e-12, err_ok),
        s=20,
        alpha=0.55,
        color="tab:gray",
        edgecolors="none",
        label="molecule runs",
    )
    lower_err = _display_lower_band(mean_err, std_err)
    upper_err = mean_err + std_err
    ax.plot(ks_summary, mean_err, color="tab:red", linewidth=2.0, label="mean")
    ax.fill_between(
        ks_summary,
        lower_err,
        upper_err,
        color="tab:red",
        alpha=0.2,
        label="±1 std",
    )
    ax.set_yscale("log")
    y_low = max(float(np.min(lower_err)) * 0.85, 1e-12)
    y_high = float(np.max(upper_err)) * 1.30
    ax.set_ylim(y_low, y_high)
    ax.set_xlabel("k = # ADAPT-selected operators")
    ax.set_ylabel("|error vs CASCI| (Ha)")
    ax.set_title("Accuracy vs parameters (all molecules)")
    ax.grid(True, which="both", alpha=0.3)
    ax.legend(fontsize=9, loc="best")
    fig.tight_layout()
    fig.savefig(output_dir / "overview_all_molecules.png", dpi=200)
    plt.close(fig)


# ──────────────────────────────────────────────────────────────────────────
# CLI
# ──────────────────────────────────────────────────────────────────────────

def _build_argparser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description=(
            "Energy as a function of the number of ADAPT-VQE parameters."
        ),
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument("--molecule", "-m", default="water")
    p.add_argument("--batch-amino-acids", action="store_true",
                   help="Run all amino acids in datasets2/.")
    p.add_argument("--molecules", nargs="+", default=None,
                   help="Explicit molecule list from datasets2/ (overrides --batch-amino-acids).")
    p.add_argument("--basis", default="sto-3g")
    p.add_argument("--cs-target-qubits", type=int, default=0,
                   help="<=0 to skip CS reduction (recommended -- CS places "
                        "the start state at the noncontextual GS, which is a "
                        "stationary point of the simplified pool operators "
                        "and makes ADAPT terminate at k=0)")
    p.add_argument("--algorithm", default=DEFAULT_ALGORITHM,
                   choices=["adapt_vqe", "qubit_adapt_vqe"])
    p.add_argument("--k-list", nargs="+", type=int, default=list(DEFAULT_K_LIST),
                   help="Operator-count values at which to report energy. "
                        "Pass any list, e.g. '1 2 3 5 10 30 100'.")
    p.add_argument("--gradient-threshold", type=float,
                   default=DEFAULT_GRADIENT_THRESHOLD,
                   help="ADAPT exits early if max pool gradient < this value.")
    p.add_argument("--optimizer", default="COBYLA")
    p.add_argument("--inner-max-iter", type=int, default=200,
                   help="Per-step inner optimiser maxiter inside ADAPT")
    p.add_argument("--convergence-threshold", type=float, default=1e-10)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--no-plots", action="store_true")
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
        batch = True
    elif args.batch_amino_acids:
        molecules = DEFAULT_MOLECULES
        batch = True
    else:
        molecules = (args.molecule,)
        batch = False

    if batch:
        out = run_accuracy_vs_params_batch(
            molecules=molecules,
            basis=args.basis,
            cs_target_qubits=cs_target,
            algorithm=args.algorithm,
            k_list=tuple(args.k_list),
            gradient_threshold=args.gradient_threshold,
            optimizer=args.optimizer,
            inner_max_iter=args.inner_max_iter,
            convergence_threshold=args.convergence_threshold,
            random_seed=args.seed,
            save_plots=not args.no_plots,
        )
        cfg: AccuracyBatchConfig = out["config"]
        print("\n" + "=" * 60)
        print("ACCURACY vs N_PARAMETERS -- BATCH")
        print("=" * 60)
        print(f"Molecules: {len(cfg.molecules)}")
        print(f"k_list: {list(cfg.k_list)}")
        print(f"Batch CSV: {out['batch_csv']}")
        print(f"Summary CSV: {out['summary_csv']}")
        print(f"Output folder: {out['output_dir']}")
    else:
        out = run_accuracy_vs_params(
            molecule=args.molecule,
            basis=args.basis,
            cs_target_qubits=cs_target,
            algorithm=args.algorithm,
            k_list=tuple(args.k_list),
            gradient_threshold=args.gradient_threshold,
            optimizer=args.optimizer,
            inner_max_iter=args.inner_max_iter,
            convergence_threshold=args.convergence_threshold,
            random_seed=args.seed,
            save_plots=not args.no_plots,
        )

        cfg: AccuracyConfig = out["config"]
        rows: List[AccuracyRow] = out["rows"]

        print("\n" + "=" * 60)
        print(f"ACCURACY vs N_PARAMETERS -- {cfg.molecule} ({cfg.n_qubits_final} qubits)")
        print("=" * 60)
        print(f"HF energy:     {cfg.hf_energy}")
        print(f"CASCI energy:  {cfg.casci_energy}")
        print(f"Pool size:     {cfg.n_pool_operators} operators")
        print(f"Reached k:     {cfg.actual_max_k} operators")
        print()
        print(f"{'k':>4} {'energy':>14} {'err_vs_CASCI':>14} {'err_vs_HF':>14} "
              f"{'cum_cost_evals':>15} {'cum_runtime_s':>14}")
        print("-" * 80)
        for r in rows:
            print(f"{r.k:>4d} {r.energy:>14.6f} {r.error_vs_casci:>14.6f} "
                  f"{r.error_vs_hf:>14.6f} {r.cumulative_cost_evals:>15d} "
                  f"{r.cumulative_runtime_s:>14.2f}")

        print(f"\nResults saved to: {out['output_dir']}")


if __name__ == "__main__":
    main()
