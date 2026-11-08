# ABOUTME: Measures ADAPT energy and cumulative cost as the ansatz gains operators.
# ABOUTME: Reports the molecules reaching each requested size and their spread in accuracy.
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

With the default ``cs_target_qubits=None``, the ansatz starts from the
molecular HF determinant. Contextual reduction changes the operator and
initial state. The encoded reference can already be stationary for the
chosen pool, in which case ADAPT exits at k=0.

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

* ``k``, ``attained_k``         -- requested and attained operator counts.
* ``status``                 -- ``ok`` when the target was reached, otherwise ``stopped_early``.
* ``energy``                 -- ADAPT energy after ``attained_k`` operators.
* ``error_vs_casci``         -- ``energy - CASCI`` (positive when above CI).
* ``error_vs_hf``            -- ``energy - HF``     (negative when below HF,
  which is the variational improvement over Hartree-Fock).
* ``cumulative_cost_evals`` and ``cumulative_runtime_s`` -- work at the
  attained operator addition, or at termination if no operator was accepted.

Outputs
-------
``framework/experiments/results/accuracy_vs_params/<timestamp>_<molecule>_adapt_vqe/``

* ``run_config.json``                -- settings and total solver cost, including the final screen.
* ``results.csv``                    -- one row per k.
* ``full_convergence.csv``           -- each accepted generator, all refitted angles, energy and cumulative cost.
* ``energy_vs_n_params.png``         -- absolute |error vs CASCI|, linear below 1e-12 Ha.
* ``energy_vs_n_params_linear.png``  -- linear energy axis with HF/CASCI lines.

Batch Outputs
-------------
``framework/experiments/results/accuracy_vs_params_batch/<timestamp>_batch/``

* ``run_config.json``
* ``batch_all_results.csv``
* ``summary_by_k.csv`` -- mean and population SD across molecules reaching each k,
  with counts of requested, contributing, stopped and failed molecules.
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
from experiments.endpoint_sector import model_reference, summarize_endpoint  # noqa: E402

from experiments.provenance import record_runtime, save_problem
from experiments._common import (  # noqa: E402
    CostEvalCounter,
    LoadedHamiltonian,
    load_small_hamiltonian,
    make_backend_config,
    make_run_dir,
)
from experiments.noise_resilience import DATASETS2_AMINO_ACIDS  # noqa: E402

logger = logging.getLogger(__name__)


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
    attained_k: int
    status: str
    energy: float                  # ADAPT energy after attained_k operators
    error_vs_casci: float          # energy - CASCI
    error_vs_hf: float             # energy - HF (negative => improvement)
    cumulative_cost_evals: int     # total cost-fn calls to reach this k
    cumulative_runtime_s: float    # approximate runtime to reach this k
    rescored_energy: float
    wrong_electron_probability: Optional[float]
    error_vs_operator_ground: Optional[float] = None
    operator_ground_energy: Optional[float] = None


@dataclass
class AccuracyConfig:
    molecule: str
    basis: str
    cs_target_qubits: Optional[int]
    algorithm: str
    qubit_adapt_pool: str
    qubit_adapt_pool_effective: str
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
    encoded_reference_energy: Optional[float]
    n_pool_operators: int          # populated after the ADAPT run
    n_screenable_operators: int    # pool size after exact commutator screening
    actual_max_k: int              # ADAPT may exit early if pool exhausted
    total_cost_evals: int
    solver_runtime_s: float


@dataclass
class AccuracyBatchRow:
    molecule: str
    k: int
    attained_k: Optional[int]
    energy: float
    error_vs_casci: float
    error_vs_hf: float
    cumulative_cost_evals: int
    cumulative_runtime_s: float
    n_qubits_final: int
    n_pool_operators: int
    n_screenable_operators: int
    qubit_adapt_pool_effective: str
    actual_max_k: int
    status: str = "ok"
    notes: str = ""
    rescored_energy: Optional[float] = None
    wrong_electron_probability: Optional[float] = None
    error_vs_operator_ground: Optional[float] = None
    operator_ground_energy: Optional[float] = None


@dataclass
class AccuracyBatchConfig:
    molecules: Tuple[str, ...]
    basis: str
    cs_target_qubits: Optional[int]
    algorithm: str
    qubit_adapt_pool: str
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
    qubit_adapt_pool: str = "full",
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
    if qubit_adapt_pool not in ("full", "restricted"):
        raise ValueError("qubit_adapt_pool must be 'full' or 'restricted'")
    if algorithm != "qubit_adapt_vqe" and qubit_adapt_pool != "full":
        raise ValueError("qubit_adapt_pool only applies to qubit_adapt_vqe")
    max_k = k_sorted[-1]

    loaded = load_small_hamiltonian(
        molecule=molecule, basis=basis, cs_target_qubits=cs_target_qubits,
    )
    reference_summary = model_reference(loaded)

    out_dir = output_dir or make_run_dir(
        "accuracy_vs_params", molecule, algorithm,
    )
    record_runtime(out_dir)
    save_problem(loaded, Path(out_dir) / "inputs")
    logger.info("Output dir: %s", out_dir)

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
    algo_kwargs = ({"use_restricted_pool": qubit_adapt_pool == "restricted"}
                   if algorithm == "qubit_adapt_vqe" else {})
    counter = CostEvalCounter()
    per_op_cost_evals: List[int] = []
    per_op_timestamps: List[float] = []
    fitted_steps = []

    def record_fit(solver, energy):
        per_op_cost_evals.append(int(counter.count))
        per_op_timestamps.append(time.perf_counter() - t_start)
        fitted_steps.append((list(solver.selected_operators), solver.parameters.tolist()))

    vqe = AlgorithmCls(
        loaded.hamiltonian,
        on_operator_fit=record_fit,
        max_operators=max_k,
        gradient_threshold=gradient_threshold,
        optimizer=optimizer,
        max_iterations=inner_max_iter,
        convergence_threshold=convergence_threshold,
        random_seed=random_seed,
        backend_config=bc,
        **algo_kwargs,
    )

    counter.attach(vqe)
    t_start = time.perf_counter()
    try:
        result = vqe.run()
        runtime = time.perf_counter() - t_start
    finally:
        counter.detach(vqe)

    full_history = list(result.convergence_history)
    n_pool = len(vqe.operator_pool)
    n_screenable_operators = int(
        result.metadata.get("n_noncommuting_operators", 0)
    )
    effective_pool_mode = (
        str(result.metadata.get("operator_pool_mode", qubit_adapt_pool))
        if algorithm == "qubit_adapt_vqe"
        else "not_applicable"
    )
    actual_max_k = result.n_parameters
    diagnostics = [summarize_endpoint(loaded, vqe, angles, selected) for selected, angles in fitted_steps]
    if not actual_max_k:
        diagnostics.append(summarize_endpoint(loaded, vqe, [], []))
    for energy, state in zip(full_history or [result.calculated_energy], diagnostics, strict=True):
        if not np.isclose(energy, state["rescored_energy"], atol=1e-8, rtol=0):
            raise ValueError(f"{algorithm} milestone does not reproduce its reported energy")

    endpoint_diagnostics = diagnostics
    cfg = AccuracyConfig(
        molecule=molecule, basis=basis,
        cs_target_qubits=cs_target_qubits,
        algorithm=algorithm,
        qubit_adapt_pool=qubit_adapt_pool,
        qubit_adapt_pool_effective=effective_pool_mode,
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
        encoded_reference_energy=loaded.encoded_reference_energy,
        n_pool_operators=n_pool,
        n_screenable_operators=n_screenable_operators,
        actual_max_k=actual_max_k,
        total_cost_evals=counter.count,
        solver_runtime_s=runtime,
    )


    casci = loaded.casci_energy if loaded.casci_energy is not None else float("nan")
    hf = loaded.hf_energy if loaded.hf_energy is not None else float("nan")

    rows: List[AccuracyRow] = []
    for k in k_sorted:
        attained_k = min(k, actual_max_k)
        idx = attained_k - 1
        e = float(full_history[idx] if attained_k else result.calculated_energy)
        rows.append(AccuracyRow(
            k=k,
            attained_k=attained_k,
            status="ok" if k == attained_k else "stopped_early",
            energy=e,
            rescored_energy=diagnostics[idx]["rescored_energy"],
            wrong_electron_probability=diagnostics[idx]["wrong_electron_probability"],
            operator_ground_energy=reference_summary["operator_ground_energy"],
            error_vs_operator_ground=(None if reference_summary["operator_ground_energy"] is None
                                      else e - reference_summary["operator_ground_energy"]),
            error_vs_casci=e - casci,
            error_vs_hf=e - hf,
            cumulative_cost_evals=per_op_cost_evals[idx] if attained_k else counter.count,
            cumulative_runtime_s=per_op_timestamps[idx] if attained_k else runtime,
        ))

    _save_outputs(
        cfg, rows, full_history, per_op_cost_evals, per_op_timestamps, fitted_steps,
        vqe.operator_pool, diagnostics,
        out_dir, save_plots=save_plots,
        reference_summary=reference_summary,
        endpoint_diagnostics=endpoint_diagnostics,
    )

    return {
        "config": cfg,
        "rows": rows,
        "full_history": full_history,
        "per_step_cost_evals": per_op_cost_evals,
        "per_step_runtime": per_op_timestamps,
        "endpoint_diagnostics": endpoint_diagnostics,
        "model_reference": reference_summary,
        "output_dir": out_dir,
    }


def run_accuracy_vs_params_batch(
    molecules: Sequence[str] = DEFAULT_MOLECULES,
    basis: str = "sto-3g",
    cs_target_qubits: Optional[int] = None,
    algorithm: str = DEFAULT_ALGORITHM,
    qubit_adapt_pool: str = "full",
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
    if qubit_adapt_pool not in ("full", "restricted"):
        raise ValueError("qubit_adapt_pool must be 'full' or 'restricted'")
    if algorithm != "qubit_adapt_vqe" and qubit_adapt_pool != "full":
        raise ValueError("qubit_adapt_pool only applies to qubit_adapt_vqe")

    molecule_list = tuple(m.strip().lower() for m in molecules)
    if len(set(molecule_list)) != len(molecule_list):
        raise ValueError("Each molecule must be requested once per batch")

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
        qubit_adapt_pool=qubit_adapt_pool,
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
            "attained_k",
            "energy",
            "error_vs_casci",
            "error_vs_hf",
            "cumulative_cost_evals",
            "cumulative_runtime_s",
            "n_qubits_final",
            "n_pool_operators",
            "n_screenable_operators",
            "qubit_adapt_pool_effective",
            "actual_max_k",
            "status",
            "notes",
            "rescored_energy", "wrong_electron_probability", "error_vs_operator_ground", "operator_ground_energy",
        ])

        for mol in molecule_list:
            subdir = output_dir / f"{mol}_{algorithm}"
            try:
                out = run_accuracy_vs_params(
                    molecule=mol,
                    basis=basis,
                    cs_target_qubits=cs_target_qubits,
                    algorithm=algorithm,
                    qubit_adapt_pool=qubit_adapt_pool,
                    k_list=k_sorted,
                    gradient_threshold=gradient_threshold,
                    optimizer=optimizer,
                    inner_max_iter=inner_max_iter,
                    convergence_threshold=convergence_threshold,
                    random_seed=random_seed,
                    output_dir=subdir,
                    save_plots=save_plots,
                )
                mol_cfg: AccuracyConfig = out["config"]
                mol_rows: List[AccuracyRow] = out["rows"]
                for r in mol_rows:
                    row = AccuracyBatchRow(
                        molecule=mol,
                        k=int(r.k),
                        attained_k=r.attained_k,
                        energy=float(r.energy),
                        error_vs_casci=float(r.error_vs_casci),
                        error_vs_hf=float(r.error_vs_hf),
                        cumulative_cost_evals=int(r.cumulative_cost_evals),
                        cumulative_runtime_s=float(r.cumulative_runtime_s),
                        n_qubits_final=int(mol_cfg.n_qubits_final),
                        n_pool_operators=int(mol_cfg.n_pool_operators),
                        n_screenable_operators=int(mol_cfg.n_screenable_operators),
                        qubit_adapt_pool_effective=mol_cfg.qubit_adapt_pool_effective,
                        actual_max_k=int(mol_cfg.actual_max_k),
                        status=r.status,
                        notes="",
                        rescored_energy=r.rescored_energy,
                        wrong_electron_probability=r.wrong_electron_probability,
                        error_vs_operator_ground=r.error_vs_operator_ground,
                        operator_ground_energy=r.operator_ground_energy,
                    )
                    rows.append(row)
                    w.writerow([
                        row.molecule,
                        row.k,
                        row.attained_k,
                        row.energy,
                        row.error_vs_casci,
                        row.error_vs_hf,
                        row.cumulative_cost_evals,
                        f"{row.cumulative_runtime_s:.4f}",
                        row.n_qubits_final,
                        row.n_pool_operators,
                        row.n_screenable_operators,
                        row.qubit_adapt_pool_effective,
                        row.actual_max_k,
                        row.status,
                        row.notes,
                        row.rescored_energy, row.wrong_electron_probability, row.error_vs_operator_ground, row.operator_ground_energy,
                    ])
            except Exception as exc:
                logger.exception("Batch accuracy_vs_params failed for %s", mol)
                failures.append({"molecule": mol, "error": str(exc)})
                for k in k_sorted:
                    row = AccuracyBatchRow(
                        molecule=mol,
                        k=int(k),
                        attained_k=None,
                        energy=float("nan"),
                        error_vs_casci=float("nan"),
                        error_vs_hf=float("nan"),
                        cumulative_cost_evals=0,
                        cumulative_runtime_s=0.0,
                        n_qubits_final=0,
                        n_pool_operators=0,
                        n_screenable_operators=0,
                        qubit_adapt_pool_effective="unknown",
                        actual_max_k=0,
                        status="failed",
                        notes=str(exc),
                    )
                    rows.append(row)
                    w.writerow([
                        row.molecule,
                        row.k,
                        row.attained_k,
                        "",
                        "",
                        "",
                        row.cumulative_cost_evals,
                        f"{row.cumulative_runtime_s:.4f}",
                        row.n_qubits_final,
                        row.n_pool_operators,
                        row.n_screenable_operators,
                        row.qubit_adapt_pool_effective,
                        row.actual_max_k,
                        row.status,
                        row.notes,
                        row.rescored_energy, row.wrong_electron_probability, row.error_vs_operator_ground, row.operator_ground_energy,
                    ])

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
    fitted_steps: list,
    operator_pool: list,
    diagnostics: list,
    out_dir: Path,
    save_plots: bool = True,
    reference_summary: Optional[dict] = None,
    endpoint_diagnostics: Optional[List[dict]] = None,
) -> None:
    import csv

    out_dir.mkdir(parents=True, exist_ok=True)

    cfg_dict = asdict(cfg)
    cfg_dict["k_list"] = list(cfg.k_list)
    with open(out_dir / "run_config.json", "w") as f:
        json.dump(cfg_dict, f, indent=2)
    if reference_summary is not None:
        with open(out_dir / "model_reference.json", "w") as f:
            json.dump(reference_summary, f, indent=2, allow_nan=False)
    if endpoint_diagnostics is not None:
        with open(out_dir / "endpoint_diagnostics.json", "w") as f:
            json.dump(endpoint_diagnostics, f, indent=2, allow_nan=False)

    with open(out_dir / "results.csv", "w", newline="") as f:
        w = csv.writer(f)
        w.writerow([
            "k", "attained_k", "status", "energy", "error_vs_casci", "error_vs_hf",
            "cumulative_cost_evals", "cumulative_runtime_s", "rescored_energy", "wrong_electron_probability",
            "error_vs_operator_ground", "operator_ground_energy",
        ])
        for r in rows:
            w.writerow([
                r.k, r.attained_k, r.status, r.energy, r.error_vs_casci,
                r.error_vs_hf, r.cumulative_cost_evals,
                f"{r.cumulative_runtime_s:.4f}", r.rescored_energy, r.wrong_electron_probability,
                r.error_vs_operator_ground, r.operator_ground_energy,
            ])

    # Save the full ADAPT trace at every operator addition (not just at k_list)
    with open(out_dir / "full_convergence.csv", "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["k", "energy", "cumulative_cost_evals", "cumulative_runtime_s", "added_generator", "parameters",
                    "rescored_energy", "wrong_electron_probability"])
        for k, (e, cev, ts, (selected, angles)) in enumerate(
            zip(full_history, per_op_cost_evals, per_op_timestamps, fitted_steps, strict=True), start=1,
        ):
            state = diagnostics[k - 1]
            w.writerow([k, e, cev, f"{ts:.4f}", json.dumps(operator_pool[selected[-1]]), json.dumps(angles),
                        state["rescored_energy"], state["wrong_electron_probability"]])

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
    reached = [r for r in rows if r.status == "ok"]
    ax.plot(full_xs, full_history, color="tab:blue", linewidth=1, alpha=0.5,
            label="ADAPT (every operator)")
    ax.plot(
        [r.k for r in reached], [r.energy for r in reached],
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
        f"Exp 4 - Accuracy vs ADAPT operator count (ADAPT-VQE)\n"
        f"{cfg.molecule}, {cfg.n_qubits_final} qubits, "
        f"pool={cfg.n_pool_operators}, screenable={cfg.n_screenable_operators} "
        f"pool mode requested={cfg.qubit_adapt_pool}, effective="
        f"{cfg.qubit_adapt_pool_effective}"
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
        ax.plot(full_xs, full_err, color="tab:red", linewidth=1, alpha=0.5,
                label="ADAPT (every operator)")
        ax.plot(
            [r.k for r in reached], [abs(r.error_vs_casci) for r in reached],
            marker="o", linestyle="", color="tab:red",
            label="reported k values",
        )
        # Chemical accuracy line: 1.6 mHa
        ax.axhline(1.6e-3, color="green", linestyle="--", linewidth=1,
                   label="chemical accuracy = 1.6 mHa")
        ax.set_yscale("symlog", linthresh=1e-12)
        ax.set_xlabel("k = # ADAPT-selected operators (= # parameters)")
        ax.set_ylabel(r"$|E_\mathrm{ADAPT}(k) - E_\mathrm{CASCI}|$ (Ha)")
        ax.set_title(
            f"Exp 4 - Energy error vs ADAPT operator count (ADAPT-VQE)\n"
            f"{cfg.molecule}, {cfg.n_qubits_final} qubits, "
            f"pool={cfg.n_pool_operators}, screenable={cfg.n_screenable_operators} "
            f"pool mode requested={cfg.qubit_adapt_pool}, effective="
            f"{cfg.qubit_adapt_pool_effective}"
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
            f"Exp 4 - ADAPT cost evaluations vs operator count\n"
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
        grouped.setdefault(row.k, []).append(row)

    with open(path, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow([
            "k",
            "n_molecules",
            "n_requested",
            "n_stopped_early",
            "n_failed",
            "mean_abs_error_vs_casci",
            "std_abs_error_vs_casci",
            "mean_cost_evals",
            "std_cost_evals",
        ])
        for k in sorted(grouped.keys()):
            requested = grouped[k]
            group = [r for r in requested if r.status == "ok"]
            abs_err = np.array([abs(r.error_vs_casci) for r in group], dtype=float)
            cost_evals = np.array([r.cumulative_cost_evals for r in group], dtype=float)
            w.writerow([
                k,
                len(group),
                len(requested),
                sum(r.status == "stopped_early" for r in requested),
                sum(r.status == "failed" for r in requested),
                np.mean(abs_err) if group else "",
                np.std(abs_err) if group else "",
                np.mean(cost_evals) if group else "",
                np.std(cost_evals) if group else "",
            ])


def _plot_cohort_mean(ax, summary: np.ndarray, field: str, color: str) -> None:
    """Show mean and population SD for the molecules attaining each requested count."""
    ks = summary["k"].astype(int)
    mean = summary[f"mean_{field}"]
    std = summary[f"std_{field}"]
    ax.errorbar(ks, mean, yerr=std, marker="o", capsize=4, linewidth=2.0,
                color=color, label="mean ±1 SD across molecules")
    counts = zip(ks, summary["n_molecules"].astype(int), summary["n_requested"].astype(int))
    ax.set_xticks(ks, [f"{k}\n({reached}/{requested})" for k, reached, requested in counts])
    ax.set_xlabel("# ADAPT-selected operators (molecules reaching k / requested)")
    ax.legend(fontsize=9)


def _plot_accuracy_batch_summary(summary_csv: Path, output_dir: Path) -> None:
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except Exception as exc:  # pragma: no cover
        logger.warning("matplotlib unavailable: %s", exc)
        return

    data = np.genfromtxt(summary_csv, delimiter=",", names=True, ndmin=1)
    if not data.size:
        return

    fig, ax = plt.subplots(figsize=(8, 4.5))
    _plot_cohort_mean(ax, data, "abs_error_vs_casci", "tab:red")
    ax.set_ylabel("mean |error vs CASCI| (Ha)")
    ax.set_title("Exp 4 - Accuracy vs ADAPT operator count (batch mean)")
    ax.grid(True, which="both", alpha=0.3)
    fig.tight_layout()
    fig.savefig(output_dir / "mean_abs_error_vs_k.png", dpi=150)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(8, 4.5))
    _plot_cohort_mean(ax, data, "cost_evals", "tab:blue")
    ax.set_ylabel("mean cumulative cost evals")
    ax.set_title("Exp 4 - ADAPT cost evaluations vs operator count (batch mean)")
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

    import csv

    with batch_csv.open(newline="") as stream:
        reached = [row for row in csv.DictReader(stream) if row["status"] == "ok"]
    summary = np.genfromtxt(summary_csv, delimiter=",", names=True, ndmin=1)
    if not summary.size:
        return
    ks = np.array([int(row["k"]) for row in reached])
    errors = np.array([abs(float(row["error_vs_casci"])) for row in reached])
    rng = np.random.default_rng(123)
    jitter = rng.uniform(-0.12, 0.12, size=len(reached))

    fig, ax = plt.subplots(figsize=(8.5, 5.0))
    ax.scatter(
        ks + jitter,
        errors,
        s=20,
        alpha=0.55,
        color="tab:gray",
        edgecolors="none",
        label="molecule runs",
    )
    _plot_cohort_mean(ax, summary, "abs_error_vs_casci", "tab:red")
    ax.set_ylabel("|error vs CASCI| (Ha)")
    ax.set_title("Exp 4 - Accuracy vs ADAPT operator count (all molecules)")
    ax.grid(True, which="both", alpha=0.3)
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
    p.add_argument("--output-dir", type=Path, default=None)
    p.add_argument("--batch-amino-acids", action="store_true",
                   help="Run all amino acids in datasets2/.")
    p.add_argument("--molecules", nargs="+", default=None,
                   help="Explicit molecule list from datasets2/ (overrides --batch-amino-acids).")
    p.add_argument("--basis", default="sto-3g")
    p.add_argument("--cs-target-qubits", type=int, default=0,
                   help="<=0 to retain the molecular HF starting determinant. "
                        "With CS the projected reference can already be stationary for the pool.")
    p.add_argument("--algorithm", default=DEFAULT_ALGORITHM,
                   choices=["adapt_vqe", "qubit_adapt_vqe"])
    p.add_argument("--qubit-adapt-pool", choices=["full", "restricted"],
                   default="full",
                   help="Qubit-ADAPT pool: all supported odd-Y Pauli strings, "
                        "or occupied-to-virtual singles/doubles.")
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
            qubit_adapt_pool=args.qubit_adapt_pool,
            k_list=tuple(args.k_list),
            gradient_threshold=args.gradient_threshold,
            optimizer=args.optimizer,
            inner_max_iter=args.inner_max_iter,
            convergence_threshold=args.convergence_threshold,
            random_seed=args.seed,
            output_dir=args.output_dir,
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
        if out["failures"]:
            raise SystemExit(1)
    else:
        out = run_accuracy_vs_params(
            molecule=args.molecule,
            basis=args.basis,
            cs_target_qubits=cs_target,
            algorithm=args.algorithm,
            qubit_adapt_pool=args.qubit_adapt_pool,
            k_list=tuple(args.k_list),
            gradient_threshold=args.gradient_threshold,
            optimizer=args.optimizer,
            inner_max_iter=args.inner_max_iter,
            convergence_threshold=args.convergence_threshold,
            random_seed=args.seed,
            output_dir=args.output_dir,
            save_plots=not args.no_plots,
        )

        cfg: AccuracyConfig = out["config"]
        rows: List[AccuracyRow] = out["rows"]

        print("\n" + "=" * 60)
        print(f"ACCURACY vs N_PARAMETERS -- {cfg.molecule} ({cfg.n_qubits_final} qubits)")
        print("=" * 60)
        print(f"HF energy:     {cfg.hf_energy}")
        print(f"CASCI energy:  {cfg.casci_energy}")
        print(f"Pool mode:     {cfg.qubit_adapt_pool}")
        print(f"Effective pool: {cfg.qubit_adapt_pool_effective}")
        print(f"Generated pool: {cfg.n_pool_operators} operators")
        print(f"After commutator screen: {cfg.n_screenable_operators} operators")
        print(f"Reached k:     {cfg.actual_max_k} operators")
        print()
        print(f"{'k':>4} {'attained':>8} {'status':>14} {'energy':>14} {'err_vs_CASCI':>14} {'err_vs_HF':>14} "
              f"{'cum_cost_evals':>15} {'cum_runtime_s':>14}")
        print("-" * 80)
        for r in rows:
            print(f"{r.k:>4d} {r.attained_k:>8d} {r.status:>14} {r.energy:>14.6f} {r.error_vs_casci:>14.6f} "
                  f"{r.error_vs_hf:>14.6f} {r.cumulative_cost_evals:>15d} "
                  f"{r.cumulative_runtime_s:>14.2f}")

        print(f"\nResults saved to: {out['output_dir']}")


if __name__ == "__main__":
    main()
