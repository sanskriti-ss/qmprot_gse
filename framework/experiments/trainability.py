"""
Experiment 3 -- Trainability at matched parameter counts
========================================================

Question
--------
For a fixed VQE problem, how does the trainability of an *adaptive*
ansatz (ADAPT-VQE) compare to that of a *fixed* hardware-efficient
ansatz when both ansatze use the same number of variational parameters?

We pre-choose target parameter counts (default ``10, 50, 100``) and run:

* **Qubit-ADAPT-VQE** with ``max_operators = target_n_params``.  Grows
  the ansatz one Pauli rotation at a time, picking the operator with
  the largest energy gradient from the qubit pool.  We default to
  qubit-ADAPT (rather than the simpler ``adapt_vqe`` with RY+CNOT
  approximations) because the simplified gates give zero finite-
  difference gradient at theta=0 by parameterisation symmetry.
* **HardwareEfficientVQE** at the smallest ``n_layers`` that yields
  ``>= target_n_params`` parameters.  Defaults to RY-only rotations so
  the parameter count grows linearly with depth.

We also default to ``cs_target_qubits=None`` (no contextual-subspace
reduction) because CS places the start state at the noncontextual
ground state, which is a stationary point of the operator pool and
makes ADAPT exit at k=0.

Comparison metric
-----------------
"How long did it take to train?" is measured as the **number of
cost-function evaluations needed to first reach a fixed accuracy** (error
vs CASCI of 100 mHa, 10 mHa and chemical accuracy 1.6 mHa), not as the
total spent by a run.  Comparing totals is misleading: a run that simply
hits its iteration cap while still ~1 Ha above CASCI looks "cheap".

Every evaluation's energy is recorded by
:class:`experiments._common.CostEvalCounter`, so the first evaluation at
which ``E - E_CASCI <= threshold`` is read off exactly.  A run that never
reaches a threshold is *censored*: it is reported as ``reached=False`` with
its spent budget as a lower bound (``> N evals``), never as a cost.

Counting rules: ``n_cost_evals`` = energy evaluations requested by the
optimiser / ADAPT gradient screening (the quantity hardware would pay for).
The bookkeeping re-evaluation done by ``BaseVQE.callback`` is excluded
(reported separately as ``n_callback_evals``).

Parameter-count matching
------------------------
ADAPT grows exactly one parameter per operator, so ``target_n_params`` is
met exactly unless it stops early (``param_match=stopped_early``: pool
exhausted / gradient below threshold).  Hardware-efficient VQE normally
has ``n_qubits * rotations * (layers + 1)`` parameters, which rarely equals
the target; by default we now freeze the surplus rotation slots at 0
(``--he-param-match exact``) so the variational parameter count equals the
target.  ``--he-param-match round_up`` restores the old behaviour
(``param_match=rounded_up``).  Both ``target_n_params`` and
``actual_n_params`` are written to every CSV row and plot label.

What we measure (per (target_n_params, algorithm) pair)
------------------------------------------------------

* ``actual_n_params``  -- the realised parameter count (+ ``param_match``).
* ``n_outer_iters``    -- algorithm-reported iterations (operators for
  ADAPT, COBYLA outer iters for HW-efficient).
* ``n_cost_evals``     -- the *true* number of optimiser-requested evaluations.
* ``evals_to_<thr>``   -- evaluations to first reach ``thr`` (blank = never).
* ``runtime_seconds``  -- wall time.
* ``final_energy``     -- last energy on convergence trace.
* ``error_vs_casci``   -- ``final_energy - CASCI`` (positive when above CI).
* ``hf_error_vs_casci`` / ``hf_within_chem_acc`` -- the free HF baseline; if
  HF already meets chemical accuracy the molecule cannot rank ansaetze.
* ``convergence_history`` -- the algorithm's own convergence trace
  (interpretation differs: ADAPT logs energy after each operator
  addition; HW-efficient logs energy after each COBYLA outer iteration).

Outputs
-------
Single-molecule mode:
``framework/experiments/results/trainability/<timestamp>_<molecule>_comparison/``

* ``run_config.json``
* ``results.csv``                  -- one row per (target, algorithm).
* ``convergence_traces.csv``       -- long-format energy traces.
* ``eval_traces.csv``              -- best-error-so-far vs evaluation index.
* ``cost_to_accuracy.png``         -- evals to reach each accuracy (censored runs marked).
* ``error_vs_cost.png``            -- accuracy-vs-cost curves.
* ``cost_evals_vs_params.png``     -- bars of n_cost_evals per algorithm.
* ``final_energy_vs_params.png``   -- final energy reached per algorithm.
* ``runtime_vs_params.png``        -- wall time per algorithm.
* ``cost_to_accuracy_report.md``   -- plain-language summary table.

All-molecules mode (``--all-molecules``):
``framework/experiments/results/trainability/<timestamp>_all_molecules_comparison/``

* one per-molecule subfolder with the same outputs as above
* ``all_molecules_results.csv``
* ``all_molecules_convergence_traces.csv``
* ``all_molecules_cost_evals_vs_params.png`` -- one chart with all molecules
* ``all_molecules_cost_to_accuracy.png`` / ``..._summary.csv`` / ``cost_to_accuracy_report.md``
* ``batch_manifest.json``

Run as a script
---------------
::

    python -m experiments.trainability \\
        --molecule water --param-targets 10 50 100 --he-eval-budget 3000

    python -m experiments.trainability \
        --all-molecules --param-targets 10 50 100 --he-eval-budget 3000

    # Re-analyse an older run (final-energy-only data -> bounds):
    python -m experiments.trainability --analyze-existing <results_dir>
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
import time
from dataclasses import asdict, dataclass, field, replace
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np

_FW_DIR = Path(__file__).resolve().parent.parent
if str(_FW_DIR) not in sys.path:
    sys.path.insert(0, str(_FW_DIR))

from algorithms import get_algorithm  # noqa: E402

from experiments import cost_to_accuracy as cta  # noqa: E402
from experiments._common import (  # noqa: E402
    ACCURACY_LADDER_HA,
    AccuracyReached,
    CHEMICAL_ACCURACY_HA,
    CostEvalCounter,
    DATASETS2_DIR,
    LoadedHamiltonian,
    he_layers_for_target_params,
    load_small_hamiltonian,
    make_backend_config,
    make_run_dir,
    sample_initial_parameters,
    threshold_label,
)

logger = logging.getLogger(__name__)


DEFAULT_PARAM_TARGETS: Tuple[int, ...] = (10, 50, 100)
DEFAULT_ALGORITHMS: Tuple[str, ...] = ("qubit_adapt_vqe", "hardware_efficient_vqe")
DEFAULT_GRADIENT_THRESHOLD: float = 1e-8
# COBYLA cannot even build its initial simplex with fewer than n+1 evals.
HE_MIN_EVALS_PER_PARAM: int = 5
DEFAULT_HE_EVAL_BUDGET: int = 2000


# ──────────────────────────────────────────────────────────────────────────
# Result containers
# ──────────────────────────────────────────────────────────────────────────

@dataclass
class TrainabilityRow:
    algorithm: str
    target_n_params: int
    actual_n_params: int
    n_outer_iters: int          # algorithm-reported iterations
    n_cost_evals: int           # optimiser-requested energy evaluations
    runtime_seconds: float
    final_energy: float
    error_vs_casci: float
    converged_below_threshold: bool
    convergence_history: List[float] = field(default_factory=list)
    notes: str = ""
    # -- parameter matching ------------------------------------------------
    param_match: str = ""             # exact | rounded_up | stopped_early
    # -- cost to reach a fixed accuracy -------------------------------------
    n_callback_evals: int = 0         # bookkeeping evals, NOT in n_cost_evals
    eval_budget: int = 0              # evaluation cap given to the optimiser (0 = none)
    best_error_vs_casci: float = float("nan")   # lowest energy seen - CASCI
    hf_error_vs_casci: float = float("nan")     # E_HF - E_CASCI
    # threshold label (e.g. '1.6mHa') -> first eval reaching it, None = never
    evals_to_threshold: Dict[str, Optional[int]] = field(default_factory=dict)
    # (eval_index, best_error_vs_casci) recorded whenever the best improves
    error_trace: List[Tuple[int, float]] = field(default_factory=list)
    stopped_at_accuracy: bool = False
    init_label: str = ""
    # True for rows rebuilt from final-energy-only data: ``evals_to_threshold``
    # values are then UPPER bounds (run total) rather than exact first hits.
    bounds_only: bool = False


@dataclass
class TrainabilityConfig:
    molecule: str
    basis: str
    cs_target_qubits: Optional[int]
    algorithms: Tuple[str, ...]
    param_targets: Tuple[int, ...]
    optimizer: str
    max_iterations: int
    convergence_threshold: float
    adapt_gradient_threshold: float
    qubit_adapt_max_gradient_candidates: Optional[int]
    he_rotation_gates: str
    random_seed: int
    n_qubits_active: int
    n_qubits_final: int
    casci_energy: Optional[float]
    hf_energy: Optional[float]
    he_param_match: str = "exact"
    he_eval_budget: int = DEFAULT_HE_EVAL_BUDGET
    he_init_strategy: str = "random_uniform"
    he_hf_init: bool = False
    accuracy_thresholds_ha: Tuple[float, ...] = ACCURACY_LADDER_HA


def _fill_accuracy_metrics(
    row: TrainabilityRow,
    counter: CostEvalCounter,
    loaded: LoadedHamiltonian,
    thresholds: Sequence[float] = ACCURACY_LADDER_HA,
) -> None:
    """Populate the cost-to-accuracy fields of ``row`` from a finished counter."""
    casci = loaded.casci_energy
    hf = loaded.hf_energy
    row.n_cost_evals = int(counter.count)
    row.n_callback_evals = int(counter.callback_evals)
    if hf is not None and casci is not None:
        row.hf_error_vs_casci = float(hf - casci)
    if casci is None or not counter.energies:
        return
    rm = counter.running_min()
    row.best_error_vs_casci = float(rm[-1] - casci)
    improved = np.flatnonzero(np.r_[True, rm[1:] < rm[:-1]])
    trace = [(int(i) + 1, float(rm[i] - casci)) for i in improved]
    if trace[-1][0] != len(rm):
        trace.append((len(rm), float(rm[-1] - casci)))
    row.error_trace = trace
    row.evals_to_threshold = {
        threshold_label(t): counter.evals_to_reach(casci + t) for t in thresholds
    }


# ──────────────────────────────────────────────────────────────────────────
# Per-algorithm runners
# ──────────────────────────────────────────────────────────────────────────

def _run_adapt_at_target(
    loaded: LoadedHamiltonian,
    *,
    algorithm: str,
    target_n_params: int,
    gradient_threshold: float,
    qubit_adapt_max_gradient_candidates: Optional[int],
    optimizer: str,
    max_iterations: int,
    convergence_threshold: float,
    random_seed: int,
) -> TrainabilityRow:
    """Run an adaptive VQE (``adapt_vqe`` or ``qubit_adapt_vqe``) capped
    at ``target_n_params`` operators.

    If the pool is smaller than ``target_n_params`` or gradients fall
    below ``gradient_threshold``, ADAPT exits early and the row records
    the actual parameter count.
    """
    if algorithm not in ("adapt_vqe", "qubit_adapt_vqe"):
        raise ValueError(
            f"_run_adapt_at_target only supports 'adapt_vqe' or "
            f"'qubit_adapt_vqe'; got {algorithm!r}"
        )
    AlgorithmCls = get_algorithm(algorithm)
    bc = make_backend_config(loaded.hamiltonian.n_qubits)
    algo_kwargs = {}
    if algorithm == "qubit_adapt_vqe" and qubit_adapt_max_gradient_candidates is not None:
        algo_kwargs["max_gradient_candidates"] = qubit_adapt_max_gradient_candidates

    vqe = AlgorithmCls(
        loaded.hamiltonian,
        max_operators=target_n_params,
        gradient_threshold=gradient_threshold,
        optimizer=optimizer,
        max_iterations=max_iterations,
        convergence_threshold=convergence_threshold,
        random_seed=random_seed,
        backend_config=bc,
        **algo_kwargs,
    )

    counter = CostEvalCounter()
    counter.attach(vqe)
    t0 = time.time()
    try:
        result = vqe.run()
    finally:
        counter.detach(vqe)
    runtime = time.time() - t0

    history = list(result.convergence_history)
    final_e = float(history[-1]) if history else float(result.calculated_energy)
    casci = loaded.casci_energy if loaded.casci_energy is not None else float("nan")

    notes = ""
    if result.n_parameters < target_n_params:
        notes = (
            f"{algorithm} terminated early at {result.n_parameters} ops "
            f"(target={target_n_params}); pool exhausted or gradient below threshold"
        )
        logger.info(notes)

    row = TrainabilityRow(
        algorithm=algorithm,
        target_n_params=target_n_params,
        actual_n_params=int(result.n_parameters),
        n_outer_iters=int(result.n_iterations),
        n_cost_evals=int(counter.count),
        runtime_seconds=runtime,
        final_energy=final_e,
        error_vs_casci=final_e - casci,
        converged_below_threshold=bool(result.converged),
        convergence_history=history,
        notes=notes,
        param_match=(
            "exact" if int(result.n_parameters) == target_n_params else "stopped_early"
        ),
        init_label="HF reference",
    )
    _fill_accuracy_metrics(row, counter, loaded)
    return row


def _run_he_at_target(
    loaded: LoadedHamiltonian,
    *,
    target_n_params: int,
    optimizer: str,
    max_iterations: int,
    convergence_threshold: float,
    random_seed: int,
    rotation_gates: str = "RY",
    param_match: str = "exact",
    eval_budget: int = DEFAULT_HE_EVAL_BUDGET,
    init_strategy: str = "random_uniform",
    hf_init: bool = False,
    stop_at_accuracy: bool = True,
) -> TrainabilityRow:
    """Run HardwareEfficientVQE with ``target_n_params`` parameters.

    ``param_match='exact'`` freezes the surplus rotation slots of the
    smallest sufficient depth so the variational parameter count equals the
    target; ``'round_up'`` keeps every slot (legacy behaviour).

    The optimiser gets ``eval_budget`` energy evaluations (raised to
    ``HE_MIN_EVALS_PER_PARAM * (n_params + 1)`` if smaller, since COBYLA
    needs ``n_params + 1`` evaluations just to build its starting simplex).
    With ``stop_at_accuracy`` the run ends as soon as chemical accuracy vs
    CASCI is reached, so the recorded cost is the cost *to reach it*.
    """
    if param_match not in ("exact", "round_up"):
        raise ValueError(f"param_match must be 'exact' or 'round_up', got {param_match!r}")
    n_layers, slots = he_layers_for_target_params(
        loaded.hamiltonian.n_qubits,
        target_n_params,
        rotation_gates=rotation_gates,
        minimum_layers=1,  # always one entangling layer minimum
    )
    exact = param_match == "exact"
    n_free = target_n_params if exact else slots

    min_budget = HE_MIN_EVALS_PER_PARAM * (n_free + 1)
    budget = max(int(eval_budget), min_budget)
    budget_note = ""
    if budget != eval_budget:
        budget_note = (
            f"; eval_budget raised {eval_budget}->{budget} (COBYLA needs >= "
            f"{n_free + 1} evals for its initial simplex)"
        )
        logger.warning("HE target=%d: %s", target_n_params, budget_note.lstrip("; "))

    AlgorithmCls = get_algorithm("hardware_efficient_vqe")
    bc = make_backend_config(loaded.hamiltonian.n_qubits)
    vqe = AlgorithmCls(
        loaded.hamiltonian,
        n_layers=n_layers,
        rotation_gates=rotation_gates,
        max_free_parameters=target_n_params if exact else None,
        prepare_hf=hf_init,
        optimizer=optimizer,
        max_iterations=budget,
        convergence_threshold=convergence_threshold,
        random_seed=random_seed,
        backend_config=bc,
    )
    vqe.build_ansatz()
    theta0 = sample_initial_parameters(
        int(vqe.n_parameters), strategy=init_strategy,
        rng=np.random.default_rng(random_seed),
    )

    casci = loaded.casci_energy
    stop_below = (
        casci + CHEMICAL_ACCURACY_HA if (stop_at_accuracy and casci is not None) else None
    )
    counter = CostEvalCounter(stop_below=stop_below)
    counter.attach(vqe)
    stopped = False
    t0 = time.time()
    try:
        vqe.optimize(initial_parameters=theta0)
    except AccuracyReached:
        stopped = True
    finally:
        counter.detach(vqe)
    runtime = time.time() - t0

    history = list(vqe.convergence_history)
    if stopped or not history:
        final_e = float(counter.energies[-1])
    else:
        final_e = float(history[-1])
    casci_f = casci if casci is not None else float("nan")

    notes = (
        f"n_layers={n_layers}, rotation_gates={rotation_gates}, "
        f"actual_n_params={vqe.n_parameters} (target={target_n_params}, "
        f"{slots} rotation slots), eval_budget={budget}{budget_note}"
    )
    if stopped:
        notes += "; stopped at chemical accuracy"

    row = TrainabilityRow(
        algorithm="hardware_efficient_vqe",
        target_n_params=target_n_params,
        actual_n_params=int(vqe.n_parameters),
        n_outer_iters=int(vqe.iteration_count),
        n_cost_evals=int(counter.count),
        runtime_seconds=runtime,
        final_energy=final_e,
        error_vs_casci=final_e - casci_f,
        converged_below_threshold=stopped,
        convergence_history=history,
        notes=notes,
        param_match="exact" if int(vqe.n_parameters) == target_n_params else "rounded_up",
        eval_budget=budget,
        stopped_at_accuracy=stopped,
        init_label=f"{init_strategy}{'+HF' if hf_init else ''}",
    )
    _fill_accuracy_metrics(row, counter, loaded)
    return row


# ──────────────────────────────────────────────────────────────────────────
# Public API
# ──────────────────────────────────────────────────────────────────────────

def run_trainability(
    molecule: str = "water",
    basis: str = "sto-3g",
    cs_target_qubits: Optional[int] = None,
    algorithms: Tuple[str, ...] = DEFAULT_ALGORITHMS,
    param_targets: Tuple[int, ...] = DEFAULT_PARAM_TARGETS,
    adapt_gradient_threshold: float = DEFAULT_GRADIENT_THRESHOLD,
    qubit_adapt_max_gradient_candidates: Optional[int] = None,
    optimizer: str = "COBYLA",
    max_iterations: int = 500,
    convergence_threshold: float = 1e-10,
    he_rotation_gates: str = "RY",
    random_seed: int = 42,
    output_dir: Optional[Path] = None,
    save_plots: bool = True,
    he_param_match: str = "exact",
    he_eval_budget: int = DEFAULT_HE_EVAL_BUDGET,
    he_init_strategy: str = "random_uniform",
    he_hf_init: bool = False,
    he_stop_at_accuracy: bool = True,
) -> Dict[str, object]:
    """Run the full trainability comparison.

    ``he_eval_budget`` is the evaluation cap for hardware-efficient runs;
    ``0`` means "give HE the same number of evaluations ADAPT used at the
    same target" (equal-cost comparison).  See the module docstring for
    ``he_param_match`` and the cost-to-accuracy metric.

    Returns a dict with keys ``"config"``, ``"rows"``, ``"output_dir"``.
    """
    loaded = load_small_hamiltonian(
        molecule=molecule, basis=basis, cs_target_qubits=cs_target_qubits,
    )

    cfg = TrainabilityConfig(
        molecule=molecule, basis=basis,
        cs_target_qubits=cs_target_qubits,
        algorithms=tuple(algorithms),
        param_targets=tuple(param_targets),
        optimizer=optimizer,
        max_iterations=max_iterations,
        convergence_threshold=convergence_threshold,
        adapt_gradient_threshold=adapt_gradient_threshold,
        qubit_adapt_max_gradient_candidates=qubit_adapt_max_gradient_candidates,
        he_rotation_gates=he_rotation_gates,
        random_seed=random_seed,
        n_qubits_active=loaded.n_qubits_active,
        n_qubits_final=loaded.n_qubits_final,
        casci_energy=loaded.casci_energy,
        hf_energy=loaded.hf_energy,
        he_param_match=he_param_match,
        he_eval_budget=he_eval_budget,
        he_init_strategy=he_init_strategy,
        he_hf_init=he_hf_init,
    )

    out_dir = output_dir or make_run_dir(
        "trainability", molecule, "comparison",
    )
    logger.info("Output dir: %s", out_dir)

    # Only relevant for ``he_param_match='round_up'``: different targets can
    # round up to the same ansatz (e.g. 5 and 10 both -> 24 parameters), so
    # HE rows are cached by actual_n_params and re-used instead of re-run.
    he_cache: Dict[int, TrainabilityRow] = {}
    adapt_evals_at_target: Dict[int, int] = {}

    rows: List[TrainabilityRow] = []
    for target in param_targets:
        for algo in algorithms:
            logger.info(
                "[trainability] target=%d  algo=%s", target, algo,
            )
            try:
                if algo in ("adapt_vqe", "qubit_adapt_vqe"):
                    row = _run_adapt_at_target(
                        loaded,
                        algorithm=algo,
                        target_n_params=target,
                        gradient_threshold=adapt_gradient_threshold,
                        qubit_adapt_max_gradient_candidates=qubit_adapt_max_gradient_candidates,
                        optimizer=optimizer,
                        max_iterations=max_iterations,
                        convergence_threshold=convergence_threshold,
                        random_seed=random_seed,
                    )
                    adapt_evals_at_target[target] = row.n_cost_evals
                elif algo == "hardware_efficient_vqe":
                    n_layers, actual = he_layers_for_target_params(
                        loaded.hamiltonian.n_qubits, target,
                        rotation_gates=he_rotation_gates, minimum_layers=1,
                    )
                    if he_param_match == "round_up" and actual in he_cache:
                        prior = he_cache[actual]
                        logger.info(
                            "HE at target=%d would round to actual=%d, "
                            "matching a previous target=%d run -- reusing "
                            "those numbers and noting the duplicate.",
                            target, actual, prior.target_n_params,
                        )
                        row = replace(
                            prior,
                            target_n_params=target,
                            convergence_history=list(prior.convergence_history),
                            error_trace=list(prior.error_trace),
                            evals_to_threshold=dict(prior.evals_to_threshold),
                            notes=(
                                f"REUSED: HE n_layers={n_layers} gives "
                                f"actual={actual} which matches target="
                                f"{prior.target_n_params}.  No new run."
                            ),
                        )
                    else:
                        budget = he_eval_budget
                        if budget <= 0:
                            budget = adapt_evals_at_target.get(target, DEFAULT_HE_EVAL_BUDGET)
                            logger.info(
                                "HE budget = ADAPT's evals at target=%d -> %d", target, budget,
                            )
                        row = _run_he_at_target(
                            loaded,
                            target_n_params=target,
                            optimizer=optimizer,
                            max_iterations=max_iterations,
                            convergence_threshold=convergence_threshold,
                            random_seed=random_seed,
                            rotation_gates=he_rotation_gates,
                            param_match=he_param_match,
                            eval_budget=budget,
                            init_strategy=he_init_strategy,
                            hf_init=he_hf_init,
                            stop_at_accuracy=he_stop_at_accuracy,
                        )
                        he_cache[row.actual_n_params] = row
                else:
                    raise ValueError(
                        f"Algorithm {algo!r} not supported by trainability "
                        "experiment.  Currently only 'adapt_vqe', "
                        "'qubit_adapt_vqe' and 'hardware_efficient_vqe' have "
                        "a defined parameter-target translation."
                    )
                rows.append(row)
                reached = row.evals_to_threshold.get(threshold_label(CHEMICAL_ACCURACY_HA))
                logger.info(
                    "[trainability] result: target=%d actual=%d (%s)  evals=%d  "
                    "err_vs_CASCI=%.3f mHa  chem.acc reached: %s",
                    row.target_n_params, row.actual_n_params, row.param_match,
                    row.n_cost_evals, row.error_vs_casci * 1e3,
                    f"after {reached} evals" if reached else "NO (censored)",
                )
            except Exception as exc:
                logger.exception("Failed for target=%d algo=%s: %s", target, algo, exc)

    _save_outputs(cfg, rows, out_dir, save_plots=save_plots)

    return {"config": cfg, "rows": rows, "output_dir": out_dir}


def discover_trainability_molecules(
    molecules: Optional[Sequence[str]] = None,
) -> List[str]:
    """Resolve molecule IDs for trainability runs.

    If ``molecules`` is provided, duplicates are removed while preserving
    order. Otherwise, all molecules with ``datasets2/<mol>/<mol>.h5`` are
    discovered from disk.
    """
    import h5py

    if molecules:
        deduped: List[str] = []
        seen = set()
        for mol in molecules:
            if mol not in seen:
                deduped.append(mol)
                seen.add(mol)
        candidates = deduped
    else:
        if not DATASETS2_DIR.is_dir():
            raise FileNotFoundError(f"Dataset directory not found: {DATASETS2_DIR}")

        candidates = sorted(
            p.name
            for p in DATASETS2_DIR.iterdir()
            if p.is_dir() and (p / f"{p.name}.h5").is_file()
        )

    discovered: List[str] = []
    skipped: List[str] = []
    for mol in candidates:
        h5_path = DATASETS2_DIR / mol / f"{mol}.h5"
        if not h5_path.is_file():
            skipped.append(mol)
            continue
        try:
            with h5py.File(h5_path, "r") as f:
                if "symbols" in f and "coordinates" in f:
                    discovered.append(mol)
                else:
                    skipped.append(mol)
        except Exception:
            skipped.append(mol)

    if not discovered:
        raise FileNotFoundError(
            f"No trainability-ready molecules found in {DATASETS2_DIR}; "
            "expected datasets2/<molecule>/<molecule>.h5 with 'symbols' and 'coordinates' datasets"
        )
    if skipped:
        logger.warning(
            "Skipping %d molecule(s) without trainability-ready geometry keys: %s",
            len(skipped),
            ", ".join(skipped),
        )
    return discovered


def run_trainability_all_molecules(
    molecules: Optional[Sequence[str]] = None,
    basis: str = "sto-3g",
    cs_target_qubits: Optional[int] = None,
    algorithms: Tuple[str, ...] = DEFAULT_ALGORITHMS,
    param_targets: Tuple[int, ...] = DEFAULT_PARAM_TARGETS,
    adapt_gradient_threshold: float = DEFAULT_GRADIENT_THRESHOLD,
    qubit_adapt_max_gradient_candidates: Optional[int] = None,
    optimizer: str = "COBYLA",
    max_iterations: int = 500,
    convergence_threshold: float = 1e-10,
    he_rotation_gates: str = "RY",
    random_seed: int = 42,
    output_root: Optional[Path] = None,
    resume: bool = False,
    save_plots: bool = True,
    he_param_match: str = "exact",
    he_eval_budget: int = DEFAULT_HE_EVAL_BUDGET,
    he_init_strategy: str = "random_uniform",
    he_hf_init: bool = False,
    he_stop_at_accuracy: bool = True,
) -> Dict[str, object]:
    """Run trainability comparison for multiple molecules and aggregate outputs."""
    import csv
    from datetime import datetime

    selected_molecules = discover_trainability_molecules(molecules)

    if output_root is None:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        output_root = (
            _FW_DIR / "experiments" / "results" / "trainability"
            / f"{timestamp}_all_molecules_comparison"
        )
    output_root.mkdir(parents=True, exist_ok=True)

    aggregated_rows: List[Tuple[str, TrainabilityRow]] = []
    succeeded: List[str] = []
    failed: Dict[str, str] = {}
    resumed: List[str] = []
    expected_rows_per_molecule = len(param_targets) * len(algorithms)

    for molecule in selected_molecules:
        molecule_out = output_root / molecule

        if resume:
            existing_rows = _load_trainability_rows(molecule_out)
            if len(existing_rows) >= expected_rows_per_molecule:
                logger.info(
                    "[trainability-batch] resume skip molecule=%s (%d existing rows)",
                    molecule,
                    len(existing_rows),
                )
                aggregated_rows.extend((molecule, row) for row in existing_rows)
                succeeded.append(molecule)
                resumed.append(molecule)
                continue

        logger.info(
            "[trainability-batch] molecule=%s output=%s",
            molecule,
            molecule_out,
        )
        try:
            out = run_trainability(
                molecule=molecule,
                basis=basis,
                cs_target_qubits=cs_target_qubits,
                algorithms=algorithms,
                param_targets=param_targets,
                adapt_gradient_threshold=adapt_gradient_threshold,
                qubit_adapt_max_gradient_candidates=qubit_adapt_max_gradient_candidates,
                optimizer=optimizer,
                max_iterations=max_iterations,
                convergence_threshold=convergence_threshold,
                he_rotation_gates=he_rotation_gates,
                random_seed=random_seed,
                output_dir=molecule_out,
                save_plots=save_plots,
                he_param_match=he_param_match,
                he_eval_budget=he_eval_budget,
                he_init_strategy=he_init_strategy,
                he_hf_init=he_hf_init,
                he_stop_at_accuracy=he_stop_at_accuracy,
            )
            rows = out["rows"]
            aggregated_rows.extend((molecule, row) for row in rows)
            succeeded.append(molecule)
        except Exception as exc:  # pragma: no cover
            failed[molecule] = str(exc)
            logger.exception(
                "Batch trainability failed for molecule=%s: %s",
                molecule,
                exc,
            )

    records = [cta.make_record(m, r) for m, r in aggregated_rows]
    cta.write_records_csv(output_root / "all_molecules_results.csv", records)

    with open(output_root / "all_molecules_convergence_traces.csv", "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["molecule", "algorithm", "target_n_params", "iteration", "energy"])
        for molecule, row in aggregated_rows:
            for it, e in enumerate(row.convergence_history):
                w.writerow([molecule, row.algorithm, row.target_n_params, it, f"{e:.10f}"])

    traces = [t for m, r in aggregated_rows for t in cta.traces_from_rows(m, [r])]
    cta_paths = cta.save_cost_to_accuracy_outputs(
        output_root, records, traces, prefix="all_molecules_", plots=save_plots,
    )
    if records:
        print("\n" + "=" * 70)
        print("COST TO REACH A FIXED ACCURACY -- all molecules")
        print("=" * 70)
        print(cta.render_console(cta.summarize(records)))
        print(f"\nReport: {cta_paths['report']}\n")

    if save_plots:
        _save_all_molecules_cost_evals_plot(output_root, aggregated_rows)

    manifest = {
        "molecules_requested": list(molecules) if molecules else None,
        "molecules_selected": selected_molecules,
        "molecules_succeeded": succeeded,
        "molecules_resumed": resumed,
        "molecules_failed": failed,
        "algorithms": list(algorithms),
        "param_targets": list(param_targets),
        "param_targets_note": (
            "target_n_params is the requested count; actual_n_params and param_match "
            "(exact | rounded_up | stopped_early) are recorded per row"
        ),
        "basis": basis,
        "cs_target_qubits": cs_target_qubits,
        "optimizer": optimizer,
        "max_iterations": max_iterations,
        "convergence_threshold": convergence_threshold,
        "adapt_gradient_threshold": adapt_gradient_threshold,
        "qubit_adapt_max_gradient_candidates": qubit_adapt_max_gradient_candidates,
        "he_rotation_gates": he_rotation_gates,
        "he_param_match": he_param_match,
        "he_eval_budget": he_eval_budget,
        "he_init_strategy": he_init_strategy,
        "he_hf_init": he_hf_init,
        "he_stop_at_accuracy": he_stop_at_accuracy,
        "accuracy_thresholds_ha": list(ACCURACY_LADDER_HA),
        "random_seed": random_seed,
        "n_rows": len(aggregated_rows),
    }
    with open(output_root / "batch_manifest.json", "w") as f:
        json.dump(manifest, f, indent=2)

    logger.info(
        "Batch outputs written to %s (succeeded=%d, failed=%d, rows=%d)",
        output_root,
        len(succeeded),
        len(failed),
        len(aggregated_rows),
    )

    return {
        "output_dir": output_root,
        "molecules": selected_molecules,
        "succeeded": succeeded,
        "failed": failed,
        "rows": aggregated_rows,
    }


# ──────────────────────────────────────────────────────────────────────────
# Output: CSV / JSON / plots
# ──────────────────────────────────────────────────────────────────────────

def _save_outputs(
    cfg: TrainabilityConfig,
    rows: List[TrainabilityRow],
    out_dir: Path,
    save_plots: bool = True,
) -> None:
    import csv

    out_dir.mkdir(parents=True, exist_ok=True)

    cfg_dict = asdict(cfg)
    cfg_dict["algorithms"] = list(cfg.algorithms)
    cfg_dict["param_targets"] = list(cfg.param_targets)
    cfg_dict["accuracy_thresholds_ha"] = list(cfg.accuracy_thresholds_ha)
    with open(out_dir / "run_config.json", "w") as f:
        json.dump(cfg_dict, f, indent=2)

    records = [cta.make_record(cfg.molecule, r, cfg.accuracy_thresholds_ha) for r in rows]
    cta.write_records_csv(out_dir / "results.csv", records, cfg.accuracy_thresholds_ha)
    cta.save_cost_to_accuracy_outputs(
        out_dir, records, cta.traces_from_rows(cfg.molecule, rows),
        thresholds=cfg.accuracy_thresholds_ha,
        single_molecule=cfg.molecule, plots=save_plots,
    )
    with open(out_dir / "convergence_traces.csv", "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["algorithm", "target_n_params", "iteration", "energy"])
        for r in rows:
            for it, e in enumerate(r.convergence_history):
                w.writerow([r.algorithm, r.target_n_params, it, f"{e:.10f}"])

    if not save_plots:
        return

    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except Exception as exc:  # pragma: no cover
        logger.warning("matplotlib unavailable: %s", exc)
        return

    # Group rows by algorithm
    by_algo: Dict[str, List[TrainabilityRow]] = {}
    for r in rows:
        by_algo.setdefault(r.algorithm, []).append(r)

    # Sort each group by actual_n_params
    for algo in by_algo:
        by_algo[algo].sort(key=lambda r: r.actual_n_params)

    # ── Plot 1: cost evals vs target params (grouped bar) ────────────────
    fig, ax = plt.subplots(figsize=(8, 4.5))
    targets = sorted({r.target_n_params for r in rows})
    n_algos = len(by_algo)
    bar_w = 0.8 / max(n_algos, 1)
    x = np.arange(len(targets))
    colors = plt.get_cmap("tab10")
    for i, (algo, algo_rows) in enumerate(by_algo.items()):
        # map target -> n_cost_evals
        evals_map = {r.target_n_params: r.n_cost_evals for r in algo_rows}
        ys = [evals_map.get(t, 0) for t in targets]
        ax.bar(x + i * bar_w, ys, bar_w, label=algo, color=colors(i))
    ax.set_xticks(x + bar_w * (n_algos - 1) / 2.0)
    ax.set_xticklabels([str(t) for t in targets])
    ax.set_xlabel("target n_parameters")
    ax.set_ylabel("# cost-function evaluations")
    ax.set_yscale("log")
    ax.set_title(
        f"Trainability: cost-fn evals per algorithm\n"
        f"{cfg.molecule}, {cfg.n_qubits_final} qubits"
    )
    ax.grid(True, axis="y", which="both", alpha=0.3)
    ax.legend()
    fig.tight_layout()
    fig.savefig(out_dir / "cost_evals_vs_params.png", dpi=150)
    plt.close(fig)

    # ── Plot 2: final energy vs target params ────────────────────────────
    fig, ax = plt.subplots(figsize=(8, 4.5))
    if cfg.hf_energy is not None:
        ax.axhline(cfg.hf_energy, color="grey", linestyle="--", linewidth=1,
                   label=f"HF = {cfg.hf_energy:.5f}")
    if cfg.casci_energy is not None:
        ax.axhline(cfg.casci_energy, color="black", linestyle=":", linewidth=1,
                   label=f"CASCI = {cfg.casci_energy:.5f}")
    for i, (algo, algo_rows) in enumerate(by_algo.items()):
        xs = [r.actual_n_params for r in algo_rows]
        ys = [r.final_energy for r in algo_rows]
        ax.plot(xs, ys, marker="o", label=algo, color=colors(i))
    ax.set_xlabel("actual n_parameters")
    ax.set_ylabel("final energy (Ha)")
    ax.set_title(
        f"Final energy vs parameter count\n"
        f"{cfg.molecule}, {cfg.n_qubits_final} qubits"
    )
    ax.grid(True, alpha=0.3)
    ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(out_dir / "final_energy_vs_params.png", dpi=150)
    plt.close(fig)

    # ── Plot 3: runtime vs target params ─────────────────────────────────
    fig, ax = plt.subplots(figsize=(8, 4.5))
    for i, (algo, algo_rows) in enumerate(by_algo.items()):
        xs = [r.actual_n_params for r in algo_rows]
        ys = [r.runtime_seconds for r in algo_rows]
        ax.plot(xs, ys, marker="o", label=algo, color=colors(i))
    ax.set_xlabel("actual n_parameters")
    ax.set_ylabel("runtime (seconds)")
    ax.set_yscale("log")
    ax.set_title(
        f"Wall-clock runtime vs parameter count\n"
        f"{cfg.molecule}, {cfg.n_qubits_final} qubits"
    )
    ax.grid(True, which="both", alpha=0.3)
    ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(out_dir / "runtime_vs_params.png", dpi=150)
    plt.close(fig)

    logger.info("Wrote outputs to %s", out_dir)


def _load_trainability_rows(molecule_out_dir: Path) -> List[TrainabilityRow]:
    """Load previously saved rows (and traces if present) from a molecule output dir."""
    import csv

    results_csv = molecule_out_dir / "results.csv"
    if not results_csv.is_file():
        return []

    rows_by_key: Dict[Tuple[str, int], TrainabilityRow] = {}
    with open(results_csv, newline="") as f:
        reader = csv.DictReader(f)
        for rec in reader:
            try:
                algo = str(rec["algorithm"])
                target = int(rec["target_n_params"])
                key = (algo, target)
                labels = [threshold_label(t) for t in ACCURACY_LADDER_HA]
                has_new = f"evals_to_{labels[-1]}" in rec
                rows_by_key[key] = TrainabilityRow(
                    algorithm=algo,
                    target_n_params=target,
                    actual_n_params=int(rec["actual_n_params"]),
                    n_outer_iters=int(rec["n_outer_iters"]),
                    n_cost_evals=int(rec["n_cost_evals"]),
                    runtime_seconds=float(rec["runtime_seconds"]),
                    final_energy=float(rec["final_energy"]),
                    error_vs_casci=float(rec["error_vs_casci"]),
                    converged_below_threshold=str(rec["converged_below_threshold"]).strip()
                    in ("1", "true", "True", "yes", "YES"),
                    convergence_history=[],
                    notes=str(rec.get("notes", "")),
                    param_match=str(rec.get("param_match", "")),
                    n_callback_evals=int(rec.get("n_callback_evals") or 0),
                    eval_budget=int(rec.get("eval_budget") or 0),
                    best_error_vs_casci=float(rec.get("best_error_vs_casci") or "nan"),
                    hf_error_vs_casci=float(rec.get("hf_error_vs_casci") or "nan"),
                    evals_to_threshold={
                        lab: (int(rec[f"evals_to_{lab}"]) if rec.get(f"evals_to_{lab}") else None)
                        for lab in labels
                    } if has_new else {},
                    stopped_at_accuracy=str(rec.get("stopped_at_accuracy", "")).strip() == "1",
                    init_label=str(rec.get("init_label", "")),
                    bounds_only=not has_new,
                )
            except Exception as exc:
                logger.warning("Could not parse row in %s: %s", results_csv, exc)

    eval_csv = molecule_out_dir / "eval_traces.csv"
    if eval_csv.is_file() and rows_by_key:
        with open(eval_csv, newline="") as f:
            for rec in csv.DictReader(f):
                try:
                    key = (str(rec["algorithm"]), int(rec["target_n_params"]))
                    if key in rows_by_key:
                        rows_by_key[key].error_trace.append(
                            (int(rec["eval_index"]), float(rec["best_error_vs_casci"]))
                        )
                except Exception:
                    continue

    traces_csv = molecule_out_dir / "convergence_traces.csv"
    if traces_csv.is_file() and rows_by_key:
        trace_map: Dict[Tuple[str, int], List[Tuple[int, float]]] = {
            k: [] for k in rows_by_key.keys()
        }
        with open(traces_csv, newline="") as f:
            reader = csv.DictReader(f)
            for rec in reader:
                try:
                    key = (str(rec["algorithm"]), int(rec["target_n_params"]))
                    if key not in trace_map:
                        continue
                    trace_map[key].append((int(rec["iteration"]), float(rec["energy"])))
                except Exception:
                    continue

        for key, pts in trace_map.items():
            pts.sort(key=lambda x: x[0])
            rows_by_key[key].convergence_history = [e for _, e in pts]

    return sorted(
        rows_by_key.values(),
        key=lambda r: (r.target_n_params, r.algorithm),
    )


def _save_all_molecules_cost_evals_plot(
    output_root: Path,
    aggregated_rows: List[Tuple[str, TrainabilityRow]],
) -> None:
    """Plot one aggregate cost-evals chart with all molecules and both algorithms."""
    if not aggregated_rows:
        return

    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except Exception as exc:  # pragma: no cover
        logger.warning("matplotlib unavailable for aggregate plot: %s", exc)
        return

    targets = sorted({r.target_n_params for _, r in aggregated_rows})
    algorithms = [
        algo for algo in DEFAULT_ALGORITHMS
        if any(r.algorithm == algo for _, r in aggregated_rows)
    ]
    if not algorithms:
        algorithms = sorted({r.algorithm for _, r in aggregated_rows})

    n_molecules = len({m for m, _ in aggregated_rows})
    x = np.arange(len(targets), dtype=float)
    offsets = np.linspace(-0.18, 0.18, num=max(len(algorithms), 1))
    colors = plt.get_cmap("tab10")
    rng = np.random.default_rng(42)

    fig, ax = plt.subplots(figsize=(11, 6))

    for i, algo in enumerate(algorithms):
        medians: List[float] = []
        for j, target in enumerate(targets):
            ys = [
                row.n_cost_evals
                for _, row in aggregated_rows
                if row.algorithm == algo and row.target_n_params == target
            ]
            if ys:
                xj = np.full(len(ys), x[j] + offsets[i]) + rng.uniform(-0.05, 0.05, size=len(ys))
                ax.scatter(
                    xj,
                    ys,
                    s=22,
                    alpha=0.45,
                    color=colors(i),
                    edgecolors="none",
                )
                medians.append(float(np.median(ys)))
            else:
                medians.append(float("nan"))

        ax.plot(
            x + offsets[i],
            medians,
            marker="o",
            linewidth=2,
            color=colors(i),
            label=f"{algo} median",
        )

    ax.set_xticks(x)
    ax.set_xticklabels([str(t) for t in targets])
    ax.set_xlabel("target n_parameters")
    ax.set_ylabel("# cost-function evaluations")
    ax.set_yscale("log")
    ax.set_title(
        f"Trainability across all molecules ({n_molecules} molecules)\n"
        "points = molecule runs, lines = algorithm median"
    )
    ax.grid(True, axis="y", which="both", alpha=0.3)
    ax.legend(fontsize=9)
    fig.tight_layout()
    out = output_root / "all_molecules_cost_evals_vs_params.png"
    fig.savefig(out, dpi=180)
    plt.close(fig)
    logger.info("Wrote aggregate cost-evals plot to %s", out)


# ──────────────────────────────────────────────────────────────────────────
# CLI
# ──────────────────────────────────────────────────────────────────────────

def _build_argparser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description="Trainability comparison at matched parameter counts.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument("--molecule", "-m", default="water",
                   help="Single molecule id for one run.")
    p.add_argument("--all-molecules", action="store_true",
                   help="Run all molecules discovered in datasets2/<mol>/<mol>.h5.")
    p.add_argument("--molecules", nargs="+", default=None,
                   help="Optional molecule subset for batch mode (also enables batch mode).")
    p.add_argument(
        "--output-root",
        type=Path,
        default=None,
        help="Batch output directory. If omitted, a timestamped folder is created.",
    )
    p.add_argument(
        "--resume",
        action="store_true",
        help="Resume batch mode by skipping molecules that already have complete results.csv rows.",
    )
    p.add_argument("--basis", default="sto-3g")
    p.add_argument("--cs-target-qubits", type=int, default=0,
                   help="<=0 to skip CS reduction (recommended -- CS makes "
                        "ADAPT terminate at k=0 because the start state is "
                        "the noncontextual GS).")
    p.add_argument("--algorithms", nargs="+", default=list(DEFAULT_ALGORITHMS),
                   help="Algorithms to compare. Supported: "
                        "{adapt_vqe, qubit_adapt_vqe, hardware_efficient_vqe}.")
    p.add_argument("--param-targets", nargs="+", type=int,
                   default=list(DEFAULT_PARAM_TARGETS),
                   help="Parameter-count targets to evaluate")
    p.add_argument("--adapt-gradient-threshold", type=float,
                   default=DEFAULT_GRADIENT_THRESHOLD,
                   help="ADAPT exits early if max pool gradient < this value.")
    p.add_argument(
        "--qubit-adapt-max-gradient-candidates",
        type=int,
        default=0,
        help=(
            "Cap gradient screening to N viable operators per ADAPT iteration "
            "(<=0 disables cap)."
        ),
    )
    p.add_argument("--optimizer", default="COBYLA")
    p.add_argument("--max-iter", type=int, default=500,
                   help="Per-ADAPT-step optimiser cap (also the generic COBYLA cap). "
                        "Hardware-efficient runs use --he-eval-budget instead.")
    p.add_argument("--convergence-threshold", type=float, default=1e-10)
    p.add_argument("--he-rotation-gates", default="RY",
                   choices=["RY", "RY_RZ", "full"])
    p.add_argument("--he-param-match", default="exact", choices=["exact", "round_up"],
                   help="exact: freeze surplus rotation slots so HE has exactly "
                        "target_n_params parameters; round_up: legacy behaviour "
                        "(next multiple of n_qubits*rotations).")
    p.add_argument("--he-eval-budget", type=int, default=DEFAULT_HE_EVAL_BUDGET,
                   help="Cost-function evaluations HE may use (0 = same as ADAPT used "
                        "at that target). Raised automatically to "
                        f"{HE_MIN_EVALS_PER_PARAM}*(n_params+1) if smaller.")
    p.add_argument("--he-init-strategy", default="random_uniform",
                   choices=["random_uniform", "small_random", "near_identity", "zeros"],
                   help="HE initial parameters (random_uniform = U[0,2pi], the "
                        "original behaviour).")
    p.add_argument("--he-hf-init", action="store_true",
                   help="Start HE from the Hartree-Fock reference instead of |0...0>.")
    p.add_argument("--no-he-stop-at-accuracy", action="store_true",
                   help="Do not end HE runs when chemical accuracy is reached; use "
                        "the full budget.")
    p.add_argument("--analyze-existing", type=Path, default=None, metavar="DIR",
                   help="Do not run anything: re-analyse an existing batch folder "
                        "(<mol>/results.csv + run_config.json) with the cost-to-"
                        "accuracy metric and print the summary.")
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

    if args.analyze_existing is not None:
        res = cta.analyze_existing(args.analyze_existing)
        print("\n" + "=" * 70)
        print(f"COST TO REACH A FIXED ACCURACY -- re-analysis of {args.analyze_existing}")
        print("=" * 70)
        print(cta.render_console(res["summary"]))
        print(f"\nReport:  {res['report']}\nFigure:  {res['output_dir'] / 'all_molecules_cost_to_accuracy.png'}")
        return

    cs_target = args.cs_target_qubits if args.cs_target_qubits > 0 else None
    qubit_adapt_cap = (
        args.qubit_adapt_max_gradient_candidates
        if args.qubit_adapt_max_gradient_candidates > 0
        else None
    )
    he_kwargs = dict(
        he_param_match=args.he_param_match,
        he_eval_budget=args.he_eval_budget,
        he_init_strategy=args.he_init_strategy,
        he_hf_init=args.he_hf_init,
        he_stop_at_accuracy=not args.no_he_stop_at_accuracy,
    )

    batch_mode = args.all_molecules or bool(args.molecules)
    if batch_mode:
        out = run_trainability_all_molecules(
            molecules=tuple(args.molecules) if args.molecules else None,
            basis=args.basis,
            cs_target_qubits=cs_target,
            algorithms=tuple(args.algorithms),
            param_targets=tuple(args.param_targets),
            adapt_gradient_threshold=args.adapt_gradient_threshold,
            qubit_adapt_max_gradient_candidates=qubit_adapt_cap,
            optimizer=args.optimizer,
            max_iterations=args.max_iter,
            convergence_threshold=args.convergence_threshold,
            he_rotation_gates=args.he_rotation_gates,
            random_seed=args.seed,
            output_root=args.output_root,
            resume=args.resume,
            save_plots=not args.no_plots,
            **he_kwargs,
        )
        failed: Dict[str, str] = out["failed"]
        succeeded: List[str] = out["succeeded"]
        print("\n" + "=" * 60)
        print("TRAINABILITY -- all molecules")
        print("=" * 60)
        print(f"Requested molecules: {len(out['molecules'])}")
        print(f"Succeeded: {len(succeeded)}")
        print(f"Failed: {len(failed)}")
        if failed:
            print("\nFailed molecules:")
            for mol, err in failed.items():
                print(f"  {mol}: {err}")
        print(f"\nBatch results saved to: {out['output_dir']}")
        return

    out = run_trainability(
        molecule=args.molecule,
        basis=args.basis,
        cs_target_qubits=cs_target,
        algorithms=tuple(args.algorithms),
        param_targets=tuple(args.param_targets),
        adapt_gradient_threshold=args.adapt_gradient_threshold,
        qubit_adapt_max_gradient_candidates=qubit_adapt_cap,
        optimizer=args.optimizer,
        max_iterations=args.max_iter,
        convergence_threshold=args.convergence_threshold,
        he_rotation_gates=args.he_rotation_gates,
        random_seed=args.seed,
        save_plots=not args.no_plots,
        **he_kwargs,
    )

    cfg: TrainabilityConfig = out["config"]
    rows: List[TrainabilityRow] = out["rows"]

    chem = threshold_label(CHEMICAL_ACCURACY_HA)
    print("\n" + "=" * 78)
    print(f"TRAINABILITY -- {cfg.molecule} ({cfg.n_qubits_final} qubits)")
    print("=" * 78)
    print(f"HF energy:     {cfg.hf_energy}")
    print(f"CASCI energy:  {cfg.casci_energy}")
    if cfg.hf_energy is not None and cfg.casci_energy is not None:
        gap = (cfg.hf_energy - cfg.casci_energy) * 1e3
        flag = "  <-- HF already within chemical accuracy: nothing to rank" if gap < 1.6 else ""
        print(f"E_HF - E_CASCI = {gap:.2f} mHa  (chemical accuracy = 1.6 mHa){flag}")
    print(f"HE settings:   param_match={cfg.he_param_match}, eval_budget={cfg.he_eval_budget}, "
          f"init={cfg.he_init_strategy}{'+HF' if cfg.he_hf_init else ''}")
    print()
    header = (
        f"{'algorithm':<24} {'target':>6} {'actual':>6} {'match':>13} "
        f"{'evals':>9} {'err_CASCI(mHa)':>15} {'evals to ' + chem:>16} {'runtime':>9}"
    )
    print(header)
    print("-" * len(header))
    for r in rows:
        hit = r.evals_to_threshold.get(chem)
        to_acc = f"{hit:,d}" if hit else f"NEVER (>{r.n_cost_evals:,d})"
        print(
            f"{r.algorithm:<24} {r.target_n_params:>6d} {r.actual_n_params:>6d} "
            f"{r.param_match:>13} {r.n_cost_evals:>9,d} "
            f"{r.error_vs_casci * 1e3:>15.3f} {to_acc:>16} {r.runtime_seconds:>8.1f}s"
        )

    print(f"\nResults saved to: {out['output_dir']}")
    print(f"  see {out['output_dir'] / 'cost_to_accuracy_report.md'}, cost_to_accuracy.png, error_vs_cost.png")


if __name__ == "__main__":
    main()
