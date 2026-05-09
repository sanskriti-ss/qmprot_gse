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
"How long did it take to train?" is intentionally measured as the
**total number of cost-function evaluations**, not as ``n_iterations``.
ADAPT reports ``n_iterations = number_of_operators_added`` (~tens),
while HardwareEfficient reports ``n_iterations = COBYLA_outer_iters``
(~hundreds).  But each ADAPT outer step issues
``|operator_pool| * 2 + ~100`` cost evaluations under the hood, so an
honest "trainability" comparison must count cost calls.

We attach :class:`experiments._common.CostEvalCounter` to each algorithm
instance to capture this number directly.

What we measure (per (target_n_params, algorithm) pair)
------------------------------------------------------

* ``actual_n_params``  -- the realised parameter count.
* ``n_outer_iters``    -- algorithm-reported iterations (operators for
  ADAPT, COBYLA outer iters for HW-efficient).
* ``n_cost_evals``     -- the *true* number of QNode evaluations.
* ``runtime_seconds``  -- wall time.
* ``final_energy``     -- last energy on convergence trace.
* ``error_vs_casci``   -- ``final_energy - CASCI`` (positive when above CI).
* ``convergence_history`` -- the algorithm's own convergence trace
  (interpretation differs: ADAPT logs energy after each operator
  addition; HW-efficient logs energy after each COBYLA outer iteration).

Outputs
-------
``framework/experiments/results/trainability/<timestamp>_<molecule>_comparison/``

* ``run_config.json``
* ``results.csv``                  -- one row per (target, algorithm).
* ``convergence_traces.csv``       -- long-format energy traces.
* ``cost_evals_vs_params.png``     -- bars of n_cost_evals per algorithm.
* ``final_energy_vs_params.png``   -- final energy reached per algorithm.
* ``runtime_vs_params.png``        -- wall time per algorithm.

Run as a script
---------------
::

    python -m experiments.trainability \\
        --molecule water --param-targets 10 50 100 --max-iter 500
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np

_FW_DIR = Path(__file__).resolve().parent.parent
if str(_FW_DIR) not in sys.path:
    sys.path.insert(0, str(_FW_DIR))

from algorithms import get_algorithm  # noqa: E402

from experiments._common import (  # noqa: E402
    CostEvalCounter,
    LoadedHamiltonian,
    he_layers_for_target_params,
    load_small_hamiltonian,
    make_backend_config,
    make_run_dir,
)

logger = logging.getLogger(__name__)


DEFAULT_PARAM_TARGETS: Tuple[int, ...] = (10, 50, 100)
DEFAULT_ALGORITHMS: Tuple[str, ...] = ("qubit_adapt_vqe", "hardware_efficient_vqe")
DEFAULT_GRADIENT_THRESHOLD: float = 1e-8


# ──────────────────────────────────────────────────────────────────────────
# Result containers
# ──────────────────────────────────────────────────────────────────────────

@dataclass
class TrainabilityRow:
    algorithm: str
    target_n_params: int
    actual_n_params: int
    n_outer_iters: int          # algorithm-reported iterations
    n_cost_evals: int           # true cost-function evaluations
    runtime_seconds: float
    final_energy: float
    error_vs_casci: float
    converged_below_threshold: bool
    convergence_history: List[float] = field(default_factory=list)
    notes: str = ""


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
    he_rotation_gates: str
    random_seed: int
    n_qubits_active: int
    n_qubits_final: int
    casci_energy: Optional[float]
    hf_energy: Optional[float]


# ──────────────────────────────────────────────────────────────────────────
# Per-algorithm runners
# ──────────────────────────────────────────────────────────────────────────

def _run_adapt_at_target(
    loaded: LoadedHamiltonian,
    *,
    algorithm: str,
    target_n_params: int,
    gradient_threshold: float,
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
    vqe = AlgorithmCls(
        loaded.hamiltonian,
        max_operators=target_n_params,
        gradient_threshold=gradient_threshold,
        optimizer=optimizer,
        max_iterations=max_iterations,
        convergence_threshold=convergence_threshold,
        random_seed=random_seed,
        backend_config=bc,
    )

    counter = CostEvalCounter()
    counter.attach(vqe)
    t0 = time.time()
    try:
        result = vqe.run()
    finally:
        counter.detach(vqe)
    runtime = time.time() - t0
    n_cost_evals = counter.count

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

    return TrainabilityRow(
        algorithm=algorithm,
        target_n_params=target_n_params,
        actual_n_params=int(result.n_parameters),
        n_outer_iters=int(result.n_iterations),
        n_cost_evals=int(n_cost_evals),
        runtime_seconds=runtime,
        final_energy=final_e,
        error_vs_casci=final_e - casci,
        converged_below_threshold=bool(result.converged),
        convergence_history=history,
        notes=notes,
    )


def _run_he_at_target(
    loaded: LoadedHamiltonian,
    *,
    target_n_params: int,
    optimizer: str,
    max_iterations: int,
    convergence_threshold: float,
    random_seed: int,
    rotation_gates: str = "RY",
) -> TrainabilityRow:
    """Run HardwareEfficientVQE at the smallest depth giving
    ``>= target_n_params`` parameters.
    """
    n_layers, actual = he_layers_for_target_params(
        loaded.hamiltonian.n_qubits,
        target_n_params,
        rotation_gates=rotation_gates,
        minimum_layers=1,  # always one entangling layer minimum
    )
    AlgorithmCls = get_algorithm("hardware_efficient_vqe")
    bc = make_backend_config(loaded.hamiltonian.n_qubits)
    vqe = AlgorithmCls(
        loaded.hamiltonian,
        n_layers=n_layers,
        rotation_gates=rotation_gates,
        optimizer=optimizer,
        max_iterations=max_iterations,
        convergence_threshold=convergence_threshold,
        random_seed=random_seed,
        backend_config=bc,
    )

    counter = CostEvalCounter()
    counter.attach(vqe)
    t0 = time.time()
    try:
        result = vqe.run()
    finally:
        counter.detach(vqe)
    runtime = time.time() - t0
    n_cost_evals = counter.count

    history = list(result.convergence_history)
    final_e = float(history[-1]) if history else float(result.calculated_energy)
    casci = loaded.casci_energy if loaded.casci_energy is not None else float("nan")

    notes = (
        f"n_layers={n_layers}, rotation_gates={rotation_gates}, "
        f"actual_n_params={actual} (target={target_n_params})"
    )

    return TrainabilityRow(
        algorithm="hardware_efficient_vqe",
        target_n_params=target_n_params,
        actual_n_params=int(actual),
        n_outer_iters=int(result.n_iterations),
        n_cost_evals=int(n_cost_evals),
        runtime_seconds=runtime,
        final_energy=final_e,
        error_vs_casci=final_e - casci,
        converged_below_threshold=bool(result.converged),
        convergence_history=history,
        notes=notes,
    )


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
    optimizer: str = "COBYLA",
    max_iterations: int = 500,
    convergence_threshold: float = 1e-10,
    he_rotation_gates: str = "RY",
    random_seed: int = 42,
    output_dir: Optional[Path] = None,
    save_plots: bool = True,
) -> Dict[str, object]:
    """Run the full trainability comparison.

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
        he_rotation_gates=he_rotation_gates,
        random_seed=random_seed,
        n_qubits_active=loaded.n_qubits_active,
        n_qubits_final=loaded.n_qubits_final,
        casci_energy=loaded.casci_energy,
        hf_energy=loaded.hf_energy,
    )

    out_dir = output_dir or make_run_dir(
        "trainability", molecule, "comparison",
    )
    logger.info("Output dir: %s", out_dir)

    # Pre-compute HE depth for each target so we can detect duplicates
    # (e.g. with RY rotations on 12 qubits, targets of 5 and 10 both
    # round up to n_layers=1 / 24 parameters).  We cache HE rows by
    # actual_n_params so duplicate targets re-use the same run instead
    # of recomputing.
    he_cache: Dict[int, TrainabilityRow] = {}

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
                        optimizer=optimizer,
                        max_iterations=max_iterations,
                        convergence_threshold=convergence_threshold,
                        random_seed=random_seed,
                    )
                elif algo == "hardware_efficient_vqe":
                    n_layers, actual = he_layers_for_target_params(
                        loaded.hamiltonian.n_qubits, target,
                        rotation_gates=he_rotation_gates, minimum_layers=1,
                    )
                    if actual in he_cache:
                        prior = he_cache[actual]
                        logger.info(
                            "HE at target=%d would round to actual=%d, "
                            "matching a previous target=%d run -- reusing "
                            "those numbers and noting the duplicate.",
                            target, actual, prior.target_n_params,
                        )
                        row = TrainabilityRow(
                            algorithm=prior.algorithm,
                            target_n_params=target,
                            actual_n_params=prior.actual_n_params,
                            n_outer_iters=prior.n_outer_iters,
                            n_cost_evals=prior.n_cost_evals,
                            runtime_seconds=prior.runtime_seconds,
                            final_energy=prior.final_energy,
                            error_vs_casci=prior.error_vs_casci,
                            converged_below_threshold=prior.converged_below_threshold,
                            convergence_history=list(prior.convergence_history),
                            notes=(
                                f"REUSED: HE n_layers={n_layers} gives "
                                f"actual={actual} which matches target="
                                f"{prior.target_n_params}.  No new run."
                            ),
                        )
                    else:
                        row = _run_he_at_target(
                            loaded,
                            target_n_params=target,
                            optimizer=optimizer,
                            max_iterations=max_iterations,
                            convergence_threshold=convergence_threshold,
                            random_seed=random_seed,
                            rotation_gates=he_rotation_gates,
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
                logger.info(
                    "[trainability] result: actual_n_params=%d  outer=%d  "
                    "cost_evals=%d  runtime=%.1fs  final_E=%.6f  err_vs_CASCI=%.6f",
                    row.actual_n_params, row.n_outer_iters, row.n_cost_evals,
                    row.runtime_seconds, row.final_energy, row.error_vs_casci,
                )
            except Exception as exc:
                logger.exception("Failed for target=%d algo=%s: %s", target, algo, exc)

    _save_outputs(cfg, rows, out_dir, save_plots=save_plots)

    return {"config": cfg, "rows": rows, "output_dir": out_dir}


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
    with open(out_dir / "run_config.json", "w") as f:
        json.dump(cfg_dict, f, indent=2)

    with open(out_dir / "results.csv", "w", newline="") as f:
        w = csv.writer(f)
        w.writerow([
            "algorithm", "target_n_params", "actual_n_params",
            "n_outer_iters", "n_cost_evals", "runtime_seconds",
            "final_energy", "error_vs_casci", "converged_below_threshold",
            "notes",
        ])
        for r in rows:
            w.writerow([
                r.algorithm, r.target_n_params, r.actual_n_params,
                r.n_outer_iters, r.n_cost_evals, f"{r.runtime_seconds:.4f}",
                f"{r.final_energy:.10f}", f"{r.error_vs_casci:.10f}",
                int(r.converged_below_threshold), r.notes,
            ])

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


# ──────────────────────────────────────────────────────────────────────────
# CLI
# ──────────────────────────────────────────────────────────────────────────

def _build_argparser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description="Trainability comparison at matched parameter counts.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument("--molecule", "-m", default="water")
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
    p.add_argument("--optimizer", default="COBYLA")
    p.add_argument("--max-iter", type=int, default=500)
    p.add_argument("--convergence-threshold", type=float, default=1e-10)
    p.add_argument("--he-rotation-gates", default="RY",
                   choices=["RY", "RY_RZ", "full"])
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

    out = run_trainability(
        molecule=args.molecule,
        basis=args.basis,
        cs_target_qubits=cs_target,
        algorithms=tuple(args.algorithms),
        param_targets=tuple(args.param_targets),
        adapt_gradient_threshold=args.adapt_gradient_threshold,
        optimizer=args.optimizer,
        max_iterations=args.max_iter,
        convergence_threshold=args.convergence_threshold,
        he_rotation_gates=args.he_rotation_gates,
        random_seed=args.seed,
        save_plots=not args.no_plots,
    )

    cfg: TrainabilityConfig = out["config"]
    rows: List[TrainabilityRow] = out["rows"]

    print("\n" + "=" * 60)
    print(f"TRAINABILITY -- {cfg.molecule} ({cfg.n_qubits_final} qubits)")
    print("=" * 60)
    print(f"HF energy:    {cfg.hf_energy}")
    print(f"CASCI energy: {cfg.casci_energy}")
    print()
    header = (
        f"{'algorithm':<24} {'target':>7} {'actual':>7} "
        f"{'outer':>7} {'cost_evals':>11} {'runtime':>9} "
        f"{'E_final':>14} {'err_CASCI':>11}"
    )
    print(header)
    print("-" * len(header))
    for r in rows:
        print(
            f"{r.algorithm:<24} {r.target_n_params:>7d} {r.actual_n_params:>7d} "
            f"{r.n_outer_iters:>7d} {r.n_cost_evals:>11d} "
            f"{r.runtime_seconds:>8.1f}s "
            f"{r.final_energy:>14.6f} {r.error_vs_casci:>11.4f}"
        )

    print(f"\nResults saved to: {out['output_dir']}")


if __name__ == "__main__":
    main()
