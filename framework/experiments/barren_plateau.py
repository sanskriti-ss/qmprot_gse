"""
Experiment 2 -- Barren plateaus and parameter initialisation
============================================================

Question
--------
For a hardware-efficient ansatz on a fixed VQE problem, how does the
*initialisation distribution* of the variational parameters affect

A. the **gradient variance** at the start of optimisation, and

B. the **convergence quality** of the optimisation that follows?

The default run is the five-molecule benchmark, ALA ARG ASP CYS GLY, each
reduced to 8 qubits.  That is the same active-space size as the main
comparison.  The circuit is the shared hardware-efficient ansatz: ``RY``
rotations.  With the contextual subspace on, the state prepared before
those rotations is the subspace reference, not the Hartree--Fock
determinant.  ``--cs-target-qubits 0`` keeps the full active space, where
that preparation really is Hartree--Fock.

What this can and cannot show: ``random_uniform`` is the hard
initialization (angles on ``[0, 2*pi]``).  A narrow distribution centred
on zero stays near the prepared reference.  The reported number is the
mean, over parameters spread through the circuit, of the sample variance
of one parameter-shift derivative.  It is not the McClean et al. (2018)
average over Haar-random circuits, and an exponential drop in that
variance is not guaranteed for this ladder and this molecular cost.

What we measure
---------------

**Part A -- gradient variance vs depth**

For each ``n_layers`` in ``--layers`` and each initialisation strategy in
``--bp-strategies`` we draw ``n_samples`` independent parameter vectors.
On each vector the parameter-shift rule is applied to parameters spread
through the circuit (every parameter, or at most ``--max-gradient-probes``,
default 16).  The recorded variance is the mean of those per-parameter
sample variances.  ``--gradient-index`` restricts the probe to one slot.

**Part B -- optimisation convergence**

Holding ``n_layers`` fixed, we run ``--n-trials`` independent
optimisations from each initialisation strategy.  The default optimizer
is gradient descent on the parameter-shift gradient, the same derivative
Part A measures.  Each step is ``theta <- theta - learning_rate * dE``,
and the default cap is 100 steps.  ``--optimizer COBYLA`` is the
derivative-free solver used in the VQE ranking; it does not read this
gradient.  Its automatic cap is ``max(n_parameters + 2, 20 * n_parameters)``.

Outputs
-------
``framework/experiments/results/barren_plateau/<timestamp>_<mol>_<algo>/``

* ``run_config.json``
* ``gradient_variance.csv``       -- one row per (n_layers, strategy); variance is the mean over probes
* ``gradient_variance_by_parameter.csv`` -- one row per probed parameter
* ``optimization_results.csv``    -- one row per trial
* ``gradient_variance_vs_layers.png``
* ``convergence_by_init.png``
* ``final_energy_by_init.png``

Run as a script
---------------
::

    python -m experiments.barren_plateau
    python -m experiments.barren_plateau --molecule gly --cs-target-qubits 0
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np

_FW_DIR = Path(__file__).resolve().parent.parent
if str(_FW_DIR) not in sys.path:
    sys.path.insert(0, str(_FW_DIR))

from core.base_vqe import incumbent_history, is_bayesian_optimizer  # noqa: E402
from experiments._common import (  # noqa: E402
    INIT_STRATEGIES,
    LoadedHamiltonian,
    build_vqe,
    load_small_hamiltonian,
    make_backend_config,
    make_run_dir,
    parameter_shift_gradient,
    sample_initial_parameters,
)

logger = logging.getLogger(__name__)


# Defaults: omit "zeros" from the BP analysis because for any ansatz where
# zeros put the gradient in a symmetry-protected zero point (e.g. UCC), the
# variance estimate is degenerate.  ``zeros`` is still useful as a control
# in the optimisation comparison.
DEFAULT_BP_STRATEGIES: Tuple[str, ...] = ("random_uniform", "small_random", "near_identity")
DEFAULT_OPT_STRATEGIES: Tuple[str, ...] = ("random_uniform", "small_random", "near_identity")
DEFAULT_LAYERS: Tuple[int, ...] = (1, 2, 4, 6, 8, 10)
# Same five molecules and qubit count as the main benchmark (14 -> 8 after CS).
DEFAULT_MOLECULES: Tuple[str, ...] = ("ala", "arg", "asp", "cys", "gly")
DEFAULT_CS_TARGET_QUBITS = 8
# Cap on how many parameters Part A differentiates.  <=0 means every parameter.
# The chosen indices are spread across the vector, not taken from the front.
DEFAULT_MAX_GRADIENT_PROBES = 16
DEFAULT_GRADIENT_STEPS = 100
DEFAULT_LEARNING_RATE = 0.1


# ──────────────────────────────────────────────────────────────────────────
# Result containers
# ──────────────────────────────────────────────────────────────────────────

@dataclass
class GradientVarianceRow:
    n_layers: int
    strategy: str
    n_parameters: int
    gradient_index: int
    n_samples: int
    var_grad: float
    mean_abs_grad: float
    n_probes: int = 1
    per_parameter: List[Tuple[int, float, float]] = field(default_factory=list)
    sample_grads: List[float] = field(default_factory=list)


@dataclass
class OptTrialRow:
    strategy: str
    trial: int
    n_layers: int
    n_parameters: int
    initial_energy: float
    final_energy: float
    n_iterations: int
    convergence_history: List[float] = field(default_factory=list)


@dataclass
class BarrenPlateauConfig:
    molecule: str
    algorithm: str
    basis: str
    cs_target_qubits: Optional[int]
    layers: Tuple[int, ...]
    bp_strategies: Tuple[str, ...]
    n_samples: int
    gradient_index: Optional[int]
    max_gradient_probes: int
    reference_state: str
    opt_layers: int
    opt_strategies: Tuple[str, ...]
    n_trials: int
    max_iterations: int
    convergence_threshold: float
    optimizer: str
    learning_rate: float
    noise_model: Optional[str]
    noise_strength: float
    random_seed: int
    n_qubits_active: int
    n_qubits_final: int
    casci_energy: Optional[float]
    hf_energy: Optional[float]


# ──────────────────────────────────────────────────────────────────────────
# Part A -- gradient variance
# ──────────────────────────────────────────────────────────────────────────

def _probe_indices(
    n_parameters: int,
    gradient_index: Optional[int],
    max_probes: int,
) -> List[int]:
    """Parameters whose derivatives enter the variance.

    An explicit ``gradient_index`` probes that one slot.  Otherwise the
    probes are spread across the whole vector.  ``max_probes <= 0`` keeps
    every parameter.  Taking the first ``max_probes`` slots would only see
    the opening rotation layer.
    """
    if n_parameters < 1:
        raise ValueError("ansatz has no parameters to differentiate")
    if gradient_index is not None:
        if not 0 <= gradient_index < n_parameters:
            raise ValueError(
                f"gradient_index={gradient_index} is outside "
                f"[0, {n_parameters - 1}]"
            )
        return [gradient_index]
    if max_probes <= 0 or n_parameters <= max_probes:
        return list(range(n_parameters))
    chosen = np.unique(np.rint(np.linspace(0, n_parameters - 1, max_probes)).astype(int))
    return [int(i) for i in chosen]


def _is_gradient_optimizer(optimizer: str) -> bool:
    key = optimizer.strip().lower().replace("-", "").replace("_", "")
    return key in {"gradientdescent", "gd"}


def resolve_iteration_budget(
    n_parameters: int,
    requested: int,
    optimizer: str = "COBYLA",
) -> int:
    """Part B iteration cap.

    Gradient descent counts parameter-shift steps.  ``requested <= 0``
    uses :data:`DEFAULT_GRADIENT_STEPS`.  A derivative-free solver instead
    gets at least a COBYLA simplex (``n_parameters + 2``) and, by default,
    20 evaluations per parameter.
    """
    if _is_gradient_optimizer(optimizer):
        budget = DEFAULT_GRADIENT_STEPS if requested <= 0 else requested
        if budget < 1:
            raise ValueError(f"gradient-descent steps must be >= 1, got {budget}")
        logger.info("Part B gradient-descent budget is %d steps.", budget)
        return budget

    simplex = n_parameters + 2
    if requested <= 0:
        budget = max(simplex, 20 * n_parameters)
        logger.info(
            "Part B iteration budget is %d for %d parameters.",
            budget, n_parameters,
        )
        return budget
    if requested < simplex:
        logger.warning(
            "max_iterations=%d is below the COBYLA simplex size "
            "n_parameters+2=%d. Part B will stop before the optimizer can move.",
            requested, simplex,
        )
    return requested


def parameter_shift_gradient_descent(
    cost_fn,
    theta: np.ndarray,
    n_steps: int,
    learning_rate: float,
) -> Tuple[np.ndarray, float, List[float], int]:
    """Move downhill along the parameter-shift gradient.

    The update is ``theta <- theta - learning_rate * dE``, with ``dE``
    from the same shift rule as Part A.  The returned energy is the cost
    of the returned parameters.  The history includes the initial energy.
    """
    if learning_rate <= 0:
        raise ValueError(f"learning_rate must be positive, got {learning_rate}")
    if n_steps < 1:
        raise ValueError(f"n_steps must be >= 1, got {n_steps}")
    theta = np.asarray(theta, dtype=float).copy()
    energy = float(cost_fn(theta))
    history = [energy]
    for _ in range(n_steps):
        grad = parameter_shift_gradient(cost_fn, theta)
        theta = theta - learning_rate * grad
        energy = float(cost_fn(theta))
        history.append(energy)
    return theta, energy, history, n_steps


def measure_gradient_variance(
    loaded: LoadedHamiltonian,
    *,
    algorithm: str,
    layers: Tuple[int, ...],
    strategies: Tuple[str, ...],
    n_samples: int,
    gradient_index: Optional[int],
    max_gradient_probes: int,
    random_seed: int,
) -> List[GradientVarianceRow]:
    """For each (n_layers, strategy), draw ``n_samples`` parameter sets.

    Each draw evaluates the parameter-shift gradient on the probed
    indices of that same vector.  The row's ``var_grad`` is the mean of
    the per-parameter sample variances, and ``mean_abs_grad`` is the mean
    of the per-parameter mean absolute derivatives.  ``gradient_index``
    on the row is that single index, or ``-1`` when several were averaged.
    """
    rows: List[GradientVarianceRow] = []
    rng = np.random.default_rng(random_seed)

    for n_layers in layers:
        # Build the ansatz once per depth so we can reuse the QNode across
        # samples (PennyLane caches gate metadata across calls).
        bc = make_backend_config(loaded.hamiltonian.n_qubits)
        vqe = build_vqe(
            loaded.hamiltonian,
            algorithm=algorithm,
            n_layers=n_layers,
            backend_config=bc,
            optimizer="COBYLA",
            max_iterations=1,  # we never call optimize here
            random_seed=random_seed,
        )
        n_params = int(vqe.n_parameters)
        cost_fn = vqe.cost_fn
        indices = _probe_indices(n_params, gradient_index, max_gradient_probes)

        for strategy in strategies:
            by_index: List[List[float]] = [[] for _ in indices]
            for _ in range(n_samples):
                theta = sample_initial_parameters(n_params, strategy, rng=rng)
                grad = parameter_shift_gradient(cost_fn, theta, indices=indices)
                for slot, value in enumerate(grad):
                    by_index[slot].append(float(value))
            per_parameter: List[Tuple[int, float, float]] = []
            for idx, samples in zip(indices, by_index):
                arr = np.asarray(samples)
                per_parameter.append((
                    idx, float(np.var(arr)), float(np.mean(np.abs(arr))),
                ))
            row = GradientVarianceRow(
                n_layers=n_layers,
                strategy=strategy,
                n_parameters=n_params,
                gradient_index=indices[0] if len(indices) == 1 else -1,
                n_samples=n_samples,
                var_grad=float(np.mean([item[1] for item in per_parameter])),
                mean_abs_grad=float(np.mean([item[2] for item in per_parameter])),
                n_probes=len(indices),
                per_parameter=per_parameter,
            )
            rows.append(row)
            logger.info(
                "BP-grad: layers=%d strategy=%s -> mean Var[g]=%.3e, "
                "mean E[|g|]=%.3e (n_params=%d, probes=%d)",
                n_layers, strategy, row.var_grad, row.mean_abs_grad,
                n_params, len(indices),
            )
    return rows


# ──────────────────────────────────────────────────────────────────────────
# Part B -- optimisation comparison
# ──────────────────────────────────────────────────────────────────────────

def run_optimization_comparison(
    loaded: LoadedHamiltonian,
    *,
    algorithm: str,
    n_layers: int,
    strategies: Tuple[str, ...],
    n_trials: int,
    optimizer: str,
    max_iterations: int,
    convergence_threshold: float,
    learning_rate: float,
    noise_model: Optional[str],
    noise_strength: float,
    random_seed: int,
) -> List[OptTrialRow]:
    """For each strategy, run ``n_trials`` independent optimisations and
    return one :class:`OptTrialRow` per trial.

    Gradient descent follows the parameter-shift gradient.  Any other
    optimizer is the solver attached to the ansatz, which for COBYLA
    never reads that gradient.

    Each trial uses a *fresh* RNG seeded by ``random_seed + trial`` so that
    the same trial index is comparable across strategies (modulo the
    different sampling distribution).
    """
    import warnings

    rows: List[OptTrialRow] = []

    for strategy in strategies:
        for trial in range(n_trials):
            seed = random_seed + 1009 * trial  # decorrelate trials
            rng = np.random.default_rng(seed)

            bc = make_backend_config(
                loaded.hamiltonian.n_qubits,
                noise_model=noise_model,
                noise_strength=noise_strength,
            )
            vqe = build_vqe(
                loaded.hamiltonian,
                algorithm=algorithm,
                n_layers=n_layers,
                backend_config=bc,
                optimizer=optimizer,
                max_iterations=max_iterations,
                convergence_threshold=convergence_threshold,
                random_seed=seed,
            )
            n_params = int(vqe.n_parameters)
            theta_init = sample_initial_parameters(n_params, strategy, rng=rng)
            with warnings.catch_warnings():
                warnings.filterwarnings(
                    "ignore",
                    category=FutureWarning,
                    message=r"functools\.partial will be a method descriptor in future Python versions.*",
                )
                initial_energy = float(vqe.cost_fn(theta_init))
            if _is_gradient_optimizer(optimizer):
                theta_star, e_star, history, n_steps = parameter_shift_gradient_descent(
                    vqe.cost_fn,
                    theta_init,
                    max_iterations,
                    learning_rate,
                )
                vqe.convergence_history = history
                vqe.iteration_count = n_steps
            else:
                theta_star, e_star = vqe.optimize(initial_parameters=theta_init.copy())
            row = OptTrialRow(
                strategy=strategy,
                trial=trial,
                n_layers=n_layers,
                n_parameters=n_params,
                initial_energy=initial_energy,
                final_energy=float(e_star),
                n_iterations=int(vqe.iteration_count),
                convergence_history=list(vqe.convergence_history),
            )
            rows.append(row)
            logger.info(
                "BP-opt: strategy=%s trial=%d  E_init=%.5f -> E_final=%.5f (%d iters)",
                strategy, trial, initial_energy, row.final_energy, row.n_iterations,
            )
    return rows


# ──────────────────────────────────────────────────────────────────────────
# Public API
# ──────────────────────────────────────────────────────────────────────────

def run_barren_plateau(
    molecule: str = "gly",
    algorithm: str = "hardware_efficient_vqe",
    basis: str = "sto-3g",
    cs_target_qubits: Optional[int] = DEFAULT_CS_TARGET_QUBITS,
    layers: Tuple[int, ...] = DEFAULT_LAYERS,
    bp_strategies: Tuple[str, ...] = DEFAULT_BP_STRATEGIES,
    n_samples: int = 30,
    gradient_index: Optional[int] = None,
    max_gradient_probes: int = DEFAULT_MAX_GRADIENT_PROBES,
    opt_layers: int = 6,
    opt_strategies: Tuple[str, ...] = DEFAULT_OPT_STRATEGIES,
    n_trials: int = 5,
    max_iterations: int = 0,
    convergence_threshold: float = 1e-10,
    optimizer: str = "GradientDescent",
    learning_rate: float = DEFAULT_LEARNING_RATE,
    noise_model: Optional[str] = None,
    noise_strength: float = 0.0,
    random_seed: int = 42,
    output_dir: Optional[Path] = None,
    save_plots: bool = True,
    skip_part_a: bool = False,
    skip_part_b: bool = False,
) -> Dict[str, object]:
    """Run the full barren-plateau study.

    Returns a dict with keys ``"config"``, ``"grad_rows"``, ``"opt_rows"``
    and ``"output_dir"``.
    """
    loaded = load_small_hamiltonian(
        molecule=molecule, basis=basis, cs_target_qubits=cs_target_qubits,
    )
    reference_state = (
        "contextual_subspace" if loaded.cs_reduced else "hartree_fock"
    )
    part_b_iterations = max_iterations
    if not skip_part_b:
        probe = build_vqe(
            loaded.hamiltonian,
            algorithm=algorithm,
            n_layers=opt_layers,
            backend_config=make_backend_config(loaded.hamiltonian.n_qubits),
            optimizer=optimizer,
            max_iterations=1,
            random_seed=random_seed,
        )
        part_b_iterations = resolve_iteration_budget(
            int(probe.n_parameters), max_iterations, optimizer,
        )

    cfg = BarrenPlateauConfig(
        molecule=molecule, algorithm=algorithm, basis=basis,
        cs_target_qubits=cs_target_qubits,
        layers=tuple(layers),
        bp_strategies=tuple(bp_strategies),
        n_samples=n_samples, gradient_index=gradient_index,
        max_gradient_probes=max_gradient_probes,
        reference_state=reference_state,
        opt_layers=opt_layers, opt_strategies=tuple(opt_strategies),
        n_trials=n_trials, max_iterations=part_b_iterations,
        convergence_threshold=convergence_threshold,
        optimizer=optimizer,
        learning_rate=learning_rate,
        noise_model=noise_model,
        noise_strength=noise_strength,
        random_seed=random_seed,
        n_qubits_active=loaded.n_qubits_active,
        n_qubits_final=loaded.n_qubits_final,
        casci_energy=loaded.casci_energy,
        hf_energy=loaded.hf_energy,
    )

    out_dir = output_dir or make_run_dir("barren_plateau", molecule, algorithm)
    logger.info("Output dir: %s", out_dir)

    grad_rows: List[GradientVarianceRow] = []
    if not skip_part_a:
        grad_rows = measure_gradient_variance(
            loaded,
            algorithm=algorithm,
            layers=tuple(layers),
            strategies=tuple(bp_strategies),
            n_samples=n_samples,
            gradient_index=gradient_index,
            max_gradient_probes=max_gradient_probes,
            random_seed=random_seed,
        )

    opt_rows: List[OptTrialRow] = []
    if not skip_part_b:
        opt_rows = run_optimization_comparison(
            loaded,
            algorithm=algorithm,
            n_layers=opt_layers,
            strategies=tuple(opt_strategies),
            n_trials=n_trials,
            optimizer=optimizer,
            max_iterations=part_b_iterations,
            convergence_threshold=convergence_threshold,
            learning_rate=learning_rate,
            noise_model=noise_model,
            noise_strength=noise_strength,
            random_seed=random_seed,
        )

    _save_outputs(cfg, grad_rows, opt_rows, out_dir, save_plots=save_plots)

    return {
        "config": cfg,
        "grad_rows": grad_rows,
        "opt_rows": opt_rows,
        "output_dir": out_dir,
    }


def _mean_std_trace(traces: List[List[float]]) -> Tuple[np.ndarray, np.ndarray]:
    """Column-wise mean and std of ragged energy traces (NaN-padded)."""
    max_len = max(len(t) for t in traces)
    mat = np.full((len(traces), max_len), np.nan)
    for i, trace in enumerate(traces):
        mat[i, : len(trace)] = trace
    return np.nanmean(mat, axis=0), np.nanstd(mat, axis=0)


# ──────────────────────────────────────────────────────────────────────────
# Output: CSV / JSON / plots
# ──────────────────────────────────────────────────────────────────────────

def _save_outputs(
    cfg: BarrenPlateauConfig,
    grad_rows: List[GradientVarianceRow],
    opt_rows: List[OptTrialRow],
    out_dir: Path,
    save_plots: bool = True,
) -> None:
    import csv

    out_dir.mkdir(parents=True, exist_ok=True)

    # run_config.json
    cfg_dict = asdict(cfg)
    cfg_dict["layers"] = list(cfg.layers)
    cfg_dict["bp_strategies"] = list(cfg.bp_strategies)
    cfg_dict["opt_strategies"] = list(cfg.opt_strategies)
    with open(out_dir / "run_config.json", "w") as f:
        json.dump(cfg_dict, f, indent=2)

    # gradient_variance.csv
    if grad_rows:
        with open(out_dir / "gradient_variance.csv", "w", newline="") as f:
            w = csv.writer(f)
            w.writerow([
                "n_layers", "strategy", "n_parameters", "gradient_index",
                "n_probes", "n_samples", "var_grad", "mean_abs_grad",
            ])
            for r in grad_rows:
                w.writerow([
                    r.n_layers, r.strategy, r.n_parameters, r.gradient_index,
                    r.n_probes, r.n_samples,
                    f"{r.var_grad:.10e}", f"{r.mean_abs_grad:.10e}",
                ])
        with open(out_dir / "gradient_variance_by_parameter.csv", "w", newline="") as f:
            w = csv.writer(f)
            w.writerow([
                "n_layers", "strategy", "gradient_index",
                "n_samples", "var_grad", "mean_abs_grad",
            ])
            for r in grad_rows:
                for idx, var_g, mean_abs in r.per_parameter:
                    w.writerow([
                        r.n_layers, r.strategy, idx, r.n_samples,
                        f"{var_g:.10e}", f"{mean_abs:.10e}",
                    ])

    # optimization_results.csv (one summary row per trial)
    if opt_rows:
        with open(out_dir / "optimization_results.csv", "w", newline="") as f:
            w = csv.writer(f)
            w.writerow([
                "strategy", "trial", "n_layers", "n_parameters",
                "initial_energy", "final_energy", "n_iterations",
            ])
            for r in opt_rows:
                w.writerow([
                    r.strategy, r.trial, r.n_layers, r.n_parameters,
                    f"{r.initial_energy:.10f}", f"{r.final_energy:.10f}",
                    r.n_iterations,
                ])

        # Long-format convergence traces, in case downstream analysis wants them
        with open(out_dir / "convergence_traces.csv", "w", newline="") as f:
            w = csv.writer(f)
            w.writerow(["strategy", "trial", "iteration", "energy"])
            for r in opt_rows:
                for it, e in enumerate(r.convergence_history):
                    w.writerow([r.strategy, r.trial, it, f"{e:.10f}"])

    if not save_plots:
        return

    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except Exception as exc:  # pragma: no cover
        logger.warning("matplotlib unavailable, skipping plots: %s", exc)
        return

    # Part A plot: variance vs n_layers per strategy
    if grad_rows:
        fig, (ax_var, ax_mag) = plt.subplots(1, 2, figsize=(12, 4.5))
        by_strategy: Dict[str, List[GradientVarianceRow]] = {}
        for r in grad_rows:
            by_strategy.setdefault(r.strategy, []).append(r)
        for strategy, rs in by_strategy.items():
            rs_sorted = sorted(rs, key=lambda r: r.n_layers)
            xs = [r.n_layers for r in rs_sorted]
            ax_var.plot(xs, [r.var_grad for r in rs_sorted], marker="o", label=strategy)
            ax_mag.plot(xs, [r.mean_abs_grad for r in rs_sorted], marker="o", label=strategy)
        ax_var.set_yscale("log")
        ax_var.set_xlabel("n_layers")
        ax_var.set_ylabel(r"mean$_k\,\mathrm{Var}[\,\partial E/\partial\theta_k\,]$ (log scale)")
        ax_var.set_title("Gradient variance vs depth")
        ax_var.grid(True, which="both", alpha=0.3)
        ax_var.legend(fontsize=9)

        ax_mag.set_yscale("log")
        ax_mag.set_xlabel("n_layers")
        ax_mag.set_ylabel(r"mean$_k\,\mathbb{E}\,[\,|\partial E/\partial\theta_k|\,]$ (log scale)")
        ax_mag.set_title("Mean |gradient| vs depth")
        ax_mag.grid(True, which="both", alpha=0.3)
        ax_mag.legend(fontsize=9)

        fig.suptitle(
            f"{cfg.molecule} | {cfg.algorithm} | {cfg.n_qubits_final}q "
            f"| {cfg.reference_state} "
            f"(n={cfg.n_samples}, probes={grad_rows[0].n_probes})"
        )
        fig.tight_layout()
        fig.savefig(out_dir / "gradient_variance_vs_layers.png", dpi=150)
        plt.close(fig)

    # Part B plots: convergence + final-energy comparison
    if opt_rows:
        # Convergence: mean +- std per strategy at common iteration index
        by_strategy: Dict[str, List[OptTrialRow]] = {}
        for r in opt_rows:
            by_strategy.setdefault(r.strategy, []).append(r)

        fig, ax = plt.subplots(figsize=(8, 5))
        if cfg.casci_energy is not None:
            ax.axhline(cfg.casci_energy, color="black", linestyle=":",
                       linewidth=1, label=f"CASCI = {cfg.casci_energy:.5f}")
        if cfg.hf_energy is not None:
            ax.axhline(cfg.hf_energy, color="grey", linestyle="--",
                       linewidth=1, label=f"HF = {cfg.hf_energy:.5f}")

        bayesian = is_bayesian_optimizer(cfg.optimizer)
        if bayesian:
            ax.plot(
                [], [], color="0.5", alpha=0.7, linewidth=1,
                label="raw acquisition (mean)",
            )
        for strategy, rs in by_strategy.items():
            raw_traces = [r.convergence_history for r in rs]
            inc_traces = [incumbent_history(t) for t in raw_traces]
            mean, std = _mean_std_trace(inc_traces)
            xs = np.arange(len(mean))
            line, = ax.plot(xs, mean, label=strategy, linewidth=2, zorder=2)
            if bayesian:
                mean_raw, _ = _mean_std_trace(raw_traces)
                ax.plot(
                    np.arange(len(mean_raw)), mean_raw,
                    color=line.get_color(), alpha=0.25, linewidth=1, zorder=1,
                )
            ax.fill_between(xs, mean - std, mean + std,
                            alpha=0.15, color=line.get_color())
        ax.set_xlabel("Optimizer iteration")
        ax.set_ylabel("Energy (Ha)")
        ax.set_title(
            f"Convergence | {cfg.molecule} | {cfg.algorithm} | "
            f"L={cfg.opt_layers}, n={cfg.n_trials}"
        )
        ax.grid(True, alpha=0.3)
        ax.legend(fontsize=9)
        fig.tight_layout()
        fig.savefig(out_dir / "convergence_by_init.png", dpi=150)
        plt.close(fig)

        # Final-energy boxplot per strategy
        fig, ax = plt.subplots(figsize=(7, 4.5))
        labels = list(by_strategy.keys())
        data = [[r.final_energy for r in by_strategy[s]] for s in labels]
        # Matplotlib >=3.9 renamed ``labels`` to ``tick_labels``; fall back
        # gracefully on older versions.
        try:
            ax.boxplot(data, tick_labels=labels, showmeans=True)
        except TypeError:
            ax.boxplot(data, labels=labels, showmeans=True)
        if cfg.casci_energy is not None:
            ax.axhline(cfg.casci_energy, color="black", linestyle=":", linewidth=1,
                       label=f"CASCI = {cfg.casci_energy:.5f}")
        if cfg.hf_energy is not None:
            ax.axhline(cfg.hf_energy, color="grey", linestyle="--", linewidth=1,
                       label=f"HF = {cfg.hf_energy:.5f}")
        ax.set_ylabel("Final energy (Ha)")
        ax.set_title(
            f"Final energy by init | L={cfg.opt_layers}, n={cfg.n_trials}"
        )
        ax.grid(True, axis="y", alpha=0.3)
        ax.legend(fontsize=8, loc="best")
        fig.tight_layout()
        fig.savefig(out_dir / "final_energy_by_init.png", dpi=150)
        plt.close(fig)

    logger.info("Wrote outputs to %s", out_dir)


# ──────────────────────────────────────────────────────────────────────────
# CLI
# ──────────────────────────────────────────────────────────────────────────

def _build_argparser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description=(
            "Barren-plateau diagnostic + initialisation-strategy comparison."
        ),
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument("--molecules", nargs="+", default=list(DEFAULT_MOLECULES),
                   help="Molecules to run. Default is ala, arg, asp, cys, gly.")
    p.add_argument("--molecule", "-m", default=None,
                   help="Run this one molecule instead of --molecules.")
    p.add_argument("--output-dir", type=Path, default=None,
                   help="Single-molecule output directory, or the parent folder for several molecules.")
    p.add_argument("--algorithm", "-a", default="hardware_efficient_vqe",
                   help="Use hardware_efficient_vqe (recommended) or pennylane_vqe")
    p.add_argument("--basis", default="sto-3g")
    p.add_argument("--cs-target-qubits", type=int, default=DEFAULT_CS_TARGET_QUBITS,
                   help="Contextual-subspace size. 8 is the benchmark reduction. "
                        "<=0 keeps the full active space and Hartree-Fock preparation.")
    # Part A
    p.add_argument("--layers", nargs="+", type=int, default=list(DEFAULT_LAYERS),
                   help="Ansatz depths to scan for the gradient-variance plot")
    p.add_argument("--bp-strategies", nargs="+", default=list(DEFAULT_BP_STRATEGIES),
                   choices=list(INIT_STRATEGIES))
    p.add_argument("--n-samples", type=int, default=30,
                   help="Independent parameter draws per (n_layers, strategy)")
    p.add_argument("--gradient-index", type=int, default=None,
                   help="Differentiate only this parameter. Default: probes spread "
                        "through the circuit.")
    p.add_argument("--max-gradient-probes", type=int, default=DEFAULT_MAX_GRADIENT_PROBES,
                   help="How many spread-out parameters to differentiate. <=0 means all.")
    # Part B
    p.add_argument("--opt-layers", type=int, default=6,
                   help="Ansatz depth for the optimisation-convergence comparison")
    p.add_argument("--opt-strategies", nargs="+", default=list(DEFAULT_OPT_STRATEGIES),
                   choices=list(INIT_STRATEGIES))
    p.add_argument("--n-trials", type=int, default=5)
    p.add_argument("--max-iter", type=int, default=0,
                   help="Part B cap. <=0 uses 100 gradient steps, or "
                        "max(n_parameters+2, 20*n_parameters) evaluations for COBYLA.")
    p.add_argument("--convergence-threshold", type=float, default=1e-10,
                   help="COBYLA rhoend. Gradient descent runs the full step count.")
    p.add_argument("--optimizer", default="GradientDescent",
                   help="GradientDescent follows the parameter-shift gradient. "
                        "COBYLA is the derivative-free ranking solver.")
    p.add_argument("--learning-rate", type=float, default=DEFAULT_LEARNING_RATE,
                   help="Step size for GradientDescent, in radians per unit gradient.")
    p.add_argument("--noise-model", default=None,
                   help="Optional noise model for Part B (e.g. depolarizing, bitflip, phaseflip).")
    p.add_argument("--noise-strength", type=float, default=0.0,
                   help="Noise strength/probability for Part B; 0 keeps the run noiseless.")
    # Global
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--no-plots", action="store_true")
    p.add_argument("--skip-part-a", action="store_true")
    p.add_argument("--skip-part-b", action="store_true")
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
    molecules = [args.molecule] if args.molecule else list(args.molecules)
    parent = args.output_dir
    if parent is None and len(molecules) > 1:
        parent = make_run_dir("barren_plateau", "batch", args.algorithm)

    for molecule in molecules:
        out = run_barren_plateau(
            molecule=molecule,
            algorithm=args.algorithm,
            basis=args.basis,
            cs_target_qubits=cs_target,
            layers=tuple(args.layers),
            bp_strategies=tuple(args.bp_strategies),
            n_samples=args.n_samples,
            gradient_index=args.gradient_index,
            max_gradient_probes=args.max_gradient_probes,
            opt_layers=args.opt_layers,
            opt_strategies=tuple(args.opt_strategies),
            n_trials=args.n_trials,
            max_iterations=args.max_iter,
            convergence_threshold=args.convergence_threshold,
            optimizer=args.optimizer,
            learning_rate=args.learning_rate,
            noise_model=args.noise_model,
            noise_strength=args.noise_strength,
            random_seed=args.seed,
            output_dir=(parent / molecule) if parent is not None and len(molecules) > 1 else parent,
            save_plots=not args.no_plots,
            skip_part_a=args.skip_part_a,
            skip_part_b=args.skip_part_b,
        )
        _print_summary(out)


def _print_summary(out: Dict[str, object]) -> None:
    cfg: BarrenPlateauConfig = out["config"]
    grad_rows: List[GradientVarianceRow] = out["grad_rows"]
    opt_rows: List[OptTrialRow] = out["opt_rows"]

    print("\n" + "=" * 60)
    print(f"BARREN PLATEAU STUDY -- {cfg.molecule} / {cfg.algorithm}")
    print("=" * 60)
    print(f"Active-space qubits: {cfg.n_qubits_active} -> final {cfg.n_qubits_final} "
          f"(CS={cfg.cs_target_qubits})")
    print(f"Reference state: {cfg.reference_state}")
    print(f"Part B optimizer: {cfg.optimizer}"
          + (f", learning_rate={cfg.learning_rate}" if _is_gradient_optimizer(cfg.optimizer) else ""))
    print(f"HF energy:    {cfg.hf_energy}")
    print(f"CASCI energy: {cfg.casci_energy}")
    if opt_rows:
        print(f"Part B iteration cap: {cfg.max_iterations}")

    if grad_rows:
        print("\n[Part A] mean over probes of Var[∂E/∂θ_k]")
        print(f"{'n_layers':>8} {'strategy':<18} {'n_params':>9} {'probes':>7} "
              f"{'Var[g]':>14} {'E[|g|]':>14}")
        for r in grad_rows:
            print(f"{r.n_layers:>8d} {r.strategy:<18} {r.n_parameters:>9d} "
                  f"{r.n_probes:>7d} {r.var_grad:>14.3e} {r.mean_abs_grad:>14.3e}")

    if opt_rows:
        print("\n[Part B] final energy by initialisation strategy")
        by_strategy: Dict[str, List[OptTrialRow]] = {}
        for r in opt_rows:
            by_strategy.setdefault(r.strategy, []).append(r)
        print(f"{'strategy':<18} {'n_trials':>9} {'mean E':>14} "
              f"{'std E':>10} {'best E':>14} {'mean iters':>12}")
        for strategy, rs in by_strategy.items():
            es = np.array([r.final_energy for r in rs])
            its = np.array([r.n_iterations for r in rs])
            print(f"{strategy:<18} {len(rs):>9d} {es.mean():>14.6f} "
                  f"{es.std():>10.4f} {es.min():>14.6f} {its.mean():>12.1f}")

    print(f"\nResults saved to: {out['output_dir']}")


if __name__ == "__main__":
    main()
