"""
Experiment 2 -- Barren plateaus and parameter initialisation
============================================================

Question
--------
For a hardware-efficient ansatz on a fixed VQE problem, how does the
*initialisation distribution* of the variational parameters affect

A. the **gradient variance** at the start of optimisation
   (the standard barren-plateau diagnostic from McClean et al., 2018), and

B. the **convergence quality** of the optimisation that follows?

The hypothesis we test is:

    Sampling the initial parameters from a *narrow* distribution centred on
    zero leaves the state close to the Hartree-Fock reference (where the
    cost landscape has substantial gradient).  Sampling from a wide
    distribution like ``[0, 2*pi]`` lands in the flat exponentially-
    suppressed region of the loss surface, so optimisation stalls or
    converges to a noticeably worse solution.

What we measure
---------------

**Part A -- gradient variance vs depth**

For each ``n_layers`` in ``--layers`` and each initialisation strategy in
``--strategies`` we draw ``n_samples`` independent parameter vectors and
compute the parameter-shift gradient ``dE/d theta_k`` at a fixed parameter
index ``k`` (default ``k = 0``).  We report ``Var[g_k]`` and ``E[|g_k|]``.

Under the standard McClean BP scaling, ``Var[g_k]`` decays exponentially in
``n_layers * n_qubits`` for ``random_uniform`` but stays approximately
constant for ``small_random`` / ``near_identity``.

**Part B -- optimisation convergence**

Holding ``n_layers`` fixed, we run ``--n-trials`` independent
optimisations from each initialisation strategy and record the convergence
trace.  We then plot mean +- std curves and report the per-strategy mean
final energy / number of iterations / best result.

Outputs
-------
``framework/experiments/results/barren_plateau/<timestamp>_<mol>_<algo>/``

* ``run_config.json``
* ``gradient_variance.csv``       -- one row per (n_layers, strategy)
* ``optimization_results.csv``    -- one row per trial
* ``gradient_variance_vs_layers.png``
* ``convergence_by_init.png``
* ``final_energy_by_init.png``

Run as a script
---------------
::

    python -m experiments.barren_plateau \
        --molecule water --algorithm hardware_efficient_vqe \
        --layers 1 2 4 6 --n-samples 20 \
        --opt-layers 4 --n-trials 5 --max-iter 100
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
    gradient_index: int
    opt_layers: int
    opt_strategies: Tuple[str, ...]
    n_trials: int
    max_iterations: int
    convergence_threshold: float
    optimizer: str
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

def measure_gradient_variance(
    loaded: LoadedHamiltonian,
    *,
    algorithm: str,
    layers: Tuple[int, ...],
    strategies: Tuple[str, ...],
    n_samples: int,
    gradient_index: int,
    random_seed: int,
) -> List[GradientVarianceRow]:
    """For each (n_layers, strategy), draw ``n_samples`` parameter sets and
    compute one component of the parameter-shift gradient at each of them.

    Returns one :class:`GradientVarianceRow` per (n_layers, strategy).
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

        idx = min(max(gradient_index, 0), n_params - 1)

        for strategy in strategies:
            grads: List[float] = []
            for _ in range(n_samples):
                theta = sample_initial_parameters(n_params, strategy, rng=rng)
                g = parameter_shift_gradient(cost_fn, theta, indices=[idx])
                grads.append(float(g[0]))
            grads_arr = np.asarray(grads)
            row = GradientVarianceRow(
                n_layers=n_layers,
                strategy=strategy,
                n_parameters=n_params,
                gradient_index=idx,
                n_samples=n_samples,
                var_grad=float(np.var(grads_arr)),
                mean_abs_grad=float(np.mean(np.abs(grads_arr))),
                sample_grads=grads,
            )
            rows.append(row)
            logger.info(
                "BP-grad: layers=%d strategy=%s -> Var[g]=%.3e, E[|g|]=%.3e (n_params=%d)",
                n_layers, strategy, row.var_grad, row.mean_abs_grad, n_params,
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
    noise_model: Optional[str],
    noise_strength: float,
    random_seed: int,
) -> List[OptTrialRow]:
    """For each strategy, run ``n_trials`` independent optimisations and
    return one :class:`OptTrialRow` per trial.

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
    molecule: str = "water",
    algorithm: str = "hardware_efficient_vqe",
    basis: str = "sto-3g",
    cs_target_qubits: Optional[int] = 4,
    layers: Tuple[int, ...] = DEFAULT_LAYERS,
    bp_strategies: Tuple[str, ...] = DEFAULT_BP_STRATEGIES,
    n_samples: int = 30,
    gradient_index: int = 0,
    opt_layers: int = 6,
    opt_strategies: Tuple[str, ...] = DEFAULT_OPT_STRATEGIES,
    n_trials: int = 5,
    max_iterations: int = 200,
    convergence_threshold: float = 1e-10,
    optimizer: str = "COBYLA",
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

    cfg = BarrenPlateauConfig(
        molecule=molecule, algorithm=algorithm, basis=basis,
        cs_target_qubits=cs_target_qubits,
        layers=tuple(layers),
        bp_strategies=tuple(bp_strategies),
        n_samples=n_samples, gradient_index=gradient_index,
        opt_layers=opt_layers, opt_strategies=tuple(opt_strategies),
        n_trials=n_trials, max_iterations=max_iterations,
        convergence_threshold=convergence_threshold,
        optimizer=optimizer,
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
            max_iterations=max_iterations,
            convergence_threshold=convergence_threshold,
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
                "n_samples", "var_grad", "mean_abs_grad",
            ])
            for r in grad_rows:
                w.writerow([
                    r.n_layers, r.strategy, r.n_parameters, r.gradient_index,
                    r.n_samples, f"{r.var_grad:.10e}", f"{r.mean_abs_grad:.10e}",
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
        ax_var.set_ylabel(r"$\mathrm{Var}[\,\partial E/\partial\theta_k\,]$ (log scale)")
        ax_var.set_title("Gradient variance vs depth")
        ax_var.grid(True, which="both", alpha=0.3)
        ax_var.legend(fontsize=9)

        ax_mag.set_yscale("log")
        ax_mag.set_xlabel("n_layers")
        ax_mag.set_ylabel(r"$\mathbb{E}\,[\,|\partial E/\partial\theta_k|\,]$ (log scale)")
        ax_mag.set_title("Mean |gradient| vs depth")
        ax_mag.grid(True, which="both", alpha=0.3)
        ax_mag.legend(fontsize=9)

        fig.suptitle(
            f"{cfg.molecule} | {cfg.algorithm} | {cfg.n_qubits_final}q "
            f"(n={cfg.n_samples}, k={cfg.gradient_index})"
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

        for strategy, rs in by_strategy.items():
            max_len = max(len(r.convergence_history) for r in rs)
            mat = np.full((len(rs), max_len), np.nan)
            for i, r in enumerate(rs):
                mat[i, : len(r.convergence_history)] = r.convergence_history
            mean = np.nanmean(mat, axis=0)
            std = np.nanstd(mat, axis=0)
            xs = np.arange(max_len)
            line, = ax.plot(xs, mean, label=strategy, linewidth=2)
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
    p.add_argument("--molecule", "-m", default="water")
    p.add_argument("--algorithm", "-a", default="hardware_efficient_vqe",
                   help="Use hardware_efficient_vqe (recommended) or pennylane_vqe")
    p.add_argument("--basis", default="sto-3g")
    p.add_argument("--cs-target-qubits", type=int, default=4,
                   help="<=0 to skip CS reduction")
    # Part A
    p.add_argument("--layers", nargs="+", type=int, default=list(DEFAULT_LAYERS),
                   help="Ansatz depths to scan for the gradient-variance plot")
    p.add_argument("--bp-strategies", nargs="+", default=list(DEFAULT_BP_STRATEGIES),
                   choices=list(INIT_STRATEGIES))
    p.add_argument("--n-samples", type=int, default=30,
                   help="Independent parameter draws per (n_layers, strategy)")
    p.add_argument("--gradient-index", type=int, default=0,
                   help="Which parameter index to differentiate (0 by default)")
    # Part B
    p.add_argument("--opt-layers", type=int, default=6,
                   help="Ansatz depth for the optimisation-convergence comparison")
    p.add_argument("--opt-strategies", nargs="+", default=list(DEFAULT_OPT_STRATEGIES),
                   choices=list(INIT_STRATEGIES))
    p.add_argument("--n-trials", type=int, default=5)
    p.add_argument("--max-iter", type=int, default=200)
    p.add_argument("--convergence-threshold", type=float, default=1e-10,
                   help="Optimiser tolerance parameter (where applicable)")
    p.add_argument("--optimizer", default="COBYLA")
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

    out = run_barren_plateau(
        molecule=args.molecule,
        algorithm=args.algorithm,
        basis=args.basis,
        cs_target_qubits=cs_target,
        layers=tuple(args.layers),
        bp_strategies=tuple(args.bp_strategies),
        n_samples=args.n_samples,
        gradient_index=args.gradient_index,
        opt_layers=args.opt_layers,
        opt_strategies=tuple(args.opt_strategies),
        n_trials=args.n_trials,
        max_iterations=args.max_iter,
        convergence_threshold=args.convergence_threshold,
        optimizer=args.optimizer,
        noise_model=args.noise_model,
        noise_strength=args.noise_strength,
        random_seed=args.seed,
        save_plots=not args.no_plots,
        skip_part_a=args.skip_part_a,
        skip_part_b=args.skip_part_b,
    )

    cfg: BarrenPlateauConfig = out["config"]
    grad_rows: List[GradientVarianceRow] = out["grad_rows"]
    opt_rows: List[OptTrialRow] = out["opt_rows"]

    print("\n" + "=" * 60)
    print(f"BARREN PLATEAU STUDY -- {cfg.molecule} / {cfg.algorithm}")
    print("=" * 60)
    print(f"Active-space qubits: {cfg.n_qubits_active} -> final {cfg.n_qubits_final} "
          f"(CS={cfg.cs_target_qubits})")
    print(f"HF energy:    {cfg.hf_energy}")
    print(f"CASCI energy: {cfg.casci_energy}")

    if grad_rows:
        print("\n[Part A] gradient variance Var[∂E/∂θ_k]")
        print(f"{'n_layers':>8} {'strategy':<18} {'n_params':>9} "
              f"{'Var[g]':>14} {'E[|g|]':>14}")
        for r in grad_rows:
            print(f"{r.n_layers:>8d} {r.strategy:<18} {r.n_parameters:>9d} "
                  f"{r.var_grad:>14.3e} {r.mean_abs_grad:>14.3e}")

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
