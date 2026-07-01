"""
Run barren-plateau Part B (optimization comparison) for all molecules.

Default optimizer is Bayesian optimization. Results are saved per molecule and
aggregated across molecules in a timestamped output folder.

Usage:
    python run_barren_plateau_part_b_all_molecules.py
    python run_barren_plateau_part_b_all_molecules.py --optimizer BAYESIAN --opt-layers 16
    python run_barren_plateau_part_b_all_molecules.py --optimizer BAYESIAN --opt-layers 16 --noise-model depolarizing --noise-strengths 0.001 0.005 0.01
"""
from __future__ import annotations

import argparse
import csv
import json
import logging
import sys
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np

_FW_DIR = Path(__file__).resolve().parent / "framework"
if str(_FW_DIR) not in sys.path:
    sys.path.insert(0, str(_FW_DIR))

from experiments.barren_plateau import OptTrialRow, run_barren_plateau
from experiments.trainability import discover_trainability_molecules


DEFAULT_OPT_STRATEGIES: Tuple[str, ...] = (
    "random_uniform",
    "small_random",
    "near_identity",
)

logger = logging.getLogger(__name__)


def _save_aggregate_outputs(
    output_root: Path,
    aggregated_rows: List[Tuple[str, OptTrialRow]],
    save_plots: bool,
) -> None:
    output_root.mkdir(parents=True, exist_ok=True)

    with open(output_root / "all_molecules_optimization_results.csv", "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow([
            "molecule",
            "strategy",
            "trial",
            "n_layers",
            "n_parameters",
            "initial_energy",
            "final_energy",
            "n_iterations",
        ])
        for molecule, row in aggregated_rows:
            writer.writerow([
                molecule,
                row.strategy,
                row.trial,
                row.n_layers,
                row.n_parameters,
                f"{row.initial_energy:.10f}",
                f"{row.final_energy:.10f}",
                row.n_iterations,
            ])

    with open(output_root / "all_molecules_convergence_traces.csv", "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["molecule", "strategy", "trial", "iteration", "energy"])
        for molecule, row in aggregated_rows:
            for it, e in enumerate(row.convergence_history):
                writer.writerow([
                    molecule,
                    row.strategy,
                    row.trial,
                    it,
                    f"{float(e):.10f}",
                ])

    # Summary table by strategy
    by_strategy: Dict[str, List[OptTrialRow]] = {}
    for _, row in aggregated_rows:
        by_strategy.setdefault(row.strategy, []).append(row)

    with open(output_root / "all_molecules_strategy_summary.csv", "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow([
            "strategy",
            "n_runs",
            "mean_initial_energy",
            "mean_final_energy",
            "std_final_energy",
            "mean_iterations",
            "best_final_energy",
        ])
        for strategy, rows in sorted(by_strategy.items()):
            initial_es = np.asarray([r.initial_energy for r in rows], dtype=float)
            final_es = np.asarray([r.final_energy for r in rows], dtype=float)
            n_iters = np.asarray([r.n_iterations for r in rows], dtype=float)
            writer.writerow([
                strategy,
                len(rows),
                f"{initial_es.mean():.10f}",
                f"{final_es.mean():.10f}",
                f"{final_es.std():.10f}",
                f"{n_iters.mean():.4f}",
                f"{final_es.min():.10f}",
            ])

    if not save_plots or not aggregated_rows:
        return

    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except Exception as exc:  # pragma: no cover
        logger.warning("matplotlib unavailable, skipping aggregate plots: %s", exc)
        return

    # Plot 1: final-energy boxplot by strategy (all molecules/trials combined)
    strategies = sorted(by_strategy.keys())
    data = [[r.final_energy for r in by_strategy[s]] for s in strategies]

    fig, ax = plt.subplots(figsize=(8, 4.8))
    try:
        ax.boxplot(data, tick_labels=strategies, showmeans=True)
    except TypeError:
        ax.boxplot(data, labels=strategies, showmeans=True)
    ax.set_ylabel("Final energy (Ha)")
    ax.set_title("All molecules | Final energy by init")
    ax.grid(True, axis="y", alpha=0.3)
    fig.tight_layout()
    fig.savefig(output_root / "all_molecules_final_energy_by_strategy.png", dpi=180)
    plt.close(fig)

    # Plot 2: convergence mean +/- std by strategy across all rows
    fig, ax = plt.subplots(figsize=(8, 5))
    for strategy in strategies:
        rows = by_strategy[strategy]
        max_len = max(len(r.convergence_history) for r in rows)
        mat = np.full((len(rows), max_len), np.nan)
        for i, row in enumerate(rows):
            mat[i, : len(row.convergence_history)] = row.convergence_history
        mean = np.nanmean(mat, axis=0)
        std = np.nanstd(mat, axis=0)
        xs = np.arange(max_len)
        line, = ax.plot(xs, mean, linewidth=2, label=strategy)
        ax.fill_between(xs, mean - std, mean + std, alpha=0.15, color=line.get_color())

    ax.set_xlabel("Optimizer iteration")
    ax.set_ylabel("Energy (Ha)")
    ax.set_title("All molecules | Convergence by init")
    ax.grid(True, alpha=0.3)
    ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(output_root / "all_molecules_convergence_by_strategy.png", dpi=180)
    plt.close(fig)


def run_part_b_all_molecules(
    molecules: Optional[Sequence[str]] = None,
    *,
    algorithm: str = "hardware_efficient_vqe",
    basis: str = "sto-3g",
    cs_target_qubits: Optional[int] = 4,
    optimizer: str = "BAYESIAN",
    opt_layers: int = 16,
    opt_strategies: Tuple[str, ...] = DEFAULT_OPT_STRATEGIES,
    n_trials: int = 5,
    max_iterations: int = 200,
    convergence_threshold: float = 1e-10,
    noise_model: Optional[str] = None,
    noise_strength: float = 0.0,
    random_seed: int = 42,
    output_root: Optional[Path] = None,
    save_plots: bool = True,
) -> Dict[str, object]:
    selected_molecules = discover_trainability_molecules(molecules)

    if output_root is None:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        output_root = (
            _FW_DIR
            / "experiments"
            / "results"
            / "barren_plateau"
            / (
                f"{timestamp}_all_molecules_part_b_{optimizer.lower()}"
                if noise_model in (None, "none", "") or noise_strength == 0.0
                else f"{timestamp}_all_molecules_part_b_{optimizer.lower()}_{noise_model}_p{noise_strength:g}"
            )
        )
    output_root.mkdir(parents=True, exist_ok=True)

    aggregated_rows: List[Tuple[str, OptTrialRow]] = []
    succeeded: List[str] = []
    failed: Dict[str, str] = {}

    for molecule in selected_molecules:
        molecule_out = output_root / molecule
        print(f"[{molecule}] Running Part B with optimizer={optimizer}...")
        try:
            out = run_barren_plateau(
                molecule=molecule,
                algorithm=algorithm,
                basis=basis,
                cs_target_qubits=cs_target_qubits,
                layers=(1,),
                bp_strategies=("random_uniform",),
                n_samples=1,
                gradient_index=0,
                opt_layers=opt_layers,
                opt_strategies=opt_strategies,
                n_trials=n_trials,
                max_iterations=max_iterations,
                convergence_threshold=convergence_threshold,
                optimizer=optimizer,
                noise_model=noise_model,
                noise_strength=noise_strength,
                random_seed=random_seed,
                output_dir=molecule_out,
                save_plots=save_plots,
                skip_part_a=True,
                skip_part_b=False,
            )
            rows: List[OptTrialRow] = out["opt_rows"]
            aggregated_rows.extend((molecule, row) for row in rows)
            succeeded.append(molecule)
            print(f"  Done. trials={len(rows)}")
        except Exception as exc:
            failed[molecule] = str(exc)
            print(f"  [WARN] failed: {exc}")

    if aggregated_rows:
        _save_aggregate_outputs(output_root, aggregated_rows, save_plots=save_plots)

    manifest = {
        "molecules_requested": list(molecules) if molecules else None,
        "molecules_selected": selected_molecules,
        "molecules_succeeded": succeeded,
        "molecules_failed": failed,
        "algorithm": algorithm,
        "basis": basis,
        "cs_target_qubits": cs_target_qubits,
        "optimizer": optimizer,
        "noise_model": noise_model,
        "noise_strength": noise_strength,
        "opt_layers": opt_layers,
        "opt_strategies": list(opt_strategies),
        "n_trials": n_trials,
        "max_iterations": max_iterations,
        "convergence_threshold": convergence_threshold,
        "random_seed": random_seed,
        "n_rows": len(aggregated_rows),
    }
    with open(output_root / "batch_manifest.json", "w") as f:
        json.dump(manifest, f, indent=2)

    print("\n" + "=" * 60)
    print("BARREN PLATEAU -- Part B all molecules")
    print("=" * 60)
    print(f"Requested molecules: {len(selected_molecules)}")
    print(f"Succeeded: {len(succeeded)}")
    print(f"Failed: {len(failed)}")
    print(f"Optimizer: {optimizer}")
    print(f"Noise: {noise_model or 'none'} (strength={noise_strength})")
    print(f"Output dir: {output_root}")

    return {
        "output_dir": output_root,
        "molecules": selected_molecules,
        "succeeded": succeeded,
        "failed": failed,
        "rows": aggregated_rows,
    }


def _build_argparser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Run barren-plateau Part B across all molecules.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--molecules", nargs="+", default=None,
                        help="Optional subset of molecule IDs")
    parser.add_argument("--algorithm", default="hardware_efficient_vqe")
    parser.add_argument("--basis", default="sto-3g")
    parser.add_argument("--cs-target-qubits", type=int, default=4,
                        help="<=0 to skip CS reduction")
    parser.add_argument("--optimizer", default="BAYESIAN")
    parser.add_argument("--noise-model", default=None,
                        help="Optional Part B noise model, e.g. depolarizing, bitflip, phaseflip")
    parser.add_argument("--noise-strength", type=float, default=0.0,
                        help="Single noise strength/probability; 0 keeps the run noiseless")
    parser.add_argument("--noise-strengths", nargs="+", type=float, default=None,
                        help="Optional list of noise strengths to sweep; overrides --noise-strength")
    parser.add_argument("--opt-layers", type=int, default=16)
    parser.add_argument("--opt-strategies", nargs="+", default=list(DEFAULT_OPT_STRATEGIES))
    parser.add_argument("--n-trials", type=int, default=5)
    parser.add_argument("--max-iter", type=int, default=200)
    parser.add_argument("--convergence-threshold", type=float, default=1e-10)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--output-root", type=Path, default=None)
    parser.add_argument("--no-plots", action="store_true")
    parser.add_argument("-v", "--verbose", action="store_true")
    return parser


def main() -> None:
    args = _build_argparser().parse_args()

    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(asctime)s %(name)s [%(levelname)s] %(message)s",
    )
    for noisy_logger in ("pyscf", "pennylane", "matplotlib"):
        logging.getLogger(noisy_logger).setLevel(logging.WARNING)

    cs_target = args.cs_target_qubits if args.cs_target_qubits > 0 else None

    noise_strengths = args.noise_strengths if args.noise_strengths else [args.noise_strength]

    if len(noise_strengths) == 1:
        run_part_b_all_molecules(
            molecules=tuple(args.molecules) if args.molecules else None,
            algorithm=args.algorithm,
            basis=args.basis,
            cs_target_qubits=cs_target,
            optimizer=args.optimizer,
            opt_layers=args.opt_layers,
            opt_strategies=tuple(args.opt_strategies),
            n_trials=args.n_trials,
            max_iterations=args.max_iter,
            convergence_threshold=args.convergence_threshold,
            noise_model=args.noise_model,
            noise_strength=float(noise_strengths[0]),
            random_seed=args.seed,
            output_root=args.output_root,
            save_plots=not args.no_plots,
        )
        return

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    sweep_root = args.output_root or (
        _FW_DIR / "experiments" / "results" / "barren_plateau"
        / f"{timestamp}_all_molecules_part_b_{args.optimizer.lower()}_noise_sweep"
    )
    sweep_root.mkdir(parents=True, exist_ok=True)

    for strength in noise_strengths:
        label = "none" if args.noise_model in (None, "none", "") or strength == 0.0 else args.noise_model
        child_root = sweep_root / f"{label}_p{float(strength):g}"
        run_part_b_all_molecules(
            molecules=tuple(args.molecules) if args.molecules else None,
            algorithm=args.algorithm,
            basis=args.basis,
            cs_target_qubits=cs_target,
            optimizer=args.optimizer,
            opt_layers=args.opt_layers,
            opt_strategies=tuple(args.opt_strategies),
            n_trials=args.n_trials,
            max_iterations=args.max_iter,
            convergence_threshold=args.convergence_threshold,
            noise_model=args.noise_model,
            noise_strength=float(strength),
            random_seed=args.seed,
            output_root=child_root,
            save_plots=not args.no_plots,
        )


if __name__ == "__main__":
    main()
