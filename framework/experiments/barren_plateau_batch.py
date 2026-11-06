"""
Barren plateau study across multiple molecules
==============================================

Extends the single-molecule barren_plateau experiment to run across multiple
amino acids from datasets2/, collecting gradient variance and convergence data
into batch CSV files and generating multi-molecule plots.

This experiment:
1. Runs barren plateau analysis for each molecule independently
2. Aggregates all gradient variance measurements into a single CSV
3. Aggregates all convergence trials into a single CSV
4. Generates plots showing trends across molecules (gradient variance vs layers,
   convergence by molecule, etc.)

Outputs
-------
``framework/experiments/results/barren_plateau_batch/<timestamp>_batch/``

* ``run_config.json``
* ``gradients_all.csv``        -- aggregated gradient variance rows from all molecules
* ``convergence_all.csv``      -- aggregated convergence trial rows from all molecules
* ``gradient_variance_multi_mol.png``     -- gradient variance vs layers, colored by molecule
* ``convergence_comparison.png``          -- convergence curves faceted by molecule
* ``per_molecule_summary.csv`` -- summary table (one row per molecule)

Run as a script
---------------
::

    python -m experiments.barren_plateau_batch \
        --molecules ala gly ser asp \
        --algorithm hardware_efficient_vqe \
        --cs-target-qubits 6 \
        --layers 1 2 4 6 8 \
        --opt-layers 6 \
        --n-trials 5 \
        --max-iter 150
"""

from __future__ import annotations

import argparse
import csv
import json
import logging
import os
import sys
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np

_FW_DIR = Path(__file__).resolve().parent.parent
if str(_FW_DIR) not in sys.path:
    sys.path.insert(0, str(_FW_DIR))

from experiments._common import make_run_dir  # noqa: E402
from experiments.barren_plateau import (  # noqa: E402
    DEFAULT_BP_STRATEGIES,
    DEFAULT_LAYERS,
    DEFAULT_OPT_STRATEGIES,
    run_barren_plateau,
)
from experiments.noise_resilience import DATASETS2_AMINO_ACIDS  # noqa: E402

logger = logging.getLogger(__name__)


@dataclass
class BatchConfig:
    """Configuration for batch barren-plateau run."""
    molecules: Tuple[str, ...] = tuple(DATASETS2_AMINO_ACIDS)
    algorithm: str = "hardware_efficient_vqe"
    basis: str = "sto-3g"
    cs_target_qubits: Optional[int] = 6
    layers: Tuple[int, ...] = DEFAULT_LAYERS
    bp_strategies: Tuple[str, ...] = DEFAULT_BP_STRATEGIES
    n_samples: int = 30
    opt_layers: int = 6
    opt_strategies: Tuple[str, ...] = DEFAULT_OPT_STRATEGIES
    n_trials: int = 5
    max_iterations: int = 200
    optimizer: str = "COBYLA"
    random_seed: int = 42


def run_barren_plateau_batch(
    molecules: Tuple[str, ...] = tuple(DATASETS2_AMINO_ACIDS[:3]),
    algorithm: str = "hardware_efficient_vqe",
    basis: str = "sto-3g",
    cs_target_qubits: Optional[int] = 6,
    layers: Tuple[int, ...] = DEFAULT_LAYERS,
    bp_strategies: Tuple[str, ...] = DEFAULT_BP_STRATEGIES,
    n_samples: int = 30,
    opt_layers: int = 6,
    opt_strategies: Tuple[str, ...] = DEFAULT_OPT_STRATEGIES,
    n_trials: int = 5,
    max_iterations: int = 200,
    optimizer: str = "COBYLA",
    random_seed: int = 42,
    output_dir: Optional[Path] = None,
    save_plots: bool = True,
) -> Dict[str, object]:
    """
    Run barren plateau experiment across multiple molecules.
    
    Returns dict with keys: config, output_dir, gradients_csv, convergence_csv,
                            per_molecule_summary_csv, molecules_results
    """
    
    cfg = BatchConfig(
        molecules=tuple(molecules),
        algorithm=algorithm,
        basis=basis,
        cs_target_qubits=cs_target_qubits,
        layers=tuple(layers),
        bp_strategies=tuple(bp_strategies),
        n_samples=n_samples,
        opt_layers=opt_layers,
        opt_strategies=tuple(opt_strategies),
        n_trials=n_trials,
        max_iterations=max_iterations,
        optimizer=optimizer,
        random_seed=random_seed,
    )
    
    out_dir = output_dir or make_run_dir("barren_plateau_batch", "batch", algorithm)
    out_dir.mkdir(parents=True, exist_ok=True)
    logger.info("Batch output dir: %s", out_dir)
    
    # Save config
    with open(out_dir / "run_config.json", "w") as f:
        json.dump(asdict(cfg), f, indent=2, default=str)
    
    # Aggregate results across molecules
    all_grad_rows: List[Dict[str, object]] = []
    all_opt_trace_rows: List[Dict[str, object]] = []  # Long-format convergence traces
    mol_summaries: List[Dict[str, object]] = []
    molecules_results = {}
    
    for mol_idx, molecule in enumerate(molecules):
        logger.info("[%d/%d] Running barren plateau for %s", 
                   mol_idx + 1, len(molecules), molecule)
        
        try:
            result = run_barren_plateau(
                molecule=molecule,
                algorithm=algorithm,
                basis=basis,
                cs_target_qubits=cs_target_qubits,
                layers=tuple(layers),
                bp_strategies=tuple(bp_strategies),
                n_samples=n_samples,
                opt_layers=opt_layers,
                opt_strategies=tuple(opt_strategies),
                n_trials=n_trials,
                max_iterations=max_iterations,
                optimizer=optimizer,
                random_seed=random_seed,
                skip_part_a=False,
                skip_part_b=False,
                save_plots=False,  # Skip per-molecule plots
            )
            
            molecules_results[molecule] = result
            
            # Collect gradient variance rows
            for grad_row in result.get("grad_rows", []):
                row_dict = asdict(grad_row)
                row_dict["molecule"] = molecule
                all_grad_rows.append(row_dict)
            
            # Collect optimization trial rows (expand convergence_history to long-format)
            for opt_row in result.get("opt_rows", []):
                for it, energy in enumerate(opt_row.convergence_history):
                    trace_row = {
                        "molecule": molecule,
                        "strategy": opt_row.strategy,
                        "trial": opt_row.trial,
                        "n_layers": opt_row.n_layers,
                        "n_parameters": opt_row.n_parameters,
                        "iteration": it,
                        "energy": energy,
                    }
                    all_opt_trace_rows.append(trace_row)
            
            # Per-molecule summary
            cfg_result = result["config"]
            mol_summaries.append({
                "molecule": molecule,
                "n_qubits_active": cfg_result.n_qubits_active,
                "n_qubits_final": cfg_result.n_qubits_final,
                "hf_energy": cfg_result.hf_energy,
                "casci_energy": cfg_result.casci_energy,
                "n_opt_trials": len(result.get("opt_rows", [])),
                "n_gradient_samples": len(result.get("grad_rows", [])),
            })
            
        except Exception as exc:
            logger.error("Failed on %s: %s", molecule, exc, exc_info=True)
            mol_summaries.append({
                "molecule": molecule,
                "error": str(exc),
            })
    
    # Write aggregated CSVs
    gradients_csv = out_dir / "gradients_all.csv"
    convergence_traces_csv = out_dir / "convergence_traces_all.csv"
    summary_csv = out_dir / "per_molecule_summary.csv"
    
    _write_csv(gradients_csv, all_grad_rows)
    _write_csv(convergence_traces_csv, all_opt_trace_rows)
    _write_csv(summary_csv, mol_summaries)
    
    logger.info("Wrote %d gradient rows to %s", len(all_grad_rows), gradients_csv)
    logger.info("Wrote %d convergence trace rows to %s", len(all_opt_trace_rows), convergence_traces_csv)
    logger.info("Wrote %d molecule summaries to %s", len(mol_summaries), summary_csv)
    
    # Generate batch plots
    if save_plots:
        _plot_multi_molecule_gradients(gradients_csv, out_dir)
        _plot_convergence_comparison(convergence_traces_csv, out_dir)
        _plot_strategy_comparison(gradients_csv, out_dir)
    
    return {
        "config": cfg,
        "output_dir": out_dir,
        "gradients_csv": gradients_csv,
        "convergence_traces_csv": convergence_traces_csv,
        "summary_csv": summary_csv,
        "molecules_results": molecules_results,
        "all_grad_rows": all_grad_rows,
        "all_opt_trace_rows": all_opt_trace_rows,
        "mol_summaries": mol_summaries,
    }


def _write_csv(filepath: Path, rows: List[Dict[str, object]]) -> None:
    """Write list of dicts to CSV with consistent ordering."""
    if not rows:
        logger.warning("No rows to write to %s", filepath)
        return
    
    # Collect all unique fieldnames from all rows
    fieldnames = set()
    for row in rows:
        fieldnames.update(row.keys())
    fieldnames = sorted(fieldnames)
    
    with open(filepath, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames, restval="")
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


def _plot_multi_molecule_gradients(csv_path: Path, output_dir: Path) -> None:
    """Plot gradient variance vs layers, with one line per molecule."""
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except Exception as exc:
        logger.warning("matplotlib unavailable: %s", exc)
        return
    
    import csv as _csv
    
    data_by_mol_strategy: Dict[Tuple[str, str], List] = {}
    
    with open(csv_path, newline="") as f:
        reader = _csv.DictReader(f)
        for row in reader:
            mol = row.get("molecule", "unknown")
            strategy = row.get("strategy", "unknown")
            key = (mol, strategy)
            if key not in data_by_mol_strategy:
                data_by_mol_strategy[key] = []
            data_by_mol_strategy[key].append(row)
    
    if not data_by_mol_strategy:
        logger.warning("No data in %s for plotting", csv_path)
        return
    
    # Get unique molecules and strategies
    molecules = sorted(set(k[0] for k in data_by_mol_strategy.keys()))
    strategies = sorted(set(k[1] for k in data_by_mol_strategy.keys()))
    
    # Create subplots: one per strategy
    fig, axes = plt.subplots(1, len(strategies), figsize=(5 * len(strategies), 5))
    if len(strategies) == 1:
        axes = [axes]
    
    colors = plt.get_cmap("tab10")
    
    for ax_idx, strategy in enumerate(strategies):
        ax = axes[ax_idx]
        
        for mol_idx, mol in enumerate(molecules):
            key = (mol, strategy)
            rows = data_by_mol_strategy.get(key, [])
            if not rows:
                continue
            
            rows_sorted = sorted(rows, key=lambda r: int(r.get("n_layers", 0)))
            xs = np.array([int(r["n_layers"]) for r in rows_sorted], dtype=float)
            ys = np.array([float(r["var_grad"]) for r in rows_sorted], dtype=float)
            
            color = colors(mol_idx % colors.N)
            ax.plot(xs, ys, marker="o", label=mol, color=color, linewidth=2)
        
        ax.set_yscale("log")
        ax.set_xlabel("# layers", fontsize=11)
        ax.set_ylabel("Var[∂E/∂θ]", fontsize=11)
        ax.set_title(f"Strategy: {strategy}", fontsize=12)
        ax.grid(True, which="both", alpha=0.3)
        if ax_idx == 0:
            ax.legend(fontsize=9, loc="best")
    
    fig.suptitle("Barren Plateau: Gradient Variance vs Depth (multi-molecule)", fontsize=14)
    fig.tight_layout()
    fig.savefig(output_dir / "gradient_variance_multi_mol.png", dpi=150)
    plt.close(fig)
    logger.info("Saved gradient variance plot")


def _plot_strategy_comparison(csv_path: Path, output_dir: Path) -> None:
    """Plot mean gradient variance vs layers, with one line per strategy (averaged across molecules)."""
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except Exception as exc:
        logger.warning("matplotlib unavailable: %s", exc)
        return

    import csv as _csv
    from collections import defaultdict

    # data[strategy][n_layers] -> list of var_grad
    data: Dict[str, Dict[int, List[float]]] = defaultdict(lambda: defaultdict(list))

    with open(csv_path, newline="") as f:
        reader = _csv.DictReader(f)
        for row in reader:
            try:
                strat = row.get("strategy", "unknown")
                n_layers = int(row.get("n_layers", 0))
                varg = float(row.get("var_grad", "nan"))
            except Exception:
                continue
            data[strat][n_layers].append(varg)

    if not data:
        logger.warning("No data in %s for strategy comparison plot", csv_path)
        return

    strategies = sorted(data.keys())
    all_layers = sorted({l for strat in data.values() for l in strat.keys()})

    fig, ax = plt.subplots(1, 1, figsize=(8, 5))

    cmap = plt.get_cmap("tab10")
    for idx, strat in enumerate(strategies):
        means = []
        xs = []
        for l in all_layers:
            vals = data[strat].get(l, [])
            if vals:
                means.append(np.mean(vals))
                xs.append(l)
        if not xs:
            continue
        ax.plot(xs, means, marker="o", label=strat, color=cmap(idx % cmap.N), linewidth=2)

    ax.set_yscale("log")
    ax.set_xlabel("# layers", fontsize=11)
    ax.set_ylabel("Mean Var[∂E/∂θ] (avg across molecules)", fontsize=11)
    ax.set_title("Strategy Comparison: Mean Gradient Variance vs Depth", fontsize=12)
    ax.grid(True, which="both", alpha=0.3)
    ax.legend(fontsize=9, loc="best")

    fig.tight_layout()
    out_path = output_dir / "strategy_comparison.png"
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    logger.info("Saved strategy comparison plot to %s", out_path)


def _plot_convergence_comparison(csv_path: Path, output_dir: Path) -> None:
    """Plot convergence curves grouped by molecule."""
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except Exception as exc:
        logger.warning("matplotlib unavailable: %s", exc)
        return
    
    import csv as _csv
    
    # Group by molecule
    data_by_mol: Dict[str, List] = {}
    
    with open(csv_path, newline="") as f:
        reader = _csv.DictReader(f)
        for row in reader:
            mol = row.get("molecule", "unknown")
            if mol not in data_by_mol:
                data_by_mol[mol] = []
            data_by_mol[mol].append(row)
    
    if not data_by_mol:
        logger.warning("No convergence data in %s", csv_path)
        return
    
    molecules = sorted(data_by_mol.keys())
    # Use up to 4 columns so plots can be 4 in a row for wider layouts
    n_cols = min(4, len(molecules))
    n_rows = (len(molecules) + n_cols - 1) // n_cols
    
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(5 * n_cols, 4 * n_rows))
    if n_rows == 1 and n_cols == 1:
        axes = np.array([[axes]])
    elif n_rows == 1:
        axes = np.array([axes])
    elif n_cols == 1:
        axes = np.array([[ax] for ax in axes])
    else:
        axes = axes
    
    colors = plt.get_cmap("Set2")
    
    for mol_idx, mol in enumerate(molecules):
        row_idx = mol_idx // n_cols
        col_idx = mol_idx % n_cols
        ax = axes[row_idx][col_idx]
        
        rows = data_by_mol[mol]
        
        # Group by strategy
        by_strategy: Dict[str, List] = {}
        for row in rows:
            strategy = row.get("strategy", "unknown")
            if strategy not in by_strategy:
                by_strategy[strategy] = []
            by_strategy[strategy].append(row)
        
        # Plot convergence for each strategy
        for strat_idx, (strategy, strat_rows) in enumerate(sorted(by_strategy.items())):
            # Sort by trial and iteration
            strat_rows_sorted = sorted(
                strat_rows,
                key=lambda r: (int(r.get("trial_idx", 0)), int(r.get("iteration", 0)))
            )
            
            # Group by trial
            by_trial: Dict[int, List] = {}
            for row in strat_rows_sorted:
                trial_idx = int(row.get("trial_idx", 0))
                if trial_idx not in by_trial:
                    by_trial[trial_idx] = []
                by_trial[trial_idx].append(row)
            
            # Plot each trial lightly, then mean
            color = colors(strat_idx % colors.N)
            energies_by_iter: Dict[int, List[float]] = {}
            
            for trial_idx, trial_rows in by_trial.items():
                iterations = np.array([float(r["iteration"]) for r in trial_rows], dtype=float)
                energies = np.array([float(r["energy"]) for r in trial_rows], dtype=float)
                
                ax.plot(iterations, energies, color=color, alpha=0.2, linewidth=0.5)
                
                # Accumulate energies for mean
                for it, en in zip(iterations, energies):
                    it_int = int(it)
                    if it_int not in energies_by_iter:
                        energies_by_iter[it_int] = []
                    energies_by_iter[it_int].append(en)
            
            # Plot mean
            if energies_by_iter:
                mean_iters = sorted(energies_by_iter.keys())
                mean_energies = [np.mean(energies_by_iter[it]) for it in mean_iters]
                ax.plot(mean_iters, mean_energies, marker="o", color=color, 
                       linewidth=2.5, label=strategy, markersize=4)
        
        ax.set_xlabel("Iteration", fontsize=10)
        ax.set_ylabel("Energy (Ha)", fontsize=10)
        ax.set_title(f"{mol}", fontsize=11, fontweight="bold")
        ax.grid(True, alpha=0.3)
        ax.legend(fontsize=9)
    
    # Hide unused subplots
    for idx in range(len(molecules), len(axes.flat)):
        axes.flat[idx].set_visible(False)
    
    fig.suptitle("Convergence Comparison by Molecule and Strategy", fontsize=14)
    fig.tight_layout()
    fig.savefig(output_dir / "convergence_comparison.png", dpi=150)
    plt.close(fig)
    logger.info("Saved convergence comparison plot")


def _build_argparser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description="Barren plateau batch experiment across multiple molecules.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument("--molecules", nargs="+", default=list(DATASETS2_AMINO_ACIDS[:3]),
                   help="Molecules to analyze from datasets2/.")
    p.add_argument("--algorithm", default="hardware_efficient_vqe")
    p.add_argument("--basis", default="sto-3g")
    p.add_argument("--cs-target-qubits", type=int, default=6,
                   help="Active-space target qubits.")
    p.add_argument("--layers", nargs="+", type=int, default=list(DEFAULT_LAYERS),
                   help="Ansatz depths to test for gradient variance.")
    p.add_argument("--bp-strategies", nargs="+", default=list(DEFAULT_BP_STRATEGIES),
                   help="Initialization strategies for gradient variance measurement.")
    p.add_argument("--n-samples", type=int, default=30,
                   help="Number of parameter samples per (depth, strategy) pair.")
    p.add_argument("--opt-layers", type=int, default=6,
                   help="Ansatz depth for optimization convergence test.")
    p.add_argument("--opt-strategies", nargs="+", default=list(DEFAULT_OPT_STRATEGIES),
                   help="Initialization strategies for optimization convergence.")
    p.add_argument("--n-trials", type=int, default=5,
                   help="Number of independent optimizations per strategy.")
    p.add_argument("--max-iter", type=int, default=200)
    p.add_argument("--optimizer", default="COBYLA")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--output-dir", type=Path, default=None)
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
    
    out = run_barren_plateau_batch(
        molecules=tuple(args.molecules),
        algorithm=args.algorithm,
        basis=args.basis,
        cs_target_qubits=args.cs_target_qubits if args.cs_target_qubits > 0 else None,
        layers=tuple(args.layers),
        bp_strategies=tuple(args.bp_strategies),
        n_samples=args.n_samples,
        opt_layers=args.opt_layers,
        opt_strategies=tuple(args.opt_strategies),
        n_trials=args.n_trials,
        max_iterations=args.max_iter,
        optimizer=args.optimizer,
        random_seed=args.seed,
        output_dir=args.output_dir,
        save_plots=not args.no_plots,
    )
    
    print("\n" + "=" * 70)
    print("BARREN PLATEAU BATCH STUDY")
    print("=" * 70)
    print(f"Molecules: {', '.join(out['config'].molecules)}")
    print(f"Algorithm: {out['config'].algorithm}")
    print(f"Basis: {out['config'].basis}")
    print(f"CS target qubits: {out['config'].cs_target_qubits}")
    print(f"\nOutput directory: {out['output_dir']}")
    print(f"Gradients CSV: {out['gradients_csv']}")
    print(f"Convergence traces CSV: {out['convergence_traces_csv']}")
    print(f"Summary CSV: {out['summary_csv']}")
    print(f"\nMolecule summaries:")
    for summary in out["mol_summaries"]:
        if "error" in summary:
            print(f"  {summary['molecule']}: ERROR - {summary['error']}")
        else:
            print(f"  {summary['molecule']}: "
                  f"{summary['n_qubits_final']} qubits, "
                  f"HF={summary['hf_energy']:.6f}, "
                  f"CASCI={summary['casci_energy']:.6f}")
    print("=" * 70 + "\n")


if __name__ == "__main__":
    main()
