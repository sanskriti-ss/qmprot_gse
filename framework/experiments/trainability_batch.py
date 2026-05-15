"""Trainability batch runner + aggregate shaded plot
================================================

Wrapper around the existing `run_trainability_batch` that runs the
experiment across multiple molecules and then produces an aggregated
plot which shows per-algorithm mean ±1σ bands and thin per-molecule
traces for quick comparison.

Run as a module::

    python -m experiments.trainability_batch --batch-amino-acids --param-targets 10 30 --max-iter 80

Outputs
-------
- ``results_all.csv`` -- copied from the batch output
- ``convergence_traces_all.csv`` -- copied from the batch output
- ``aggregate_trainability_shaded.png`` -- summary plot
"""

from __future__ import annotations

import argparse
import json
import logging
from collections import defaultdict
from pathlib import Path
from typing import Tuple, Optional

import csv
import numpy as np

_FW_DIR = Path(__file__).resolve().parent

import sys
if str(_FW_DIR.parent) not in sys.path:
    sys.path.insert(0, str(_FW_DIR.parent))

from experiments.trainability import run_trainability_batch, TrainabilityBatchConfig  # noqa: E402
from experiments.noise_resilience import DATASETS2_AMINO_ACIDS  # noqa: E402

logger = logging.getLogger(__name__)


def _aggregate_and_plot(results_csv: Path, out_dir: Path) -> None:
    if not results_csv.exists():
        logger.error("Missing results CSV: %s", results_csv)
        return

    rows = []
    with open(results_csv, newline="") as f:
        reader = csv.DictReader(f)
        for r in reader:
            # normalize numeric types
            for k in ("actual_n_params", "n_cost_evals"):
                if r.get(k):
                    r[k] = int(float(r[k]))
            for k in ("error_vs_casci",):
                if r.get(k):
                    r[k] = float(r[k])
            rows.append(r)

    # Organize: by (algorithm, actual_n_params) collect values
    stats = defaultdict(lambda: {"n_cost": [], "err": [], "by_mol": defaultdict(lambda: {"n_cost": [], "err": []})})

    molecules = sorted({r.get("molecule") for r in rows if r.get("molecule")})

    for r in rows:
        algo = r["algorithm"]
        p = r["actual_n_params"]
        stats[(algo, p)]["n_cost"].append(r.get("n_cost_evals", 0))
        stats[(algo, p)]["err"].append(r.get("error_vs_casci", 0.0))
        mol = r.get("molecule")
        if mol:
            stats[(algo, p)]["by_mol"][mol]["n_cost"].append(r.get("n_cost_evals", 0))
            stats[(algo, p)]["by_mol"][mol]["err"].append(r.get("error_vs_casci", 0.0))

    algos = sorted({k[0] for k in stats.keys()})
    all_params = sorted({k[1] for k in stats.keys()})

    # Plot
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except Exception as exc:
        logger.error("matplotlib unavailable: %s", exc)
        return

    fig, axes = plt.subplots(2, 1, figsize=(10, 8), sharex=True)
    cmap = plt.get_cmap("tab10")

    for i, algo in enumerate(algos):
        xs = []
        mean_n = []
        std_n = []
        mean_err = []
        std_err = []
        for p in all_params:
            key = (algo, p)
            vals_n = np.array(stats[key]["n_cost"], dtype=float) if stats[key]["n_cost"] else np.array([])
            vals_e = np.array(stats[key]["err"], dtype=float) if stats[key]["err"] else np.array([])
            if vals_n.size == 0:
                continue
            xs.append(p)
            mean_n.append(vals_n.mean())
            std_n.append(vals_n.std())
            mean_err.append(vals_e.mean() if vals_e.size else 0.0)
            std_err.append(vals_e.std() if vals_e.size else 0.0)

        if not xs:
            continue

        color = cmap(i % cmap.N)

        # top: cost evals (log)
        axes[0].plot(xs, mean_n, marker="o", color=color, linewidth=2, label=algo)
        axes[0].fill_between(xs, np.maximum(1e-1, np.array(mean_n) - np.array(std_n)), np.array(mean_n) + np.array(std_n),
                             color=color, alpha=0.2)

        # draw per-molecule thin traces
        for mol in molecules:
            mol_vals = []
            mol_xs = []
            for p in xs:
                key = (algo, p)
                mol_n = stats[key]["by_mol"].get(mol, {}).get("n_cost", [])
                if mol_n:
                    mol_xs.append(p)
                    mol_vals.append(np.mean(mol_n))
            if mol_xs:
                axes[0].plot(mol_xs, mol_vals, color=color, linewidth=0.7, alpha=0.25)

        # bottom: error vs CASCI
        axes[1].plot(xs, mean_err, marker="o", color=color, linewidth=2, label=algo)
        axes[1].fill_between(xs, np.array(mean_err) - np.array(std_err), np.array(mean_err) + np.array(std_err),
                             color=color, alpha=0.2)
        for mol in molecules:
            mol_vals = []
            mol_xs = []
            for p in xs:
                key = (algo, p)
                mol_e = stats[key]["by_mol"].get(mol, {}).get("err", [])
                if mol_e:
                    mol_xs.append(p)
                    mol_vals.append(np.mean(mol_e))
            if mol_xs:
                axes[1].plot(mol_xs, mol_vals, color=color, linewidth=0.7, alpha=0.25)

    axes[0].set_yscale("log")
    axes[0].set_ylabel("# cost-fn evals (mean ±1σ)")
    axes[0].set_title("Trainability: cost evaluations (per-algo mean ±1σ) with per-molecule traces")
    axes[0].grid(True, which="both", alpha=0.3)
    axes[0].legend(fontsize=9)

    axes[1].set_xlabel("actual n_parameters")
    axes[1].set_ylabel("error vs CASCI (Ha)")
    axes[1].axhline(0, color="gray", linewidth=0.8, linestyle="--")
    axes[1].grid(True, alpha=0.3)
    axes[1].legend(fontsize=9)

    out_png = out_dir / "aggregate_trainability_shaded.png"
    fig.tight_layout()
    fig.savefig(out_png, dpi=150)
    plt.close(fig)
    logger.info("Saved aggregate shaded plot to %s", out_png)


def _build_argparser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser()
    p.add_argument("--molecules", nargs="+", default=None)
    p.add_argument("--batch-amino-acids", action="store_true")
    p.add_argument("--param-targets", nargs="+", type=int, default=[10, 30])
    p.add_argument("--algorithms", nargs="+", default=["qubit_adapt_vqe", "hardware_efficient_vqe"])
    p.add_argument("--max-iter", type=int, default=80)
    p.add_argument("--cs-target-qubits", type=int, default=0)
    p.add_argument("--output-dir", type=Path, default=None)
    p.add_argument("--no-plots", action="store_true")
    p.add_argument("-v", "--verbose", action="store_true")
    return p


def main() -> None:
    args = _build_argparser().parse_args()
    logging.basicConfig(level=logging.DEBUG if args.verbose else logging.INFO,
                        format="%(asctime)s %(name)s [%(levelname)s] %(message)s")

    molecules = tuple(DATASETS2_AMINO_ACIDS) if args.batch_amino_acids else tuple(args.molecules or [])
    cs_target = args.cs_target_qubits if args.cs_target_qubits > 0 else None

    out = run_trainability_batch(
        molecules=molecules,
        basis="sto-3g",
        cs_target_qubits=cs_target,
        algorithms=tuple(args.algorithms),
        param_targets=tuple(args.param_targets),
        max_iterations=args.max_iter,
        random_seed=42,
        output_dir=args.output_dir,
        save_plots=False,
    )

    results_csv = out["output_dir"] / "results_all.csv"
    # Also copy any traces file exists
    traces_csv = out["output_dir"] / "convergence_traces_all.csv"

    # Generate aggregate shaded plot (per-algo mean ±1σ + per-molecule traces)
    _aggregate_and_plot(results_csv, out["output_dir"])

    print("Done. Outputs in:", out["output_dir"])


if __name__ == "__main__":
    main()
