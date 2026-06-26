"""
Run barren-plateau Part A (gradient variance vs depth) for all molecules
up to 16 layers, then produce a publication-quality aggregate figure.

Usage:
    python run_barren_plateau_all_molecules.py
"""
from __future__ import annotations

import sys
from pathlib import Path
from datetime import datetime

_FW_DIR = Path(__file__).resolve().parent / "framework"
if str(_FW_DIR) not in sys.path:
    sys.path.insert(0, str(_FW_DIR))

import matplotlib.pyplot as plt
import pandas as pd

from experiments.barren_plateau import run_barren_plateau
from experiments.trainability import discover_trainability_molecules


def main() -> None:
    root = _FW_DIR / "experiments" / "results" / "barren_plateau"
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_root = root / f"{timestamp}_all_molecules_gradient_variance"
    output_root.mkdir(parents=True, exist_ok=True)

    molecules = discover_trainability_molecules()
    print(f"Found {len(molecules)} molecules: {molecules}\n")

    all_rows = []

    for molecule in molecules:
        print(f"[{molecule}] Running Part A (gradient variance)...")
        try:
            out = run_barren_plateau(
                molecule=molecule,
                algorithm="hardware_efficient_vqe",
                basis="sto-3g",
                cs_target_qubits=4,
                layers=(1, 2, 4, 8, 16),
                bp_strategies=("near_identity", "random_uniform", "small_random"),
                n_samples=30,
                gradient_index=0,
                opt_layers=6,
                opt_strategies=("near_identity", "random_uniform", "small_random"),
                n_trials=5,
                max_iterations=200,
                convergence_threshold=1e-10,
                optimizer="COBYLA",
                random_seed=42,
                output_dir=output_root / molecule,
                save_plots=True,
                skip_part_a=False,
                skip_part_b=True,
            )
        except Exception as exc:
            print(f"  [WARN] {molecule} failed: {exc}")
            continue

        grad_csv = Path(out["output_dir"]) / "gradient_variance.csv"
        if not grad_csv.is_file():
            print(f"  [WARN] gradient_variance.csv missing for {molecule}, skipping")
            continue

        df = pd.read_csv(grad_csv)
        df.insert(0, "molecule", molecule)
        all_rows.append(df)
        print(f"  Done. {len(df)} rows.")

    if not all_rows:
        print("No results collected — exiting.")
        return

    combined = pd.concat(all_rows, ignore_index=True)
    combined.to_csv(output_root / "all_molecules_gradient_variance.csv", index=False)

    summary = (
        combined.groupby(["n_layers", "strategy"], as_index=False)
        .agg(
            mean_var_grad=("var_grad", "mean"),
            std_var_grad=("var_grad", "std"),
            mean_abs_grad=("mean_abs_grad", "mean"),
            n_molecules=("molecule", "nunique"),
        )
    )
    summary.to_csv(output_root / "all_molecules_gradient_variance_summary.csv", index=False)

    # ── Publication-quality figure ──────────────────────────────────────────
    strategy_order = ["near_identity", "random_uniform", "small_random"]
    colors = {
        "near_identity":  "#2F6FAE",
        "random_uniform": "#D97732",
        "small_random":   "#4C9A3F",
    }

    plt.rcParams.update({
        "font.family":      "serif",
        "font.size":        9,
        "axes.labelsize":   9,
        "axes.titlesize":   10,
        "legend.fontsize":  8,
        "xtick.labelsize":  8,
        "ytick.labelsize":  8,
        "savefig.dpi":      600,
        "axes.linewidth":   0.7,
        "xtick.major.width": 0.6,
        "ytick.major.width": 0.6,
    })

    fig, ax = plt.subplots(figsize=(6.8, 4.2))

    for strategy in strategy_order:
        part = summary[summary["strategy"] == strategy].sort_values("n_layers")
        ax.plot(
            part["n_layers"],
            part["mean_var_grad"],
            marker="o",
            markersize=4,
            linewidth=1.6,
            color=colors[strategy],
            label=strategy,
        )

    ax.set_yscale("log")
    ax.set_xlabel("# layers")
    ax.set_ylabel(
        r"Mean $\mathrm{Var}[\partial E/\partial\theta]$ (avg across molecules)"
    )
    ax.set_title("Strategy Comparison: Mean Gradient Variance vs Depth")
    ax.set_xticks([1, 2, 4, 8, 16])
    ax.grid(True, which="both", alpha=0.25, linewidth=0.5)
    ax.legend(frameon=True, loc="best")

    fig.tight_layout()
    fig.savefig(
        output_root / "mean_gradient_variance_vs_depth_all_molecules.pdf",
        bbox_inches="tight",
    )
    fig.savefig(
        output_root / "mean_gradient_variance_vs_depth_all_molecules.png",
        bbox_inches="tight",
    )
    plt.close(fig)

    print(f"\nDone. Outputs saved to:\n  {output_root}")


if __name__ == "__main__":
    main()
