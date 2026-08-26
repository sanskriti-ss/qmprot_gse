"""
paper_combined_figure.py
========================
Generate a single combined publication figure with three panels:
  (a) Barren plateau – gradient variance vs circuit depth
  (b) Noise resilience – parameter drift heatmaps (3 noise models)
  (c) Trainability – accuracy vs cost tradeoff scatter

Usage:
    cd framework
    python plots/paper_combined_figure.py
"""

from __future__ import annotations

from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import matplotlib.colors as mcolors
import numpy as np
import pandas as pd

# ── Paths ────────────────────────────────────────────────────────────────
FW = Path(__file__).resolve().parent.parent
RESULTS = FW / "experiments" / "results"

BP_DIR = RESULTS / "barren_plateau_batch" / "20260514_153420_batch_hardware_efficient_vqe"
NOISE_DIR = RESULTS / "noise_resilience_batch" / "20260509_172216_nq6_hardware_efficient_vqe_batch"
TRAIN_DIR = RESULTS / "trainability_batch" / "20260514_153030_batch_comparison"

OUT_DIR = FW / "plots" / "poster_figures"
OUT_DIR.mkdir(parents=True, exist_ok=True)

# ── Purple / Sunset palette ──────────────────────────────────────────────
PURPLE_DEEP   = "#3b0764"
PURPLE_MID    = "#7c3aed"
MAGENTA       = "#c026d3"
CORAL         = "#f43f5e"
AMBER         = "#f59e0b"
GOLD_LIGHT    = "#fbbf24"

# Strategy colours (purple, orange, muted pink)
STRATEGY_COLORS = {
    "near_identity":  "#7c3aed",
    "random_uniform": "#f97316",
    "small_random":   "#d4779c",
}
STRATEGY_LABELS = {
    "near_identity":  "Near-identity",
    "random_uniform": "Random uniform",
    "small_random":   "Small random",
}

# Algorithm colours
ALGO_COLORS = {
    "qubit_adapt_vqe":        "#7c3aed",
    "hardware_efficient_vqe": "#f97316",
}
ALGO_MARKERS = {
    "qubit_adapt_vqe":        "o",
    "hardware_efficient_vqe": "s",
}
ALGO_LABELS = {
    "qubit_adapt_vqe":        "ADAPT-VQE",
    "hardware_efficient_vqe": "HE-VQE",
}

# Noise model labels
NOISE_LABELS = {
    "amplitude_damping": "Amplitude damping",
    "depolarizing":      "Depolarizing",
    "phaseflip":         "Phase flip",
}

# Custom sunset colormap for heatmaps (dark purple → lavender → warm rose → soft gold)
_sunset_colors = ["#1e1b4b", "#4c1d95", "#7e57a0", "#a678b8",
                   "#c9849e", "#d9977e", "#e6b566", "#f0d48a"]
SUNSET_CMAP = mcolors.LinearSegmentedColormap.from_list("sunset", _sunset_colors, N=256)


# ── RC params ────────────────────────────────────────────────────────────
PAPER_RC = {
    "font.family":       "sans-serif",
    "font.size":         9,
    "axes.titlesize":    10,
    "axes.labelsize":    9,
    "xtick.labelsize":   8,
    "ytick.labelsize":   8,
    "legend.fontsize":   8,
    "lines.linewidth":   1.8,
    "axes.spines.top":   False,
    "axes.spines.right": False,
    "axes.grid":         True,
    "grid.alpha":        0.2,
    "grid.linewidth":    0.5,
    "figure.dpi":        300,
}


def _panel_label(ax, letter, x=-0.10, y=1.06):
    ax.text(x, y, f"({letter})", transform=ax.transAxes,
            fontsize=12, fontweight="bold", va="top", ha="right",
            color=PURPLE_DEEP)


# ═════════════════════════════════════════════════════════════════════════
# Panel (a): Barren plateau – gradient variance vs circuit depth
# ═════════════════════════════════════════════════════════════════════════

def plot_barren_plateau(ax):
    gdf = pd.read_csv(BP_DIR / "gradients_all.csv")
    gdf["var_grad"] = pd.to_numeric(gdf["var_grad"], errors="coerce")

    strategies = ["small_random", "random_uniform", "near_identity"]
    layers = sorted(gdf["n_layers"].unique())

    for strategy in strategies:
        sub = gdf[gdf["strategy"] == strategy]
        grp = sub.groupby("n_layers")["var_grad"]
        xs = np.array(layers, dtype=float)
        ys = np.array([grp.mean().get(l, np.nan) for l in layers])
        ye = np.array([grp.sem().get(l, np.nan) for l in layers])
        color = STRATEGY_COLORS[strategy]
        label = STRATEGY_LABELS[strategy]
        ax.plot(xs, ys, marker="o", color=color, label=label,
                markersize=5, zorder=3)
        ax.fill_between(xs, ys - ye, ys + ye, alpha=0.15, color=color)

    ax.set_yscale("log")
    ax.set_xlabel("Ansatz depth (# layers)")
    ax.set_ylabel(r"Mean Var[$\partial E / \partial \theta$]")
    ax.set_title("Gradient variance vs circuit depth", fontweight="semibold")
    ax.set_xticks(layers)
    ax.set_xticklabels([str(int(l)) for l in layers])
    ax.legend(framealpha=0.9, edgecolor="none")

    _panel_label(ax, "a")


# ═════════════════════════════════════════════════════════════════════════
# Panel (b): Noise resilience – three side-by-side heatmaps
# ═════════════════════════════════════════════════════════════════════════

def plot_noise_heatmaps(axes):
    """Plot three heatmaps on the given list of 3 axes."""
    df = pd.read_csv(NOISE_DIR / "batch_all_results.csv")
    df = df[df["noise_model"].str.lower() != "none"].copy()
    df["noise_strength"] = pd.to_numeric(df["noise_strength"], errors="coerce")
    df["param_drift_l2"] = pd.to_numeric(df["param_drift_l2"], errors="coerce")

    models = ["amplitude_damping", "depolarizing", "phaseflip"]
    strengths = sorted(df["noise_strength"].dropna().unique())
    strengths = [s for s in strengths if s > 0]
    mol_order = sorted(df["molecule"].unique())

    vmax = 0
    for model in models:
        sub = df[(df["noise_model"] == model) & (df["noise_strength"] > 0)]
        if len(sub):
            vmax = max(vmax, sub["param_drift_l2"].max())
    vmax = vmax * 1.02

    for i, (model, ax) in enumerate(zip(models, axes)):
        sub = df[df["noise_model"] == model]
        pivot = sub.pivot_table(
            index="molecule", columns="noise_strength",
            values="param_drift_l2", aggfunc="mean",
        )
        pivot = pivot.reindex(index=mol_order, columns=strengths)

        im = ax.imshow(
            pivot.values, aspect="auto", cmap=SUNSET_CMAP,
            vmin=0, vmax=vmax,
            interpolation="nearest",
        )
        ax.set_xticks(range(len(strengths)))
        ax.set_xticklabels([f"{s:g}" for s in strengths], fontsize=7)
        ax.set_yticks(range(len(mol_order)))
        if i == 0:
            ax.set_yticklabels(mol_order, fontsize=7)
        else:
            ax.set_yticklabels([])

        ax.set_title(NOISE_LABELS[model], fontsize=9, fontweight="semibold")
        ax.grid(False)
        ax.spines[:].set_visible(True)
        for spine in ax.spines.values():
            spine.set_linewidth(0.5)

        if i == 0:
            ax.set_ylabel("Molecule")
        if i == 1:
            ax.set_xlabel("Noise strength  $p$")

    cbar = plt.colorbar(im, ax=axes, shrink=0.82, pad=0.02,
                        label=r"$\|\Delta\theta\|_2$")
    cbar.ax.tick_params(labelsize=7)

    _panel_label(axes[0], "b")


# ═════════════════════════════════════════════════════════════════════════
# Panel (c): Trainability – accuracy vs cost tradeoff
# ═════════════════════════════════════════════════════════════════════════

def plot_trainability(ax):
    """Scatter plot: energy error vs cost-function evaluations.

    Each point is one molecule. This shows the accuracy-cost tradeoff:
    ADAPT-VQE reaches much lower error but at higher computational cost.
    Grey lines connect the same molecule across algorithms.
    """
    df = pd.read_csv(TRAIN_DIR / "results_all.csv")
    df["n_cost_evals"] = pd.to_numeric(df["n_cost_evals"], errors="coerce")
    df["error_vs_casci"] = pd.to_numeric(df["error_vs_casci"], errors="coerce")
    df["abs_error"] = df["error_vs_casci"].abs()

    algos = ["qubit_adapt_vqe", "hardware_efficient_vqe"]
    present = [a for a in algos if a in df["algorithm"].unique()]

    # Draw connector lines between same molecule
    if len(present) == 2:
        sub0 = df[df["algorithm"] == present[0]].set_index("molecule")
        sub1 = df[df["algorithm"] == present[1]].set_index("molecule")
        for mol in sub0.index:
            if mol in sub1.index:
                ax.plot(
                    [sub0.loc[mol, "n_cost_evals"], sub1.loc[mol, "n_cost_evals"]],
                    [sub0.loc[mol, "abs_error"], sub1.loc[mol, "abs_error"]],
                    color="#d1d5db", linewidth=0.7, zorder=1,
                )

    # Scatter each algorithm
    for algo in present:
        sub = df[df["algorithm"] == algo]
        color = ALGO_COLORS.get(algo, "grey")
        marker = ALGO_MARKERS.get(algo, "o")
        label = ALGO_LABELS.get(algo, algo)
        ax.scatter(
            sub["n_cost_evals"], sub["abs_error"],
            c=color, marker=marker, s=40, alpha=0.85,
            label=label, zorder=3, edgecolors="white", linewidths=0.4,
        )

    # Chemical accuracy reference line
    CHEM_ACC = 1.6e-3
    ax.axhline(CHEM_ACC, color="#a855f7", linestyle="--", linewidth=1.2,
               alpha=0.7, zorder=2, label="Chem. accuracy")

    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlabel("Cost-function evaluations")
    ax.set_ylabel("|Energy error|  (Ha)")
    ax.set_title("Accuracy–cost tradeoff", fontweight="semibold")
    ax.legend(framealpha=0.9, edgecolor="none", loc="lower left", fontsize=7.5)

    _panel_label(ax, "c")


# ═════════════════════════════════════════════════════════════════════════
# Compose the combined figure
# ═════════════════════════════════════════════════════════════════════════

def _build_figure():
    """Build and return the combined figure."""
    fig = plt.figure(figsize=(11, 8.5))

    # Tight layout:
    #   Top row:     (a) barren plateau   |  (c) trainability
    #   Bottom row:  (b) noise heatmaps (3 sub-panels, spanning full width)
    gs = gridspec.GridSpec(
        2, 2, figure=fig,
        height_ratios=[0.85, 1.2],
        width_ratios=[1, 1],
        hspace=0.30, wspace=0.30,
    )

    # (a) Top-left
    ax_bp = fig.add_subplot(gs[0, 0])
    plot_barren_plateau(ax_bp)

    # (c) Top-right
    ax_train = fig.add_subplot(gs[0, 1])
    plot_trainability(ax_train)

    # (b) Bottom row: noise heatmaps (3 side-by-side)
    gs_noise = gs[1, :].subgridspec(1, 3, wspace=0.08)
    ax_n0 = fig.add_subplot(gs_noise[0])
    ax_n1 = fig.add_subplot(gs_noise[1])
    ax_n2 = fig.add_subplot(gs_noise[2])
    plot_noise_heatmaps([ax_n0, ax_n1, ax_n2])

    fig.suptitle(
        "VQE Performance on Amino-Acid Hamiltonians",
        fontsize=14, fontweight="bold", color=PURPLE_DEEP, y=1.00,
    )
    return fig


def main():
    for ext in [".png", ".pdf"]:
        with plt.rc_context(PAPER_RC):
            fig = _build_figure()
            out = OUT_DIR / f"paper_combined_figure{ext}"
            fig.savefig(
                out, dpi=300, bbox_inches="tight",
                facecolor="white", edgecolor="none",
            )
            plt.close(fig)
        print(f"Saved → {out}")


if __name__ == "__main__":
    main()
