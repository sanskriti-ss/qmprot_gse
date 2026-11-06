"""
make_poster_figures.py
======================
Generate four poster-quality figures, one per experiment.

Usage
-----
    cd framework
    python plots/make_poster_figures.py

    # Override individual data directories:
    python plots/make_poster_figures.py \
        --noise-dir experiments/results/noise_resilience_batch/20260509_172216_nq6_hardware_efficient_vqe_batch \
        --bp-dir    experiments/results/barren_plateau_batch/20260514_153420_batch_hardware_efficient_vqe \
        --train-dir experiments/results/trainability_batch/20260514_153030_batch_comparison \
        --acc-dir   experiments/results/accuracy_vs_params_batch/20260514_154726_batch \
        --out-dir   plots/poster_figures

All four --*-dir arguments auto-discover the *newest* matching folder under
``experiments/results/`` when omitted, so the script stays evergreen as new
runs land.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Optional

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import numpy as np
import pandas as pd

# ── Poster-style defaults ─────────────────────────────────────────────────
POSTER_RC = {
    "font.family": "sans-serif",
    "font.size": 13,
    "axes.titlesize": 14,
    "axes.labelsize": 13,
    "xtick.labelsize": 11,
    "ytick.labelsize": 11,
    "legend.fontsize": 11,
    "lines.linewidth": 2.2,
    "axes.spines.top": False,
    "axes.spines.right": False,
    "axes.grid": True,
    "grid.alpha": 0.25,
    "figure.dpi": 180,
}

STRATEGY_LABELS = {
    "random_uniform": "Random uniform",
    "small_random": "Small random",
    "near_identity": "Near-identity",
}
STRATEGY_COLORS = {
    "random_uniform": "#e15759",
    "small_random": "#4e79a7",
    "near_identity": "#59a14f",
}

ALGO_LABELS = {
    "qubit_adapt_vqe": "ADAPT-VQE",
    "hardware_efficient_vqe": "HE-VQE",
}
ALGO_COLORS = {
    "qubit_adapt_vqe": "#4e79a7",
    "hardware_efficient_vqe": "#e15759",
}

NOISE_COLORS = {
    "depolarizing": "#4e79a7",
    "phaseflip": "#e15759",
    "amplitude_damping": "#f28e2b",
}
NOISE_LABELS = {
    "depolarizing": "Depolarizing",
    "phaseflip": "Phase flip",
    "amplitude_damping": "Amplitude damping",
}

# ── Auto-discovery helpers ────────────────────────────────────────────────

def _fw_dir() -> Path:
    return Path(__file__).resolve().parent.parent


def _newest(parent: Path, glob: str = "*") -> Optional[Path]:
    """Return the lexicographically newest (= most recent timestamp) dir
    that actually has useful data (contains at least one CSV file)."""
    candidates = sorted(parent.glob(glob))
    candidates = [p for p in candidates if p.is_dir()]
    # Prefer dirs that contain a batch_all_results.csv or results_all.csv
    rich = [p for p in candidates
            if any(p.glob("*.csv")) or any(p.glob("*/*.csv"))]
    pool = rich if rich else candidates
    return pool[-1] if pool else None


def _resolve_dir(override: Optional[str], default_parent: Path, glob: str = "*") -> Path:
    if override:
        p = Path(override)
        if not p.is_absolute():
            p = _fw_dir() / p
        if not p.exists():
            raise FileNotFoundError(f"Directory not found: {p}")
        return p
    found = _newest(default_parent, glob)
    if found is None:
        raise FileNotFoundError(f"No matching directory under {default_parent}")
    return found


# ═══════════════════════════════════════════════════════════════════════════
# Figure 1 — Noise resilience (heatmap + line, depolarizing focus)
# ═══════════════════════════════════════════════════════════════════════════

def make_noise_figure(noise_dir: Path, out_dir: Path) -> Path:
    """
    Two-panel figure:
      Left  — Line plot: mean L2 parameter drift vs noise strength,
               one line per noise model, across all molecules.
      Right — Heatmap: L2 drift at p=0.02 for every molecule × noise model.
    """
    csv = noise_dir / "batch_all_results.csv"
    if not csv.exists():
        raise FileNotFoundError(f"batch_all_results.csv not found in {noise_dir}")

    df = pd.read_csv(csv)
    df = df[df["noise_model"].str.lower() != "none"].copy()
    df["noise_strength"] = pd.to_numeric(df["noise_strength"], errors="coerce")
    df["param_drift_l2"] = pd.to_numeric(df["param_drift_l2"], errors="coerce")

    models = [m for m in ["depolarizing", "phaseflip", "amplitude_damping"]
              if m in df["noise_model"].unique()]

    with plt.rc_context(POSTER_RC):
        fig, (ax_line, ax_heat) = plt.subplots(
            1, 2, figsize=(13, 5),
            gridspec_kw={"width_ratios": [1.1, 1.5]},
        )

        # ── Left: mean drift vs p ──────────────────────────────────────
        for model in models:
            sub = df[df["noise_model"] == model]
            grp = sub.groupby("noise_strength")["param_drift_l2"]
            xs = grp.mean().index.to_numpy()
            ys = grp.mean().to_numpy()
            ye = grp.sem().to_numpy()
            color = NOISE_COLORS.get(model, "grey")
            label = NOISE_LABELS.get(model, model)
            ax_line.plot(xs, ys, marker="o", color=color, label=label, linewidth=2.2)
            ax_line.fill_between(xs, ys - ye, ys + ye, alpha=0.18, color=color)

        ax_line.set_xlabel("Noise strength  p")
        ax_line.set_ylabel(r"Mean $\|\|\Delta\theta\|\|_2$  (±1 s.e.m.)")
        ax_line.set_title("Parameter drift vs noise strength")
        ax_line.legend()

        # ── Right: heatmap at p≈0.02 ──────────────────────────────────
        target_p = 0.02
        avail_p = sorted(df["noise_strength"].dropna().unique())
        closest_p = min(avail_p, key=lambda x: abs(x - target_p))
        ref = df[np.isclose(df["noise_strength"].to_numpy(float), float(closest_p), atol=1e-9)]

        mol_order = sorted(ref["molecule"].unique())
        heat_data = (
            ref.pivot_table(
                index="molecule", columns="noise_model",
                values="param_drift_l2", aggfunc="mean",
            )
            .reindex(index=mol_order, columns=models)
        )

        im = ax_heat.imshow(
            heat_data.values, aspect="auto", cmap="hot_r",
            vmin=0, vmax=heat_data.values[~np.isnan(heat_data.values)].max() * 1.05,
        )
        plt.colorbar(im, ax=ax_heat, label=r"$\|\|\Delta\theta\|\|_2$", shrink=0.82)
        ax_heat.set_xticks(range(len(models)))
        ax_heat.set_xticklabels([NOISE_LABELS.get(m, m) for m in models], fontsize=10)
        ax_heat.set_yticks(range(len(mol_order)))
        ax_heat.set_yticklabels(mol_order, fontsize=9)
        ax_heat.set_title(f"L2 drift at  p = {closest_p:g}  (all molecules)")
        ax_heat.set_xlabel("Noise channel")
        ax_heat.set_ylabel("Molecule")
        ax_heat.grid(False)
        ax_heat.spines[:].set_visible(True)

        fig.suptitle("Exp 1 — Noise resilience of VQE", fontsize=15, fontweight="bold", y=1.01)
        fig.tight_layout()
        out_path = out_dir / "poster_exp1_noise_resilience.png"
        fig.savefig(out_path, dpi=180, bbox_inches="tight")
        plt.close(fig)

    print(f"  [Exp 1] saved → {out_path}")
    return out_path


# ═══════════════════════════════════════════════════════════════════════════
# Figure 2 — Barren plateaus (gradient variance + convergence boxplot)
# ═══════════════════════════════════════════════════════════════════════════

def make_barren_plateau_figure(bp_dir: Path, out_dir: Path) -> Path:
    """
    Two-panel figure:
      Left  — Gradient variance Var[g] vs ansatz depth, per init strategy,
               averaged across all molecules.
      Right — Final-energy error-vs-CASCI boxplot per init strategy
               (all molecules pooled), showing the optimisation consequence.
    """
    grad_csv = bp_dir / "gradients_all.csv"
    conv_csv = bp_dir / "convergence_traces_all.csv"

    if not grad_csv.exists():
        raise FileNotFoundError(f"gradients_all.csv not found in {bp_dir}")

    gdf = pd.read_csv(grad_csv)
    gdf["var_grad"] = pd.to_numeric(gdf["var_grad"], errors="coerce")

    strategies = [s for s in ["random_uniform", "small_random", "near_identity"]
                  if s in gdf["strategy"].unique()]
    layers = sorted(gdf["n_layers"].unique())

    # Load final energies from the batch convergence_traces_all.csv
    # (last iteration per strategy/trial/molecule = final energy)
    opt_df = None
    traces_csv = bp_dir / "convergence_traces_all.csv"
    if traces_csv.exists():
        tdf = pd.read_csv(traces_csv)
        tdf["energy"] = pd.to_numeric(tdf["energy"], errors="coerce")
        tdf["iteration"] = pd.to_numeric(tdf["iteration"], errors="coerce")
        # Final energy = last iteration per (molecule, strategy, trial)
        final_idx = tdf.groupby(["molecule", "strategy", "trial"])["iteration"].idxmax()
        opt_df = tdf.loc[final_idx].rename(columns={"energy": "final_energy"}).copy()

    with plt.rc_context(POSTER_RC):
        fig, axes = plt.subplots(1, 2, figsize=(13, 5))
        ax_var, ax_box = axes

        # ── Left: Var[g] vs depth ──────────────────────────────────────
        for strategy in strategies:
            sub = gdf[gdf["strategy"] == strategy]
            grp = sub.groupby("n_layers")["var_grad"]
            xs = np.array([int(l) for l in layers])
            ys = np.array([grp.mean().get(l, np.nan) for l in layers])
            ye = np.array([grp.sem().get(l, np.nan) for l in layers])
            color = STRATEGY_COLORS[strategy]
            label = STRATEGY_LABELS[strategy]
            ax_var.plot(xs, ys, marker="o", color=color, label=label)
            ax_var.fill_between(xs, ys - ye, ys + ye, alpha=0.18, color=color)

        ax_var.set_yscale("log")
        ax_var.set_xlabel("Ansatz depth (# layers)")
        ax_var.set_ylabel(r"Var$[\,\partial E / \partial\theta_k\,]$  (log scale)")
        ax_var.set_title("Gradient variance vs circuit depth")
        ax_var.legend()

        # ── Right: final-energy boxplot ───────────────────────────────
        if opt_df is not None and "final_energy" in opt_df.columns and "strategy" in opt_df.columns:
            # Load per-molecule CASCI from per_molecule_summary.csv or gradient run_configs
            import json as _json
            casci_map: dict[str, float] = {}
            hf_map_bp: dict[str, float] = {}
            # gradients_all.csv doesn't have CASCI; look at per_molecule_summary
            summary_f = bp_dir / "per_molecule_summary.csv"
            if summary_f.exists():
                sdf = pd.read_csv(summary_f)
                if "casci_energy" in sdf.columns and "molecule" in sdf.columns:
                    for _, row in sdf.iterrows():
                        mol = str(row["molecule"])
                        if pd.notna(row.get("casci_energy")):
                            casci_map[mol] = float(row["casci_energy"])
                        if pd.notna(row.get("hf_energy")):
                            hf_map_bp[mol] = float(row["hf_energy"])
            if casci_map:
                opt_df["casci_energy"] = opt_df["molecule"].map(casci_map)
                opt_df["error_vs_casci"] = opt_df["final_energy"] - opt_df["casci_energy"]
            else:
                opt_df["error_vs_casci"] = np.nan

            data_by_strat = []
            labels_used = []
            for strategy in strategies:
                vals = opt_df[opt_df["strategy"] == strategy]["error_vs_casci"].dropna().values
                if len(vals):
                    data_by_strat.append(vals)
                    labels_used.append(STRATEGY_LABELS[strategy])

            if data_by_strat:
                bp_obj = ax_box.boxplot(
                    data_by_strat,
                    patch_artist=True,
                    showmeans=True,
                    meanprops=dict(marker="D", markerfacecolor="white",
                                   markeredgecolor="black", markersize=6),
                    medianprops=dict(linewidth=2),
                )
                for patch, strategy in zip(bp_obj["boxes"], strategies):
                    patch.set_facecolor(STRATEGY_COLORS[strategy])
                    patch.set_alpha(0.7)
                ax_box.set_xticks(range(1, len(labels_used) + 1))
                ax_box.set_xticklabels(labels_used, fontsize=11)
                ax_box.set_ylabel("Final energy − CASCI  (Ha)")
                ax_box.set_title("Convergence quality by init strategy")
                ax_box.axhline(0, color="black", linestyle=":", linewidth=1.2,
                               label="CASCI (exact)")
                ax_box.legend(fontsize=10)
                ax_box.set_ylim(bottom=None)
        else:
            ax_box.text(0.5, 0.5, "No optimisation data found",
                        ha="center", va="center", transform=ax_box.transAxes)

        fig.suptitle("Exp 2 — Barren plateaus & parameter initialisation",
                     fontsize=15, fontweight="bold", y=1.01)
        fig.tight_layout()
        out_path = out_dir / "poster_exp2_barren_plateau.png"
        fig.savefig(out_path, dpi=180, bbox_inches="tight")
        plt.close(fig)

    print(f"  [Exp 2] saved → {out_path}")
    return out_path


# ═══════════════════════════════════════════════════════════════════════════
# Figure 3 — Trainability: ADAPT vs HE-VQE across amino acids
# ═══════════════════════════════════════════════════════════════════════════

def make_trainability_figure(train_dir: Path, out_dir: Path) -> Path:
    """
    Two-panel figure:
      Left  — Paired dot plot: error vs CASCI for ADAPT-VQE vs HE-VQE,
               one dot per molecule, connected by grey lines.
      Right — Bar chart: cost-function evaluations for both algorithms,
               grouped by molecule.
    """
    csv = train_dir / "results_all.csv"
    if not csv.exists():
        raise FileNotFoundError(f"results_all.csv not found in {train_dir}")

    df = pd.read_csv(csv)
    df["error_vs_casci"] = pd.to_numeric(df["error_vs_casci"], errors="coerce")
    df["n_cost_evals"] = pd.to_numeric(df["n_cost_evals"], errors="coerce")

    algos = ["qubit_adapt_vqe", "hardware_efficient_vqe"]
    present_algos = [a for a in algos if a in df["algorithm"].unique()]

    # Keep only molecules that have both algorithms
    mol_counts = df.groupby("molecule")["algorithm"].nunique()
    good_mols = sorted(mol_counts[mol_counts == len(present_algos)].index.tolist())

    with plt.rc_context(POSTER_RC):
        fig, (ax_err, ax_cost) = plt.subplots(1, 2, figsize=(14, 5.5))

        # ── Left: error comparison ────────────────────────────────────
        y_pos = np.arange(len(good_mols))
        for algo in present_algos:
            sub = df[df["algorithm"] == algo].set_index("molecule")
            errs = [sub.loc[m, "error_vs_casci"] if m in sub.index else np.nan
                    for m in good_mols]
            color = ALGO_COLORS.get(algo, "grey")
            label = ALGO_LABELS.get(algo, algo)
            ax_err.scatter(errs, y_pos, color=color, label=label,
                           s=60, zorder=3)

        # Grey connector lines
        if len(present_algos) == 2:
            sub0 = df[df["algorithm"] == present_algos[0]].set_index("molecule")
            sub1 = df[df["algorithm"] == present_algos[1]].set_index("molecule")
            for i, mol in enumerate(good_mols):
                if mol in sub0.index and mol in sub1.index:
                    e0 = sub0.loc[mol, "error_vs_casci"]
                    e1 = sub1.loc[mol, "error_vs_casci"]
                    ax_err.plot([e0, e1], [i, i], color="grey",
                                linewidth=0.9, alpha=0.5, zorder=1)

        ax_err.set_yticks(y_pos)
        ax_err.set_yticklabels(good_mols, fontsize=9)
        ax_err.set_xlabel("Final energy − CASCI  (Ha)")
        ax_err.set_title("Accuracy at matched parameter budget")
        ax_err.legend()
        ax_err.set_xscale("log")

        # ── Right: cost evaluations ───────────────────────────────────
        bar_width = 0.38
        for j, algo in enumerate(present_algos):
            sub = df[df["algorithm"] == algo].set_index("molecule")
            evals = [sub.loc[m, "n_cost_evals"] if m in sub.index else 0
                     for m in good_mols]
            offset = (j - 0.5) * bar_width
            color = ALGO_COLORS.get(algo, "grey")
            label = ALGO_LABELS.get(algo, algo)
            ax_cost.barh(y_pos + offset, evals,
                         height=bar_width, color=color, alpha=0.82, label=label)

        ax_cost.set_yticks(y_pos)
        ax_cost.set_yticklabels(good_mols, fontsize=9)
        ax_cost.set_xlabel("Cost-function evaluations")
        ax_cost.set_title("Training cost (# QNode calls)")
        ax_cost.set_xscale("log")
        ax_cost.legend()

        fig.suptitle("Exp 3 — Trainability at matched parameter counts",
                     fontsize=15, fontweight="bold", y=1.01)
        fig.tight_layout()
        out_path = out_dir / "poster_exp3_trainability.png"
        fig.savefig(out_path, dpi=180, bbox_inches="tight")
        plt.close(fig)

    print(f"  [Exp 3] saved → {out_path}")
    return out_path


# ═══════════════════════════════════════════════════════════════════════════
# Figure 4 — Accuracy vs parameters (ADAPT convergence across molecules)
# ═══════════════════════════════════════════════════════════════════════════

def make_accuracy_figure(acc_dir: Path, out_dir: Path) -> Path:
    """
    Two-panel figure:
      Left  — |error vs CASCI| vs k, one thin line per molecule +
               bold mean line, log-y scale.
      Right — Same data on linear scale, normalised as fraction of
               HF→CASCI correlation energy recovered, to make all
               molecules comparable on one axis.
    """
    csv = acc_dir / "batch_all_results.csv"
    if not csv.exists():
        raise FileNotFoundError(f"batch_all_results.csv not found in {acc_dir}")

    df = pd.read_csv(csv)
    df = df[df["status"] == "ok"].copy()
    df["k"] = pd.to_numeric(df["k"], errors="coerce")
    df["error_vs_casci"] = pd.to_numeric(df["error_vs_casci"], errors="coerce")
    df["error_vs_hf"] = pd.to_numeric(df["error_vs_hf"], errors="coerce")

    # Fraction of correlation energy recovered = |E_HF - E_k| / |E_HF - E_CASCI|
    # error_vs_hf = energy - hf_energy  (negative = improvement)
    # error_vs_casci = energy - casci_energy (positive = above CASCI)
    # correlation energy gap = hf_energy - casci_energy > 0
    # fraction recovered = (hf_energy - energy) / (hf_energy - casci_energy)
    #                     = -error_vs_hf / (hf_energy - casci_energy)
    # We load hf and casci from per-molecule sub-dirs if available
    # Dir names are like "ala_qubit_adapt_vqe" — extract molecule prefix
    import json as _json2
    hf_map: dict = {}
    casci_map: dict = {}
    for mol_dir in acc_dir.iterdir():
        if not mol_dir.is_dir():
            continue
        rc_f = mol_dir / "run_config.json"
        if rc_f.exists():
            rc = _json2.loads(rc_f.read_text())
            mol_key = rc.get("molecule") or mol_dir.name.split("_")[0]
            if rc.get("hf_energy") is not None:
                hf_map[mol_key] = float(rc["hf_energy"])
            if rc.get("casci_energy") is not None:
                casci_map[mol_key] = float(rc["casci_energy"])

    if hf_map and casci_map:
        df["hf_energy"] = df["molecule"].map(hf_map)
        df["casci_energy_val"] = df["molecule"].map(casci_map)
        gap = df["hf_energy"] - df["casci_energy_val"]
        df["frac_recovered"] = (-df["error_vs_hf"]) / gap
    else:
        df["frac_recovered"] = np.nan

    mols = sorted(df["molecule"].unique())
    ks = sorted(df["k"].unique())

    CHEM_ACC_HA = 1.6e-3  # chemical accuracy threshold

    with plt.rc_context(POSTER_RC):
        fig, (ax_log, ax_frac) = plt.subplots(1, 2, figsize=(13, 5))

        # ── Left: |E - CASCI| log scale ───────────────────────────────
        mol_cmap = plt.cm.get_cmap("tab20", max(len(mols), 1))
        for i, mol in enumerate(mols):
            sub = df[df["molecule"] == mol].sort_values("k")
            ax_log.plot(sub["k"], sub["error_vs_casci"].abs(),
                        color=mol_cmap(i), alpha=0.45, linewidth=1.4,
                        marker="o", markersize=4)

        # Mean line
        mean_err = df.groupby("k")["error_vs_casci"].apply(lambda x: x.abs().mean())
        ax_log.plot(mean_err.index, mean_err.values,
                    color="black", linewidth=2.8, label="Mean (all mols)",
                    zorder=5, marker="D", markersize=6)

        # Chemical accuracy line
        ax_log.axhline(CHEM_ACC_HA, color="#f28e2b", linestyle="--",
                       linewidth=1.8, label=f"Chem. accuracy (1.6 mHa)")

        ax_log.set_yscale("log")
        ax_log.set_xlabel("k  = # ADAPT operators")
        ax_log.set_ylabel("|Energy − CASCI|  (Ha)")
        ax_log.set_title("Accuracy improves with parameter count")
        ax_log.legend(fontsize=10)
        ax_log.set_xticks(ks)

        # ── Right: fraction correlation energy recovered ───────────────
        if df["frac_recovered"].notna().any():
            for i, mol in enumerate(mols):
                sub = df[df["molecule"] == mol].sort_values("k")
                ax_frac.plot(sub["k"], sub["frac_recovered"],
                             color=mol_cmap(i), alpha=0.45, linewidth=1.4,
                             marker="o", markersize=4)

            mean_frac = df.groupby("k")["frac_recovered"].mean()
            ax_frac.plot(mean_frac.index, mean_frac.values,
                         color="black", linewidth=2.8, label="Mean (all mols)",
                         zorder=5, marker="D", markersize=6)

            ax_frac.axhline(1.0, color="grey", linestyle=":", linewidth=1.2,
                            label="CASCI (100%)")
            ax_frac.set_ylim(0, 1.05)
        else:
            # Fallback: absolute error on linear scale
            for i, mol in enumerate(mols):
                sub = df[df["molecule"] == mol].sort_values("k")
                ax_frac.plot(sub["k"], sub["error_vs_casci"].abs(),
                             color=mol_cmap(i), alpha=0.45, linewidth=1.4,
                             marker="o", markersize=4)
            mean_err2 = df.groupby("k")["error_vs_casci"].apply(lambda x: x.abs().mean())
            ax_frac.plot(mean_err2.index, mean_err2.values,
                         color="black", linewidth=2.8, label="Mean", zorder=5)

        ax_frac.set_xlabel("k  = # ADAPT operators")
        ax_frac.set_ylabel("Fraction of correlation energy recovered")
        ax_frac.set_title("Correlation energy captured per operator")
        ax_frac.legend(fontsize=10)
        ax_frac.set_xticks(ks)

        # Add molecule count annotation
        n_ok = df["molecule"].nunique()
        fig.text(0.5, -0.02,
                 f"Qubit-ADAPT-VQE on {n_ok} amino acids (12-qubit active space, STO-3G)",
                 ha="center", fontsize=11, color="dimgrey")

        fig.suptitle("Exp 4 — Accuracy vs number of variational parameters",
                     fontsize=15, fontweight="bold", y=1.01)
        fig.tight_layout()
        out_path = out_dir / "poster_exp4_accuracy_vs_params.png"
        fig.savefig(out_path, dpi=180, bbox_inches="tight")
        plt.close(fig)

    print(f"  [Exp 4] saved → {out_path}")
    return out_path


# ═══════════════════════════════════════════════════════════════════════════
# Optional: one combined 2×2 panel for the poster
# ═══════════════════════════════════════════════════════════════════════════

def make_combined_figure(panel_paths: list[Path], out_dir: Path) -> Path:
    """Tile the four individual panels into one 2×2 figure."""
    from PIL import Image  # type: ignore

    imgs = [Image.open(p) for p in panel_paths]
    widths = [img.width for img in imgs]
    heights = [img.height for img in imgs]
    W = max(widths[0], widths[1], widths[2], widths[3])
    H = max(heights[0], heights[2])

    # Rescale each image to fit equal cell size
    cell_w = max(widths[:2] + widths[2:]) // 2 * 2  # even number
    cell_h = max(heights)
    cell_w = max(widths)
    cell_h = max(heights)

    resized = []
    for img in imgs:
        scale = min(cell_w / img.width, cell_h / img.height)
        nw, nh = int(img.width * scale), int(img.height * scale)
        resized.append(img.resize((nw, nh), Image.LANCZOS))

    canvas_w = cell_w * 2
    canvas_h = cell_h * 2
    canvas = Image.new("RGB", (canvas_w, canvas_h), "white")

    positions = [(0, 0), (cell_w, 0), (0, cell_h), (cell_w, cell_h)]
    for ri, (pos, img) in enumerate(zip(positions, resized)):
        canvas.paste(img, pos)

    out_path = out_dir / "poster_all_experiments.png"
    canvas.save(out_path, dpi=(180, 180))
    print(f"  [Combined] saved → {out_path}")
    return out_path


# ═══════════════════════════════════════════════════════════════════════════
# CLI
# ═══════════════════════════════════════════════════════════════════════════

def _build_parser() -> argparse.ArgumentParser:
    fw = _fw_dir()
    p = argparse.ArgumentParser(
        description="Generate four poster-quality figures from experiment results.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument(
        "--noise-dir", default=None,
        help="Path to noise-resilience batch results folder "
             "(auto: newest under experiments/results/noise_resilience_batch/)",
    )
    p.add_argument(
        "--bp-dir", default=None,
        help="Path to barren-plateau batch results folder "
             "(auto: newest under experiments/results/barren_plateau_batch/)",
    )
    p.add_argument(
        "--train-dir", default=None,
        help="Path to trainability batch comparison folder "
             "(auto: newest under experiments/results/trainability_batch/ matching *comparison)",
    )
    p.add_argument(
        "--acc-dir", default=None,
        help="Path to accuracy-vs-params batch folder "
             "(auto: newest under experiments/results/accuracy_vs_params_batch/)",
    )
    p.add_argument(
        "--out-dir", default=str(fw / "plots" / "poster_figures"),
        help="Output directory for poster PNG files.",
    )
    p.add_argument(
        "--skip-combined", action="store_true",
        help="Skip generating the combined 2×2 panel (requires Pillow).",
    )
    p.add_argument(
        "--only", choices=["noise", "bp", "train", "acc"], default=None,
        help="Generate only one of the four figures.",
    )
    return p


def main() -> None:
    args = _build_parser().parse_args()
    fw = _fw_dir()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    results_root = fw / "experiments" / "results"

    # Resolve data directories
    noise_dir = _resolve_dir(
        args.noise_dir, results_root / "noise_resilience_batch",
    )
    bp_dir = _resolve_dir(
        args.bp_dir, results_root / "barren_plateau_batch", "*batch_hardware_efficient_vqe",
    )
    train_dir = _resolve_dir(
        args.train_dir, results_root / "trainability_batch",
        "*comparison",
    )
    # If the auto-discovered dir lacks results_all.csv, fall back to previous
    if not (train_dir / "results_all.csv").exists():
        candidates = sorted(
            (results_root / "trainability_batch").glob("*comparison")
        )
        candidates = [p for p in candidates if (p / "results_all.csv").exists()]
        if candidates:
            train_dir = candidates[-1]
    acc_dir = _resolve_dir(
        args.acc_dir, results_root / "accuracy_vs_params_batch",
    )

    print("Data sources:")
    print(f"  Noise:      {noise_dir}")
    print(f"  BP:         {bp_dir}")
    print(f"  Train:      {train_dir}")
    print(f"  Accuracy:   {acc_dir}")
    print(f"  Output dir: {out_dir}\n")

    panel_paths: list[Path] = []

    def _run(label: str, fn, *args_inner):
        try:
            path = fn(*args_inner)
            panel_paths.append(path)
        except Exception as exc:
            print(f"  [{label}] FAILED: {exc}")

    only = args.only
    if only in (None, "noise"):
        _run("Exp 1", make_noise_figure, noise_dir, out_dir)
    if only in (None, "bp"):
        _run("Exp 2", make_barren_plateau_figure, bp_dir, out_dir)
    if only in (None, "train"):
        _run("Exp 3", make_trainability_figure, train_dir, out_dir)
    if only in (None, "acc"):
        _run("Exp 4", make_accuracy_figure, acc_dir, out_dir)

    if not args.skip_combined and len(panel_paths) == 4:
        try:
            make_combined_figure(panel_paths, out_dir)
        except ImportError:
            print("  [Combined] skipped (Pillow not installed; pip install Pillow)")
        except Exception as exc:
            print(f"  [Combined] FAILED: {exc}")

    print(f"\nDone. Figures written to {out_dir}")


if __name__ == "__main__":
    main()
