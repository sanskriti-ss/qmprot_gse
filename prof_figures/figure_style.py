"""Shared plotting utilities for appendix figures. No VQE calculations run here."""
from pathlib import Path
import argparse
import json
import math

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap
from matplotlib.lines import Line2D
import numpy as np
import pandas as pd

PURPLE = "#3b0764"
STRATEGIES = {"random_uniform": ("#f97316", "Random uniform"),
              "small_random": ("#d4779c", "Small random"),
              "near_identity": ("#7c3aed", "Near-identity"),
              "zeros": ("#64748b", "Zeros")}
ALGORITHMS = {"qubit_adapt_vqe": ("#7c3aed", "Qubit-ADAPT"),
              "hardware_efficient_vqe": ("#f97316", "Hardware-efficient")}
SUNSET = LinearSegmentedColormap.from_list("sunset", ["#1e1b4b", "#4c1d95", "#7e57a0", "#a678b8", "#d9977e", "#f0d48a"])


def style():
    plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 9, "text.color": PURPLE,
                         "axes.labelcolor": PURPLE, "axes.titlecolor": PURPLE,
                         "xtick.color": PURPLE, "ytick.color": PURPLE, "axes.edgecolor": "#d3cbd9",
                         "figure.facecolor": "white", "axes.facecolor": "white",
                         "savefig.facecolor": "white", "axes.spines.top": False, "axes.spines.right": False})


def parser(default_input, description, default_label):
    p = argparse.ArgumentParser(description=description)
    p.add_argument("--input", type=Path, default=default_input, help="Result CSV or directory")
    p.add_argument("--output", type=Path, default=Path(__file__).resolve().parent / "figure.png")
    p.add_argument("--label", default=None, help="Short provenance label shown under the title")
    p.set_defaults(sample_label=default_label)
    return p


def label(args, default_input):
    if args.label:
        return args.label
    return args.sample_label if args.input.resolve() == default_input.resolve() else "User-supplied results"


def finish(fig, output):
    output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output, dpi=220, bbox_inches="tight")
    fig.savefig(output.with_suffix(".pdf"), bbox_inches="tight")
    plt.close(fig)
    print(f"Saved {output} and {output.with_suffix('.pdf')}")


def load_table(source, file_name, aggregate=None):
    source = Path(source)
    if source.is_file():
        return pd.read_csv(source)
    if aggregate and (source / aggregate).is_file():
        return pd.read_csv(source / aggregate)
    files = sorted(source.glob(f"*/{file_name}"))
    if (source / file_name).is_file():
        files.insert(0, source / file_name)
    if not files:
        raise FileNotFoundError(f"No {file_name} found in {source}")
    frames = []
    for file in files:
        frame = pd.read_csv(file)
        if "molecule" not in frame:
            config = file.parent / "run_config.json"
            if not config.is_file():
                raise ValueError(f"Missing molecule and run_config.json: {file}")
            frame["molecule"] = json.loads(config.read_text())["molecule"]
        frames.append(frame)
    return pd.concat(frames, ignore_index=True)


def grid(n):
    cols = min(4, max(1, math.ceil(math.sqrt(n))))
    rows = math.ceil(n / cols)
    fig, axes = plt.subplots(rows, cols, figsize=(max(7, 3.45 * cols), max(4.6, 2.55 * rows)), squeeze=False)
    return fig, axes.ravel()


def title(fig, heading, provenance):
    fig.suptitle(heading, color=PURPLE, fontweight="bold", fontsize=16, y=1.02)
    fig.text(.5, .965, provenance, ha="center", va="top", color="#6b5a77", fontsize=9)


def traces(args, default_input, heading):
    style()
    df = load_table(args.input, "convergence_traces.csv", "convergence_traces_all.csv")
    needed = {"molecule", "strategy", "trial", "iteration", "energy"}
    if not needed.issubset(df.columns):
        raise ValueError(f"Missing columns: {sorted(needed - set(df.columns))}")
    if df.duplicated(["molecule", "strategy", "trial", "iteration"]).any():
        raise ValueError("Duplicate trial steps; select one run and circuit depth per molecule")
    molecules = sorted(df.molecule.unique())
    fig, axes = grid(len(molecules))
    for ax, mol in zip(axes, molecules):
        part = df[df.molecule == mol]
        for strategy, (color, _) in STRATEGIES.items():
            group = part[part.strategy == strategy]
            if group.empty: continue
            for _, trial in group.groupby("trial", sort=True):
                trial = trial.sort_values("iteration")
                ax.plot(trial.iteration, trial.energy, color=color, linewidth=.6, alpha=.22)
            counts = group.groupby("iteration").energy.agg(["mean", "count"])
            counts = counts[counts["count"] == group.trial.nunique()]
            ax.plot(counts.index, counts["mean"], color=color, linewidth=1.8)
        ax.set_title(str(mol).upper(), fontsize=10, fontweight="bold", loc="left")
        ax.grid(alpha=.16); ax.tick_params(labelsize=8)
        ax.set_xlabel("Optimizer step"); ax.set_ylabel("Energy (Ha)")
    for ax in axes[len(molecules):]: ax.remove()
    used = [k for k in STRATEGIES if k in set(df.strategy)]
    fig.legend([Line2D([0], [0], color=STRATEGIES[k][0], lw=2) for k in used],
               [STRATEGIES[k][1] for k in used], loc="lower center", ncol=max(1, len(used)), frameon=False)
    title(fig, heading, label(args, default_input) + " | Thin lines: trials; bold: mean over shared steps")
    fig.tight_layout(rect=(0, .05, 1, .955), h_pad=1.5, w_pad=1.4)
    finish(fig, args.output)


def finals(args, default_input):
    style(); df = load_table(args.input, "optimization_results.csv")
    needed = {"molecule", "strategy", "trial", "final_energy"}
    if not needed.issubset(df.columns): raise ValueError(f"Missing columns: {sorted(needed - set(df.columns))}")
    if df.duplicated(["molecule", "strategy", "trial"]).any(): raise ValueError("Duplicate trials in input")
    molecules = sorted(df.molecule.unique()); fig, axes = grid(len(molecules))
    for ax, mol in zip(axes, molecules):
        part = df[df.molecule == mol]; present = [k for k in STRATEGIES if k in set(part.strategy)]
        for i, strategy in enumerate(present):
            values = pd.to_numeric(part.loc[part.strategy == strategy, "final_energy"], errors="raise").dropna().to_numpy()
            color = STRATEGIES[strategy][0]
            if len(values) >= 2:
                box = ax.boxplot([values], positions=[i], widths=.52, patch_artist=True, showfliers=False)
                box["boxes"][0].set(facecolor=color, alpha=.45, edgecolor=color)
                for line in box["medians"]: line.set(color=PURPLE, linewidth=1.5)
            if len(values): ax.scatter(np.full(len(values), i), values, color=color, s=20, zorder=3, alpha=.8)
        ax.set_xticks(range(len(present)), [STRATEGIES[k][1].replace(" ", "\n") for k in present], fontsize=8)
        ax.set_title(str(mol).upper(), fontsize=10, fontweight="bold", loc="left")
        ax.set_ylabel("Final energy (Ha)"); ax.grid(axis="y", alpha=.16)
    for ax in axes[len(molecules):]: ax.remove()
    title(fig, "Final energies by initialization", label(args, default_input) + " | Points: individual trials")
    fig.tight_layout(rect=(0, 0, 1, .94), h_pad=1.8)
    finish(fig, args.output)


def trainability(args, default_input):
    from scipy.stats import t

    style(); df = pd.read_csv(args.input)
    needed = {"molecule", "algorithm", "target_n_params", "n_cost_evals"}
    if not needed.issubset(df.columns): raise ValueError(f"Missing columns: {sorted(needed - set(df.columns))}")
    df = df[df.algorithm.isin(ALGORITHMS)].copy()
    if df.empty: raise ValueError("No ADAPT or hardware-efficient rows")
    df["target_n_params"] = pd.to_numeric(df.target_n_params, errors="raise")
    df["n_cost_evals"] = pd.to_numeric(df.n_cost_evals, errors="coerce")
    excluded = int((~np.isfinite(df.n_cost_evals) | (df.n_cost_evals <= 0)).sum())
    df = df[np.isfinite(df.n_cost_evals) & (df.n_cost_evals > 0)].copy()
    if df.empty: raise ValueError("No positive evaluation counts to display on a log axis")
    keys = ["molecule", "algorithm", "target_n_params"]
    seed = next((name for name in ("seed", "trial_seed", "random_seed") if name in df), None)
    if df.duplicated(keys).any():
        if seed is None or df[seed].isna().any() or df.duplicated(keys + [seed]).any():
            raise ValueError("Repeated runs require a distinct seed, trial_seed or random_seed per condition")

    has_counts = "actual_n_params" in df.columns
    if has_counts:
        df["actual_n_params"] = pd.to_numeric(df.actual_n_params, errors="coerce")
    molecules = sorted(df.molecule.astype(str).unique())
    targets = sorted(df.target_n_params.unique())
    fig, axes = plt.subplots(1, len(targets), sharey=True, squeeze=False,
                             figsize=(max(7, 4.7 * len(targets)), max(5.5, .28 * len(molecules) + 2.5)))
    counts = []
    any_mismatch = False
    any_unknown = False
    for ax, target in zip(axes.ravel(), targets):
        for index, (algorithm, (color, _)) in enumerate(ALGORITHMS.items()):
            for y, molecule in enumerate(molecules):
                part = df[(df.molecule == molecule) & (df.algorithm == algorithm) & (df.target_n_params == target)]
                if part.empty: continue
                values = np.log(part.n_cost_evals.to_numpy(dtype=float))
                n = len(values); counts.append(n)
                mean = float(np.mean(values))
                half = float(t.ppf(.975, n - 1) * np.std(values, ddof=1) / np.sqrt(n)) if n >= 2 else 0.
                point, low, high = np.exp([mean, mean - half, mean + half])
                actual = part.actual_n_params if has_counts else pd.Series(dtype=float)
                known = len(actual) == n and actual.notna().all()
                mismatch = bool(known and (actual != target).any())
                any_mismatch |= mismatch
                any_unknown |= not known
                ax.errorbar(point, y + (-.14 if index == 0 else .14),
                            xerr=np.array([[point - low], [high - point]]),
                            fmt="o" if index == 0 else "s", color=color,
                            markerfacecolor="none" if mismatch else color,
                            markeredgewidth=1.4, markersize=5, capsize=3 if n >= 2 else 0,
                            elinewidth=1.2, linestyle="none")
        ax.set_xscale("log"); ax.grid(axis="x", which="major", alpha=.17)
        ax.set_title(f"Requested target: {target:g}", fontsize=10, fontweight="bold")
        ax.set_xlabel("Cost-function evaluations (log scale)")
    axes[0, 0].set_yticks(range(len(molecules)), [m.upper() for m in molecules])
    axes[0, 0].set_ylim(len(molecules) - .5, -.5)
    handles = [Line2D([0], [0], color=color, marker="o" if i == 0 else "s", lw=0,
                      label=name) for i, (color, name) in enumerate(ALGORITHMS.values())]
    if any_mismatch:
        handles.append(Line2D([0], [0], marker="o", color=PURPLE, markerfacecolor="none",
                              lw=0, label="Attained count differs from target"))
    fig.legend(handles=handles, loc="lower center", ncol=min(3, len(handles)), frameon=False)
    uncertainty = ("One run per condition; uncertainty unavailable" if max(counts) == 1
                   else "95% t intervals across seeds where n ≥ 2; geometric means")
    notes = [label(args, default_input), uncertainty]
    if any_unknown: notes.append("Some attained counts unavailable")
    if excluded: notes.append(f"{excluded} zero or unavailable counts omitted")
    title(fig, "Evaluation cost by molecule", " | ".join(notes))
    fig.tight_layout(rect=(0, .08, 1, .92), w_pad=1.8)
    finish(fig, args.output)


def noise(args, default_input):
    style(); df = pd.read_csv(args.input)
    needed = {"molecule", "noise_model", "noise_strength", "param_drift_cosine"}
    if not needed.issubset(df.columns): raise ValueError(f"Missing columns: {sorted(needed - set(df.columns))}")
    df = df[df.noise_model.isin(["amplitude_damping", "depolarizing", "phaseflip"])].copy()
    if df.duplicated(["molecule", "noise_model", "noise_strength"]).any():
        raise ValueError("Multiple rows for a molecule, channel and strength; select or aggregate trials explicitly")
    molecules = sorted(df.molecule.unique()); strengths = sorted(df.noise_strength.unique())
    if not molecules or not strengths: raise ValueError("No noise-channel data")
    fig, axes = plt.subplots(1, 3, figsize=(12.2, max(3.7, .27 * len(molecules) + 2.0)), sharey=True)
    image = None
    for ax, model, name in zip(axes, ["amplitude_damping", "depolarizing", "phaseflip"], ["Amplitude damping", "Depolarizing", "Phase flip"]):
        matrix = df[df.noise_model == model].pivot(index="molecule", columns="noise_strength", values="param_drift_cosine").reindex(index=molecules, columns=strengths)
        image = ax.imshow(matrix, aspect="auto", cmap=SUNSET, vmin=-1, vmax=1, interpolation="nearest")
        ax.set_title(name, fontweight="bold", fontsize=10); ax.set_xticks(range(len(strengths)), [f"{v:g}" for v in strengths], rotation=45, ha="right", fontsize=8)
        ax.set_yticks(range(len(molecules)), [str(x).upper() for x in molecules], fontsize=8); ax.set_xlabel("Noise strength p")
        ax.tick_params(length=0)
    title(fig, "Parameter cosine similarity under noise", label(args, default_input) + " | Blank cells: unavailable")
    fig.subplots_adjust(left=.09, right=.86, bottom=.16, top=.86, wspace=.08)
    cax = fig.add_axes([.89, .20, .015, .60])
    fig.colorbar(image, cax=cax, label="Cosine similarity")
    finish(fig, args.output)
