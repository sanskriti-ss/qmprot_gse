"""Plot saved experiment results with uncertainty over identified independent seeds.

A 95% Student-t interval is shown when at least two independent seeds contribute
at a point. The supplied examples contain one run, so no interval is estimated.
"""
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.ticker import LogLocator, NullLocator
import numpy as np
import pandas as pd
from scipy.stats import t
from figure_style import ALGORITHMS, PURPLE, finish, grid, label, style, title

CHANNELS = {"amplitude_damping": ("#d4779c", "Amplitude damping"),
            "depolarizing": ("#f97316", "Depolarizing"),
            "phaseflip": ("#7c3aed", "Phase flip")}


def summarize(df, keys, value, *, positive=False):
    """Per-condition t intervals, with seed identifiers required for replicates.

    On a logarithmic runtime axis, estimate the geometric mean and its interval
    in log space. Other axes use the arithmetic mean in the original units.
    """
    data = df.copy()
    data[value] = pd.to_numeric(data[value], errors="coerce")
    data = data[np.isfinite(data[value])].copy()
    if positive:
        data = data[data[value] > 0].copy()
    seed = next((c for c in ("seed", "trial_seed", "random_seed") if c in data.columns), None)
    duplicated = data.duplicated(keys, keep=False)
    if duplicated.any():
        if seed is None:
            raise ValueError("Repeated observations need a seed, trial_seed, or random_seed column")
        if data[seed].isna().any() or data.duplicated(keys + [seed]).any():
            raise ValueError("Each observation needs a distinct, nonempty seed within its condition")
    rows = []
    for group, part in data.groupby(keys, dropna=False, sort=True):
        if not isinstance(group, tuple): group = (group,)
        values = part[value].to_numpy(dtype=float)
        transformed = np.log(values) if positive else values
        n = len(values)
        mean = float(np.mean(transformed))
        half = float(t.ppf(.975, n - 1) * np.std(transformed, ddof=1) / np.sqrt(n)) if n >= 2 else 0.
        if positive:
            estimate, low, high = np.exp([mean, mean - half, mean + half])
        else:
            estimate, low, high = mean, mean - half, mean + half
        row = dict(zip(keys, group))
        row.update(estimate=float(estimate), low=float(low), high=float(high), n=n)
        rows.append(row)
    return pd.DataFrame(rows, columns=keys + ["estimate", "low", "high", "n"])


def ci_caption(*tables):
    counts = [int(n) for table in tables for n in table.get("n", [])]
    if not counts or max(counts) == 1:
        return "One run per condition; uncertainty unavailable"
    if min(counts) < 2:
        return "95% t intervals where n ≥ 2 seeds; other points have n = 1"
    return "95% t intervals over independent seeds"


def errorbar(ax, x, y, low, high, *, horizontal=False, **kwargs):
    err = np.array([[y - low], [high - y]])
    if horizontal:
        ax.errorbar(y, x, xerr=err, **kwargs)
    else:
        ax.errorbar(x, y, yerr=err, **kwargs)


def pick_target(df, requested):
    df = df.copy()
    df["target_n_params"] = pd.to_numeric(df.target_n_params, errors="raise")
    targets = sorted(df.target_n_params.dropna().unique())
    if requested is None:
        if len(targets) != 1:
            raise ValueError(f"Select one parameter target with --target; available: {targets}")
        requested = targets[0]
    df = df[df.target_n_params == requested].copy()
    if df.empty: raise ValueError(f"No rows for target {requested}")
    return df, requested


def trainability_data(args):
    df = pd.read_csv(args.input)
    required = {"molecule", "algorithm", "target_n_params", "final_energy", "hf_energy", "runtime_seconds"}
    missing = required - set(df.columns)
    if missing: raise ValueError(f"Missing columns: {sorted(missing)}")
    df = df[df.algorithm.isin(ALGORITHMS)].copy()
    df, target = pick_target(df, args.target)
    for col in ("final_energy", "hf_energy", "runtime_seconds"):
        df[col] = pd.to_numeric(df[col], errors="coerce")
    df["energy_minus_hf"] = df.final_energy - df.hf_energy
    return df, target


def _trainability_summaries(df):
    keys = ["molecule", "algorithm"]
    return summarize(df, keys, "energy_minus_hf"), summarize(df, keys, "runtime_seconds", positive=True)


def draw_forest(ax, table, molecules, *, runtime=False):
    for index, (algo, (color, _)) in enumerate(ALGORITHMS.items()):
        part = table[table.algorithm == algo].set_index("molecule")
        for y, molecule in enumerate(molecules):
            if molecule not in part.index: continue
            row = part.loc[molecule]
            errorbar(ax, y + (-.14 if index == 0 else .14), row.estimate, row.low, row.high,
                     horizontal=True, fmt="o" if index == 0 else "s", color=color, markersize=4.5,
                     capsize=3 if row.n >= 2 else 0, elinewidth=1.2, linestyle="none")
    ax.set_yticks(range(len(molecules)), [str(m).upper() for m in molecules])
    ax.invert_yaxis(); ax.grid(axis="x", alpha=.16)
    if runtime:
        ax.set_xscale("log"); ax.set_xlabel("Geometric mean runtime (seconds; log scale)")
        ax.xaxis.set_major_locator(LogLocator(base=10, subs=(1, 2, 5), numticks=6))
        ax.xaxis.set_minor_locator(NullLocator())
    else:
        ax.axvline(0, color="#9b8ba7", lw=1, linestyle=":")
        ax.set_xlabel("Final energy − molecular HF (Ha)")


def legend(fig):
    fig.legend([Line2D([0], [0], color=v[0], marker="o" if i == 0 else "s", lw=0)
                for i, v in enumerate(ALGORITHMS.values())],
               [v[1] for v in ALGORITHMS.values()], loc="lower center", ncol=2, frameon=False)


def noise_energy(args, default_input):
    style(); df = pd.read_csv(args.input)
    required = {"noise_model", "noise_strength", "energy_noisy", "energy_clean_at_noisy_params", "energy_clean_baseline"}
    missing = required - set(df.columns)
    if missing: raise ValueError(f"Missing columns: {sorted(missing)}")
    if "molecule" in df:
        if args.molecule is None: raise ValueError("Batch input requires --molecule")
        df = df[df.molecule.astype(str).str.lower() == args.molecule.lower()].copy()
    df = df[df.noise_model.isin(CHANNELS)].copy()
    if df.empty: raise ValueError("No requested noise-channel rows")
    for col in ("noise_strength", "energy_noisy", "energy_clean_at_noisy_params", "energy_clean_baseline"):
        df[col] = pd.to_numeric(df[col], errors="raise")
    seed = next((c for c in ("seed", "trial_seed", "random_seed") if c in df), None)
    groups = df.groupby(seed, dropna=False) if seed else [(None, df)]
    for _, group in groups:
        base = group.energy_clean_baseline.dropna().to_numpy()
        if not len(base) or not np.allclose(base, base[0], rtol=0, atol=1e-8):
            raise ValueError("Expected one clean fitted baseline per independent seed")
    df["noisy_delta"] = df.energy_noisy - df.energy_clean_baseline
    df["clean_delta"] = df.energy_clean_at_noisy_params - df.energy_clean_baseline
    keys = ["noise_model", "noise_strength"]
    noisy = summarize(df, keys, "noisy_delta"); clean = summarize(df, keys, "clean_delta")
    fig, axes = plt.subplots(1, 3, figsize=(12, 4.8), sharey=True)
    for ax, (channel, (color, name)) in zip(axes, CHANNELS.items()):
        ax.axhline(0, color="#9b8ba7", lw=1, linestyle=":")
        for table, marker, linestyle in ((noisy, "o", "-"), (clean, "s", "--")):
            part = table[table.noise_model == channel].sort_values("noise_strength")
            ax.plot(part.noise_strength, part.estimate, color=color, lw=1.6, ls=linestyle)
            for row in part.itertuples():
                errorbar(ax, row.noise_strength, row.estimate, row.low, row.high, fmt=marker,
                         color=color, markersize=4.5, capsize=3 if row.n >= 2 else 0, linestyle="none")
        ax.set_title(name, fontweight="bold"); ax.set_xlabel("Noise strength p"); ax.grid(alpha=.16)
    axes[0].set_ylabel("Energy − clean fitted baseline (Ha)")
    fig.legend([Line2D([0], [0], color=PURPLE, marker="o", lw=1.8),
                Line2D([0], [0], color=PURPLE, marker="s", lw=1.6, ls="--")],
               ["Noisy objective", "Clean at noisy optimum"], loc="lower center", ncol=2, frameon=False)
    molecule = f" | {args.molecule.upper()}" if args.molecule else ""
    title(fig, "Noise effects on optimized energies", label(args, default_input) + molecule + " | " + ci_caption(noisy, clean))
    fig.tight_layout(rect=(0, .09, 1, .91), w_pad=1.4); finish(fig, args.output)


def trainability_metric(args, default_input, metric):
    style(); df, target = trainability_data(args)
    energy, runtime = _trainability_summaries(df)
    table = energy if metric == "energy" else runtime
    if table.empty: raise ValueError("No finite values for this view")
    molecules = sorted(df.molecule.astype(str).unique())
    fig, ax = plt.subplots(figsize=(8, max(5, len(molecules) * .28 + 2.2)))
    draw_forest(ax, table, molecules, runtime=metric == "runtime")
    heading = "Trainability: energy relative to HF" if metric == "energy" else "Trainability: runtime by molecule"
    title(fig, heading, f"{label(args, default_input)} | target {target:g} | {ci_caption(table)}")
    legend(fig); fig.tight_layout(rect=(0, .08, 1, .92)); finish(fig, args.output)


def adapt_data(args):
    df = pd.read_csv(args.input)
    required = {"molecule", "k", "cumulative_cost_evals"}
    missing = required - set(df.columns)
    if missing: raise ValueError(f"Missing columns: {sorted(missing)}")
    failed = set()
    if "status" in df:
        failed = set(df.loc[df.status.astype(str).str.lower() == "failed", "molecule"].astype(str))
        df = df[df.status.astype(str).str.lower() != "failed"].copy()
    df["operators"] = pd.to_numeric(df["attained_k"] if "attained_k" in df else df["k"], errors="coerce")
    df["cost"] = pd.to_numeric(df.cumulative_cost_evals, errors="coerce")
    df = df.dropna(subset=["operators", "cost"])
    # Repeated requested targets can point to the same attained early-stop state.
    seed = next((c for c in ("seed", "trial_seed", "random_seed") if c in df), None)
    keys = ["molecule", "operators"] + ([seed] if seed else [])
    for column in ["cost"] + (["error_vs_hf"] if "error_vs_hf" in df else []):
        if (df.groupby(keys, dropna=False)[column].nunique(dropna=False) > 1).any():
            raise ValueError("Conflicting values for the same molecule, seed and attained operator count")
    df = df.drop_duplicates(keys)
    if df.empty: raise ValueError("No completed ADAPT data")
    return df, failed


def adapt_cost(args, default_input):
    style(); df, failed = adapt_data(args)
    table = summarize(df, ["molecule", "operators"], "cost")
    molecules = sorted(df.molecule.unique()); fig, axes = grid(len(molecules))
    for ax, mol in zip(axes, molecules):
        part = table[table.molecule == mol].sort_values("operators")
        ax.plot(part.operators, part.estimate, color="#7c3aed", lw=1.6)
        for row in part.itertuples():
            errorbar(ax, row.operators, row.estimate, row.low, row.high, fmt="o", color="#7c3aed",
                     markersize=4, capsize=3 if row.n >= 2 else 0, linestyle="none")
        ax.set_xticks(sorted(part.operators.unique()))
        ax.set_title(str(mol).upper(), loc="left", fontsize=10, fontweight="bold")
        ax.set_xlabel("Attained operators"); ax.set_ylabel("Cumulative cost evaluations"); ax.grid(alpha=.16)
    for ax in axes[len(molecules):]: ax.remove()
    suffix = f" | {len(failed)} failed molecule(s) omitted" if failed else ""
    title(fig, "ADAPT cost as operators are added", label(args, default_input) + suffix + " | " + ci_caption(table))
    fig.tight_layout(rect=(0, 0, 1, .955), h_pad=1.6, w_pad=1.4); finish(fig, args.output)


def combined_trainability(args, default_input):
    style(); df, target = trainability_data(args)
    energy, runtime = _trainability_summaries(df)
    molecules = sorted(df.molecule.astype(str).unique())
    fig, axes = plt.subplots(1, 2, figsize=(10.8, max(5.3, .27 * len(molecules) + 2.6)), sharey=True)
    draw_forest(axes[0], energy, molecules)
    draw_forest(axes[1], runtime, molecules, runtime=True)
    # Shared y-axis should be inverted only once.
    axes[0].set_ylim(len(molecules) - .5, -.5)
    for ax, panel, heading in zip(axes, ("(a)", "(b)"), ("Final energy", "Runtime")):
        ax.set_title(heading, fontsize=11, fontweight="bold")
        ax.text(-.08, 1.06, panel, transform=ax.transAxes, fontweight="bold", fontsize=12, color=PURPLE)
    legend(fig)
    title(fig, "Trainability: energy and runtime", f"{label(args, default_input)} | target {target:g} | {ci_caption(energy, runtime)}")
    fig.tight_layout(rect=(0, .07, 1, .92), w_pad=2); finish(fig, args.output)


def combined_adapt(args, default_input):
    style(); df, _ = adapt_data(args)
    if "error_vs_hf" not in df: raise ValueError("Missing error_vs_hf column")
    df = df[df.molecule.astype(str).str.lower() == args.molecule.lower()].copy()
    if df.empty: raise ValueError(f"No completed rows for {args.molecule}")
    energy = summarize(df, ["operators"], "error_vs_hf")
    cost = summarize(df, ["operators"], "cost")
    fig, axes = plt.subplots(1, 2, figsize=(10.5, 4.7))
    axes[0].axhline(0, color="#9b8ba7", lw=1, linestyle=":")
    for ax, table, color in ((axes[0], energy, "#d4779c"), (axes[1], cost, "#7c3aed")):
        ax.plot(table.operators, table.estimate, color=color, lw=1.8)
        for row in table.itertuples():
            errorbar(ax, row.operators, row.estimate, row.low, row.high, fmt="o", color=color,
                     markersize=5, capsize=3 if row.n >= 2 else 0, linestyle="none")
    axes[0].set_ylabel("Energy − molecular HF (Ha)"); axes[1].set_ylabel("Cumulative cost evaluations")
    for ax, panel, heading in zip(axes, ("(a)", "(b)"), ("Final energy", "Evaluation cost")):
        ax.set_title(heading, fontsize=11, fontweight="bold")
        ax.text(-.10, 1.06, panel, transform=ax.transAxes, fontweight="bold", fontsize=12, color=PURPLE)
        ax.set_xlabel("Attained operators"); ax.set_xticks(sorted(df.operators.unique())); ax.grid(alpha=.16)
    title(fig, "ADAPT energy and evaluation cost", f"{label(args, default_input)} | {args.molecule.upper()} | {ci_caption(energy, cost)}")
    fig.tight_layout(rect=(0, 0, 1, .90), w_pad=2); finish(fig, args.output)
