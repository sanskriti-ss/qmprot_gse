"""Plot saved prefix summaries without importing or running experiment solvers."""
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from figure_style import parser, style, title, finish
import matplotlib.pyplot as plt
from matplotlib.ticker import MaxNLocator, FixedLocator, NullLocator
import numpy as np
import pandas as pd

p = parser(None, __doc__, "Fresh rerun")
p.add_argument("--exclude-qaoa", action="store_true")
args = p.parse_args()
if args.input is None:
    p.error("--input is required")
df = pd.read_csv(args.input)
if args.exclude_qaoa:
    df = df[df.algorithm != "qaoa_inspired_vqe"]
if df.empty or df.duplicated(["algorithm", "n_parameters"]).any():
    raise ValueError("Expected one nonempty summary row per algorithm and term count")
style()
fig, axes = plt.subplots(1, 2, figsize=(14, 6))
colors = plt.get_cmap("tab20")
for index, (algorithm, group) in enumerate(df.groupby("algorithm", sort=True)):
    group = group.sort_values("n_parameters")
    for ax, reference in zip(axes, ("full", "prefix")):
        mean = group[f"mean_abs_error_vs_{reference}_reference"].to_numpy(float)
        sd = group[f"std_abs_error_vs_{reference}_reference"].to_numpy(float)
        xs = group.n_parameters.to_numpy(int)
        ax.plot(xs, mean, marker="o", markersize=3, color=colors(index), label=algorithm)
        ax.fill_between(xs, np.maximum(0, mean - sd), mean + sd, alpha=.12, color=colors(index))
for ax, heading, ylabel in zip(axes,
        ("Prefix energy versus full CASCI reference", "Optimization error on the prefix Hamiltonian"),
        ("Mean |prefix energy - full CASCI| (Ha)", "Mean |prefix energy - prefix ground energy| (Ha)")):
    ax.set_title(heading, fontsize=10)
    ax.set_xlabel("Retained Hamiltonian terms")
    ax.set_ylabel(ylabel)
    ax.xaxis.set_major_locator(MaxNLocator(integer=True))
    if ax is axes[1]:
        ax.set_yscale("symlog", linthresh=1e-10)
        ax.yaxis.set_major_locator(FixedLocator([0, 1e-8, 1e-6, 1e-4, 1e-2, 1, 100]))
        ax.yaxis.set_minor_locator(NullLocator())
    ax.set_ylim(bottom=0)
    ax.grid(alpha=.2)
handles, labels = axes[0].get_legend_handles_labels()
fig.legend(handles, labels, loc="lower center", ncol=4, fontsize=8, frameon=False)
subtitle = args.label or "Fresh rerun"
if args.exclude_qaoa:
    subtitle += " | QAOA-inspired VQE excluded"
title(fig, "Hamiltonian prefix study", subtitle)
fig.text(.5, .18, "Shading: +/-1 population SD across molecule means; not confidence intervals.\n"
         "Left panel compares different Hamiltonians; it is not a full-Hamiltonian rescore.",
         ha="center", fontsize=9)
fig.tight_layout(rect=(0, .24, 1, .94))
finish(fig, args.output)
