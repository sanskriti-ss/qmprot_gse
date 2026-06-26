from pathlib import Path
import math

import matplotlib.pyplot as plt
import pandas as pd
from matplotlib.ticker import LogLocator, NullFormatter

root = Path(r"framework/experiments/results/trainability/20260619_222333_all_molecules_comparison")
csv_path = root / "all_molecules_results.csv"

df = pd.read_csv(csv_path)

algorithm_labels = {
    "qubit_adapt_vqe": "Qubit-ADAPT-VQE",
    "hardware_efficient_vqe": "Hardware-efficient VQE",
}

algorithm_colors = {
    "qubit_adapt_vqe": "#2F6FAE",
    "hardware_efficient_vqe": "#D97732",
}

df["algorithm_label"] = df["algorithm"].map(algorithm_labels).fillna(df["algorithm"])

molecules = sorted(df["molecule"].unique())
ncols = 4
nrows = math.ceil(len(molecules) / ncols)

plt.rcParams.update({
    "font.family": "serif",
    "font.size": 8,
    "axes.labelsize": 8,
    "axes.titlesize": 8,
    "legend.fontsize": 8,
    "xtick.labelsize": 7,
    "ytick.labelsize": 7,
    "figure.dpi": 150,
    "savefig.dpi": 600,
    "axes.linewidth": 0.6,
    "xtick.major.width": 0.6,
    "ytick.major.width": 0.6,
})

fig, axes = plt.subplots(
    nrows=nrows,
    ncols=ncols,
    figsize=(7.2, 9.2),
    sharex=True,
    sharey=True,
)

axes = axes.ravel()

for ax, molecule in zip(axes, molecules):
    sub = df[df["molecule"] == molecule]

    for algorithm in ["qubit_adapt_vqe", "hardware_efficient_vqe"]:
        part = sub[sub["algorithm"] == algorithm].sort_values("target_n_params")
        if part.empty:
            continue

        ax.plot(
            part["target_n_params"],
            part["n_cost_evals"],
            marker="o",
            markersize=3.0,
            linewidth=1.2,
            color=algorithm_colors[algorithm],
            label=algorithm_labels[algorithm],
        )

    ax.set_title(molecule)
    ax.set_yscale("log")
    ax.grid(True, which="major", axis="y", alpha=0.25, linewidth=0.5)
    ax.grid(True, which="minor", axis="y", alpha=0.12, linewidth=0.4)
    ax.yaxis.set_major_locator(LogLocator(base=10))
    ax.yaxis.set_minor_formatter(NullFormatter())
    ax.set_xticks(sorted(df["target_n_params"].unique()))

for ax in axes[len(molecules):]:
    ax.axis("off")

fig.supxlabel("Target number of variational parameters", y=0.045)
fig.supylabel("Cost-function evaluations", x=0.035)

handles, labels = axes[0].get_legend_handles_labels()
fig.legend(
    handles,
    labels,
    loc="upper center",
    ncol=2,
    frameon=False,
    bbox_to_anchor=(0.5, 0.995),
)

fig.suptitle(
    "Trainability comparison across molecules",
    y=1.025,
    fontsize=10,
)

fig.tight_layout(rect=(0.055, 0.06, 1.0, 0.965))

fig.savefig(root / "cost_evals_vs_params_faceted.pdf", bbox_inches="tight")
fig.savefig(root / "cost_evals_vs_params_faceted.png", bbox_inches="tight")
plt.close(fig)