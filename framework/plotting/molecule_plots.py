import matplotlib.pyplot as plt
import numpy as np
import os
import sys
from pathlib import Path
from typing import List, Dict, Any, Optional

# Add parent directory to path to access utils
sys.path.insert(0, str(Path(__file__).parent.parent.parent))
from utils.csv_export import export_results_to_csv


def plot_selected_molecules_with_hf_ref(
    results: List[Any],
    output_dir: str,
    filename: str = "selected_molecules_withHFref.png",
):
    """
    Plot VQE energies per molecule with HF reference energy shown.
    
    Each molecule gets a group of bars (one per VQE method) plus a
    horizontal dashed line for the HF energy and a dotted line for exact energy.
    """
    # Organize results by molecule
    molecules: Dict[str, Dict[str, Any]] = {}
    for r in results:
        mol = getattr(r, "molecule_name", "unknown")
        if mol not in molecules:
            molecules[mol] = {
                "methods": [],
                "energies": [],
                "hf_energy": getattr(r, "hf_energy", None),
                "exact_energy": getattr(r, "exact_energy", getattr(r, "reference_energy", None)),
            }
        molecules[mol]["methods"].append(getattr(r, "method", getattr(r, "algorithm_name", "?")))
        molecules[mol]["energies"].append(getattr(r, "energy", getattr(r, "calculated_energy", 0.0)))
        # Update hf/exact if not yet set
        if molecules[mol]["hf_energy"] is None:
            molecules[mol]["hf_energy"] = getattr(r, "hf_energy", None)
        if molecules[mol]["exact_energy"] is None:
            molecules[mol]["exact_energy"] = getattr(r, "exact_energy", getattr(r, "reference_energy", None))

    n_molecules = len(molecules)
    if n_molecules == 0:
        return

    fig, axes = plt.subplots(1, n_molecules, figsize=(6 * n_molecules, 6), squeeze=False)
    axes = axes.flatten()

    colors = plt.cm.Set2(np.linspace(0, 1, 8))

    for idx, (mol_name, data) in enumerate(molecules.items()):
        ax = axes[idx]
        methods = data["methods"]
        energies = data["energies"]
        hf_e = data["hf_energy"]
        exact_e = data["exact_energy"]

        x = np.arange(len(methods))
        bars = ax.bar(x, energies, color=[colors[i % len(colors)] for i in range(len(methods))],
                      edgecolor="black", linewidth=0.5, zorder=3)

        # HF reference line
        if hf_e is not None:
            ax.axhline(y=hf_e, color="red", linestyle="--", linewidth=1.5,
                        label=f"HF energy = {hf_e:.6f}", zorder=4)

        # Exact energy line
        if exact_e is not None:
            ax.axhline(y=exact_e, color="green", linestyle=":", linewidth=1.5,
                        label=f"Exact = {exact_e:.6f}", zorder=4)

        ax.set_xticks(x)
        ax.set_xticklabels(methods, rotation=45, ha="right", fontsize=9)
        ax.set_ylabel("Energy (Hartree)")
        ax.set_title(mol_name, fontsize=12, fontweight="bold")
        ax.legend(fontsize=8, loc="best")
        ax.grid(axis="y", alpha=0.3, zorder=0)

        # Annotate bars with values
        for bar, val in zip(bars, energies):
            ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height(),
                    f"{val:.6f}", ha="center", va="bottom", fontsize=7)

    plt.suptitle("VQE Energies with HF Reference", fontsize=14, fontweight="bold")
    plt.tight_layout()

    os.makedirs(output_dir, exist_ok=True)
    out_path = os.path.join(output_dir, filename)
    plt.savefig(out_path, dpi=200, bbox_inches="tight")
    plt.close()
    print(f"Plot saved to {out_path}")

    # Also export CSV alongside the plot
    csv_path = os.path.join(output_dir, "vqe_energies_summary.csv")
    export_results_to_csv(results, csv_path)
