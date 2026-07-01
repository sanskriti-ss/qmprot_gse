"""
Create HF verification bar chart from a completed Part B all-molecules output folder.

The chart mirrors the paper-style figure:
Reference (CASCI) vs <HF|H|HF> vs Best VQE (minimum final energy per molecule).

Usage:
    python plot_hf_verification_from_part_b.py \
        --output-root framework/experiments/results/barren_plateau/<timestamp>_all_molecules_part_b_bayesian
"""
from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import List, Tuple

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


def _read_best_vqe_energy(optimization_csv: Path) -> float:
    with open(optimization_csv, newline="") as handle:
        reader = csv.DictReader(handle)
        vals = [float(row["final_energy"]) for row in reader if row.get("final_energy")]
    if not vals:
        raise ValueError(f"No final_energy values in {optimization_csv}")
    return min(vals)


def _collect_rows(output_root: Path) -> List[Tuple[str, float, float, float]]:
    manifest_path = output_root / "batch_manifest.json"
    if not manifest_path.is_file():
        raise FileNotFoundError(f"Missing batch_manifest.json in {output_root}")

    with open(manifest_path) as handle:
        manifest = json.load(handle)

    molecules = manifest.get("molecules_succeeded") or manifest.get("molecules_selected") or []
    rows: List[Tuple[str, float, float, float]] = []

    for molecule in molecules:
        molecule_dir = output_root / molecule
        run_cfg = molecule_dir / "run_config.json"
        opt_csv = molecule_dir / "optimization_results.csv"
        if not run_cfg.is_file() or not opt_csv.is_file():
            continue

        with open(run_cfg) as handle:
            cfg = json.load(handle)

        reference_energy = cfg.get("casci_energy")
        hf_energy = cfg.get("hf_energy")
        if reference_energy is None or hf_energy is None:
            continue

        best_vqe_energy = _read_best_vqe_energy(opt_csv)
        rows.append((molecule, float(reference_energy), float(hf_energy), float(best_vqe_energy)))

    if not rows:
        raise RuntimeError("No valid molecule rows found to plot")

    return rows


def make_plot(output_root: Path, title: str) -> Path:
    rows = _collect_rows(output_root)

    molecules = [r[0] for r in rows]
    ref_energies = [r[1] for r in rows]
    hf_energies = [r[2] for r in rows]
    vqe_energies = [r[3] for r in rows]

    x = np.arange(len(molecules))
    width = 0.26

    fig, ax = plt.subplots(figsize=(16, 7))
    ax.bar(x - width, ref_energies, width, label="Reference (exact)", color="#7f7f7f", alpha=0.85)
    ax.bar(x, hf_energies, width, label="<HF|H|HF>", color="#ff7f0e", alpha=0.85)
    ax.bar(x + width, vqe_energies, width, label="Best VQE", color="#1f77b4", alpha=0.85)

    ax.set_xticks(x)
    ax.set_xticklabels(molecules, rotation=35, ha="right")
    ax.set_ylabel("Energy (Hartree)")
    ax.set_title(title)
    ax.legend(loc="best")
    ax.grid(True, axis="y", alpha=0.25)

    fig.tight_layout()

    png_path = output_root / "hf_verification_all_molecules.png"
    fig.savefig(png_path, dpi=180, bbox_inches="tight")
    plt.close(fig)

    summary_path = output_root / "hf_verification_all_molecules.csv"
    with open(summary_path, "w", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(["molecule", "reference_exact", "hf_expectation", "best_vqe"])
        for row in rows:
            writer.writerow(row)

    return png_path


def _build_argparser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Create HF verification chart from a Part B all-molecules output folder.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument(
        "--title",
        default="HF Verification | 24 molecules",
        help="Plot title",
    )
    return parser


def main() -> None:
    args = _build_argparser().parse_args()
    output_path = make_plot(args.output_root, args.title)
    print(f"Saved plot: {output_path}")


if __name__ == "__main__":
    main()
