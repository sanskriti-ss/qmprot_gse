"""Export VQE results to CSV files."""

import csv
import os
from typing import List, Dict, Any, Optional


def export_results_to_csv(
    results: List[Any],
    output_path: str,
    additional_fields: Optional[Dict[str, Any]] = None,
):
    """
    Export a list of VQEResult objects to a CSV file.
    
    Parameters
    ----------
    results : list of VQEResult
        The results from all VQE runs.
    output_path : str
        Path to the output CSV file.
    additional_fields : dict, optional
        Extra columns to include for every row.
    """
    if not results:
        return

    fieldnames = [
        "molecule",
        "method",
        "vqe_energy",
        "hf_energy",
        "exact_energy",
        "error_vs_exact",
        "error_vs_hf",
        "iterations",
    ]
    if additional_fields:
        fieldnames.extend(additional_fields.keys())

    os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)

    with open(output_path, "w", newline="") as csvfile:
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
        writer.writeheader()

        for r in results:
            exact = getattr(r, "exact_energy", None)
            hf = getattr(r, "hf_energy", None)
            energy = getattr(r, "energy", None)

            row = {
                "molecule": getattr(r, "molecule_name", "unknown"),
                "method": getattr(r, "method", "unknown"),
                "vqe_energy": f"{energy:.10f}" if energy is not None else "",
                "hf_energy": f"{hf:.10f}" if hf is not None else "",
                "exact_energy": f"{exact:.10f}" if exact is not None else "",
                "error_vs_exact": (
                    f"{abs(energy - exact):.10f}"
                    if energy is not None and exact is not None
                    else ""
                ),
                "error_vs_hf": (
                    f"{energy - hf:.10f}"
                    if energy is not None and hf is not None
                    else ""
                ),
                "iterations": getattr(r, "iterations", ""),
            }
            if additional_fields:
                row.update(additional_fields)

            writer.writerow(row)

    print(f"Results exported to {output_path}")


def build_summary_table(results: List[Any]) -> str:
    """
    Build a formatted text summary table from results.
    
    Returns a string suitable for logging or printing.
    """
    lines = []
    header = f"{'Molecule':<20} {'Method':<15} {'VQE Energy':>14} {'HF Energy':>14} {'Exact Energy':>14} {'Err vs Exact':>14}"
    lines.append(header)
    lines.append("-" * len(header))

    for r in results:
        exact = getattr(r, "exact_energy", None)
        hf = getattr(r, "hf_energy", None)
        energy = getattr(r, "energy", None)
        err = abs(energy - exact) if energy is not None and exact is not None else None

        lines.append(
            f"{getattr(r, 'molecule_name', '?'):<20} "
            f"{getattr(r, 'method', '?'):<15} "
            f"{energy:>14.8f} "
            f"{hf if hf is not None else 'N/A':>14} "
            f"{exact if exact is not None else 'N/A':>14} "
            f"{err if err is not None else 'N/A':>14}"
        )

    return "\n".join(lines)
