"""
Active-space headroom report: E_CASCI - E_HF for every molecule
================================================================

A VQE benchmark can only rank ansaetze if there is *something to rank*.
The Hartree-Fock (HF) state is the free, classical starting point.  If the
active-space CASCI energy lies below HF by less than chemical accuracy
(1.6 mHa), then HF alone already "solves" the problem and every ansatz
looks equally good.  This script quantifies that headroom per molecule:

    gap = E_CASCI - E_HF          (<= 0; more negative = more correlation
                                   energy for an ansatz to recover)

For each molecule we report

* the active space ``(n_electrons e, n_orbitals o)`` and qubit count --
  one place to read the numbers that the paper must quote;
* ``E_HF`` from the PySCF pipeline and ``<HF|H|HF>`` evaluated on the
  *qubit Hamiltonian that the VQEs actually use* -- they must agree
  (``hf_consistency_ha``), otherwise the qubit Hamiltonian is wrong;
* ``E_CASCI`` and the gap in Ha and mHa;
* ``hf_within_chem_acc`` -- True when ``|gap| < 1.6 mHa`` (the molecule
  cannot discriminate between ansaetze);
* ``headroom_x_chem_acc`` -- ``|gap| / 1.6 mHa``; values near/below 1 mean
  no headroom, values >> 1 mean there is a real accuracy range to win.

Outputs  (``framework/experiments/results/active_space_report/<ts>_.../``)
-------------------------------------------------------------------------
* ``active_space_report.csv``  -- machine-readable table.
* ``active_space_report.md``   -- paper-ready table + verdict.
* ``active_space_gap.png``     -- bar chart of |gap| (mHa, log scale) with
  the chemical-accuracy line; molecules with no headroom are red.
* the same table printed to the console.

Run (WSL)
---------
::

    cd framework && source framework.venv/bin/activate
    python -m experiments.active_space_report                      # all molecules
    python -m experiments.active_space_report --molecules met cys glu
    python -m experiments.active_space_report --from-results experiments/results/trainability/<run>
        # reuse hf/casci already stored in run_config.json files (no PySCF)
"""

from __future__ import annotations

import argparse
import csv
import json
import logging
import sys
from pathlib import Path
from typing import Dict, List, Optional, Sequence

import numpy as np

_FW_DIR = Path(__file__).resolve().parent.parent
if str(_FW_DIR) not in sys.path:
    sys.path.insert(0, str(_FW_DIR))

from experiments._common import (  # noqa: E402
    CHEMICAL_ACCURACY_HA,
    DATASETS2_DIR,
    load_small_hamiltonian,
    make_run_dir,
)

logger = logging.getLogger(__name__)

CSV_COLUMNS = [
    "molecule", "n_electrons", "n_orbitals", "n_qubits",
    "e_hf_ha", "e_hf_qubit_ha", "hf_consistency_ha",
    "e_casci_ha", "gap_casci_minus_hf_ha", "gap_mha",
    "hf_within_chem_acc", "headroom_x_chem_acc",
]


def _row_from_energies(
    molecule: str,
    e_hf: float,
    e_casci: float,
    *,
    n_electrons: Optional[int] = None,
    n_orbitals: Optional[int] = None,
    n_qubits: Optional[int] = None,
    e_hf_qubit: Optional[float] = None,
) -> Dict[str, object]:
    gap = e_casci - e_hf
    return {
        "molecule": molecule,
        "n_electrons": n_electrons,
        "n_orbitals": n_orbitals,
        "n_qubits": n_qubits,
        "e_hf_ha": e_hf,
        "e_hf_qubit_ha": e_hf_qubit,
        "hf_consistency_ha": (
            abs(e_hf_qubit - e_hf) if e_hf_qubit is not None else None
        ),
        "e_casci_ha": e_casci,
        "gap_casci_minus_hf_ha": gap,
        "gap_mha": gap * 1e3,
        "hf_within_chem_acc": abs(gap) < CHEMICAL_ACCURACY_HA,
        "headroom_x_chem_acc": abs(gap) / CHEMICAL_ACCURACY_HA,
    }


def collect_row(molecule: str, basis: str = "sto-3g") -> Dict[str, object]:
    """Run the active-space pipeline for one molecule and build its row."""
    from core.hf_verification import _compute_hf_energy_diagonal

    loaded = load_small_hamiltonian(
        molecule=molecule, basis=basis, cs_target_qubits=None,
    )
    mol = loaded.hamiltonian.molecule
    try:
        # HF is a single basis state, so only the diagonal (I/Z-only) Pauli
        # terms contribute: exact, and fast even for 12+ qubits.
        n_elec = mol.n_electrons or loaded.n_qubits_final // 2
        e_hf_qubit = float(_compute_hf_energy_diagonal(loaded.hamiltonian, n_elec))
    except Exception as exc:  # pragma: no cover - diagnostic only
        logger.warning("%s: <HF|H|HF> on qubit Hamiltonian failed: %s", molecule, exc)
        e_hf_qubit = None
    return _row_from_energies(
        molecule,
        float(loaded.hf_energy),
        float(loaded.casci_energy),
        n_electrons=mol.n_electrons,
        n_orbitals=mol.n_orbitals,
        n_qubits=loaded.n_qubits_final,
        e_hf_qubit=e_hf_qubit,
    )


def rows_from_results(results_root: Path) -> List[Dict[str, object]]:
    """Read hf/casci from per-molecule ``run_config.json`` files of an
    existing experiment folder (no recomputation)."""
    rows = []
    for cfg_path in sorted(results_root.glob("*/run_config.json")):
        with open(cfg_path) as fh:
            cfg = json.load(fh)
        if cfg.get("hf_energy") is None or cfg.get("casci_energy") is None:
            continue
        rows.append(_row_from_energies(
            cfg_path.parent.name,
            float(cfg["hf_energy"]),
            float(cfg["casci_energy"]),
            n_qubits=cfg.get("n_qubits_active"),
        ))
    return rows


def discover_molecules() -> List[str]:
    return sorted(
        p.name for p in DATASETS2_DIR.iterdir()
        if p.is_dir() and (p / f"{p.name}.h5").is_file()
    )


# ──────────────────────────────────────────────────────────────────────────
# Output
# ──────────────────────────────────────────────────────────────────────────

def _fmt(v, spec: str = "") -> str:
    if v is None or (isinstance(v, float) and np.isnan(v)):
        return "-"
    return format(v, spec)


def render_table(rows: Sequence[Dict[str, object]]) -> str:
    head = (
        f"{'molecule':<14}{'(e,o)':>8}{'qubits':>7}{'E_HF (Ha)':>16}"
        f"{'E_CASCI (Ha)':>16}{'gap (mHa)':>11}{'x chem.acc':>11}  HF already enough?"
    )
    lines = [head, "-" * len(head)]
    for r in rows:
        eo = (f"({r['n_electrons']},{r['n_orbitals']})"
              if r["n_electrons"] is not None else "-")
        verdict = "YES - cannot rank ansaetze" if r["hf_within_chem_acc"] else "no"
        lines.append(
            f"{r['molecule']:<14}{eo:>8}{_fmt(r['n_qubits']):>7}"
            f"{r['e_hf_ha']:>16.6f}{r['e_casci_ha']:>16.6f}"
            f"{r['gap_mha']:>11.2f}{r['headroom_x_chem_acc']:>11.1f}  {verdict}"
        )
    return "\n".join(lines)


def render_markdown(rows: Sequence[Dict[str, object]]) -> str:
    n = len(rows)
    flat = [r for r in rows if r["hf_within_chem_acc"]]
    gaps = np.array([abs(r["gap_mha"]) for r in rows])
    out = [
        "# Active-space headroom: E_CASCI - E_HF",
        "",
        f"Chemical accuracy = {CHEMICAL_ACCURACY_HA * 1e3:.1f} mHa.  "
        "`gap = E_CASCI - E_HF` is the correlation energy an ansatz can recover "
        "inside the chosen active space.",
        "",
        "| molecule | (e,o) | qubits | E_HF (Ha) | E_CASCI (Ha) | gap (mHa) | x chem.acc | HF within chem. acc.? |",
        "|---|---|---|---|---|---|---|---|",
    ]
    for r in rows:
        eo = (f"({r['n_electrons']}e,{r['n_orbitals']}o)"
              if r["n_electrons"] is not None else "-")
        out.append(
            f"| {r['molecule']} | {eo} | {_fmt(r['n_qubits'])} | "
            f"{r['e_hf_ha']:.6f} | {r['e_casci_ha']:.6f} | {r['gap_mha']:.2f} | "
            f"{r['headroom_x_chem_acc']:.1f} | {'**YES**' if r['hf_within_chem_acc'] else 'no'} |"
        )
    out += [
        "",
        "## Verdict",
        "",
        f"* {len(flat)} of {n} molecules have HF already within chemical accuracy "
        f"of CASCI ({', '.join(r['molecule'] for r in flat) or 'none'}). "
        "On these, no ansatz can be distinguished from simply returning HF.",
        f"* Median |gap| = {np.median(gaps):.2f} mHa, "
        f"min = {gaps.min():.2f} mHa, max = {gaps.max():.2f} mHa.",
    ]
    bad = [r for r in rows
           if r["hf_consistency_ha"] is not None and r["hf_consistency_ha"] > 1e-6]
    if bad:
        out.append(
            "* WARNING: <HF|H|HF> on the qubit Hamiltonian disagrees with the "
            "pipeline HF energy for: "
            + ", ".join(f"{r['molecule']} ({r['hf_consistency_ha']:.2e} Ha)" for r in bad)
        )
    else:
        out.append(
            "* <HF|H|HF> evaluated on the qubit Hamiltonian matches the pipeline "
            "HF energy (< 1e-6 Ha) for every molecule that was checked."
        )
    return "\n".join(out) + "\n"


def make_plot(rows: Sequence[Dict[str, object]], path: Path) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    rows = sorted(rows, key=lambda r: abs(r["gap_mha"]))
    names = [r["molecule"] for r in rows]
    vals = [abs(r["gap_mha"]) for r in rows]
    colors = ["#d62728" if r["hf_within_chem_acc"] else "#4c78a8" for r in rows]

    fig, ax = plt.subplots(figsize=(max(8, 0.55 * len(rows) + 3), 5.5))
    ax.bar(range(len(rows)), vals, color=colors)
    ax.axhline(CHEMICAL_ACCURACY_HA * 1e3, color="black", linestyle="--", linewidth=1.2,
               label=f"chemical accuracy ({CHEMICAL_ACCURACY_HA * 1e3:.1f} mHa)")
    ax.set_yscale("log")
    ax.set_xticks(range(len(rows)))
    ax.set_xticklabels(names, rotation=45, ha="right")
    ax.set_ylabel(r"$|E_{\mathrm{CASCI}} - E_{\mathrm{HF}}|$  (mHa, log scale)")
    ax.set_title(
        "Correlation energy available inside each active space\n"
        "red = HF already within chemical accuracy (no headroom to rank ansaetze)"
    )
    ax.grid(True, axis="y", alpha=0.3)
    ax.legend(loc="upper left")
    fig.tight_layout()
    fig.savefig(path, dpi=160)
    plt.close(fig)


def save_report(rows: Sequence[Dict[str, object]], out_dir: Path) -> Dict[str, Path]:
    out_dir.mkdir(parents=True, exist_ok=True)
    csv_path = out_dir / "active_space_report.csv"
    with open(csv_path, "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=CSV_COLUMNS)
        w.writeheader()
        for r in rows:
            w.writerow({k: r.get(k) for k in CSV_COLUMNS})
    md_path = out_dir / "active_space_report.md"
    md_path.write_text(render_markdown(rows), encoding="utf-8")
    png_path = out_dir / "active_space_gap.png"
    make_plot(rows, png_path)
    return {"csv": csv_path, "markdown": md_path, "plot": png_path}


def run_active_space_report(
    molecules: Optional[Sequence[str]] = None,
    basis: str = "sto-3g",
    output_dir: Optional[Path] = None,
    from_results: Optional[Path] = None,
) -> Dict[str, object]:
    if from_results is not None:
        rows = rows_from_results(Path(from_results))
    else:
        rows = []
        for mol in (molecules or discover_molecules()):
            try:
                logger.info("[active-space] %s", mol)
                rows.append(collect_row(mol, basis))
            except Exception as exc:
                logger.exception("active-space report failed for %s: %s", mol, exc)
    if not rows:
        raise RuntimeError("No molecules produced a row")
    rows.sort(key=lambda r: r["molecule"])
    out_dir = Path(output_dir) if output_dir else make_run_dir(
        "active_space_report", "all_molecules", "report",
    )
    paths = save_report(rows, out_dir)
    return {"rows": rows, "output_dir": out_dir, **paths}


def main() -> None:
    p = argparse.ArgumentParser(
        description="Report E_CASCI - E_HF (correlation headroom) per molecule.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument("--molecules", nargs="+", default=None,
                   help="Folder names under datasets2/ (default: all)")
    p.add_argument("--basis", default="sto-3g")
    p.add_argument("--output-dir", type=Path, default=None)
    p.add_argument("--from-results", type=Path, default=None,
                   help="Read hf_energy/casci_energy from <dir>/*/run_config.json "
                        "instead of recomputing")
    p.add_argument("-v", "--verbose", action="store_true")
    args = p.parse_args()
    logging.basicConfig(level=logging.INFO if args.verbose else logging.WARNING)

    out = run_active_space_report(
        molecules=args.molecules, basis=args.basis,
        output_dir=args.output_dir, from_results=args.from_results,
    )
    print(render_table(out["rows"]))
    print()
    print(render_markdown(out["rows"]).split("## Verdict")[1].strip())
    print(f"\nWrote: {out['csv']}\n       {out['markdown']}\n       {out['plot']}")


if __name__ == "__main__":
    main()
