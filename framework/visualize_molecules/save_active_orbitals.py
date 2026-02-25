"""Save a composite image showing all chosen active-space orbitals.

Renders each active orbital as a separate subplot in a single PyVista
off-screen window, then saves to PNG. This gives a permanent record
of which orbitals were selected for the active space.

Usage
-----
    python -m framework.visualize_molecules.save_active_orbitals gly
"""

import sys
import os
import numpy as np
from pathlib import Path
from typing import Optional

import pyvista as pv

_parent = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _parent not in sys.path:
    sys.path.insert(0, _parent)

from active_space_truncation.geometry import MoleculeGeometry, get_geometry, list_available
from active_space_truncation.step1_orbitals import OrbitalDiagnostics, run_orbital_analysis
from visualize_molecules.orbital_grid import (
    build_grid,
    evaluate_orbital,
    compute_isosurface_level,
    orbital_angular_momentum_label,
)
from visualize_molecules.renderer import (
    add_atoms,
    add_bonds,
    add_orbital_isosurface,
    ORBITAL_COLORS,
)


def save_active_orbital_summary(
    geometry: MoleculeGeometry,
    diagnostics: OrbitalDiagnostics,
    save_path: Optional[str] = None,
    resolution: float = 0.25,
    iso_fraction: float = 0.03,
) -> str:
    """Render all active orbitals into a multi-panel PNG.

    Parameters
    ----------
    geometry : MoleculeGeometry
    diagnostics : OrbitalDiagnostics (must have mol, mo_coeff)
    save_path : output file path. Auto-generated if None.
    resolution : grid spacing in Angstroms
    iso_fraction : isosurface level as fraction of peak

    Returns
    -------
    Path to the saved image.
    """
    active_indices = diagnostics.proposed_active_indices
    n_active = len(active_indices)

    if n_active == 0:
        raise ValueError("No active orbitals to visualize.")

    # Grid layout: up to 3 columns
    n_cols = min(n_active, 3)
    n_rows = (n_active + n_cols - 1) // n_cols

    # Build shared grid
    grid_points, grid_shape, x, y, z = build_grid(
        geometry.coords, padding=4.0, resolution=resolution,
    )

    # Create multi-panel plotter
    plotter = pv.Plotter(
        shape=(n_rows, n_cols),
        off_screen=True,
        window_size=(550 * n_cols, 500 * n_rows),
        border=False,
    )

    for panel_idx, orb_idx in enumerate(active_indices):
        row = panel_idx // n_cols
        col = panel_idx % n_cols
        plotter.subplot(row, col)
        plotter.set_background("#1a1a2e")

        # Molecule structure
        add_atoms(plotter, geometry.atoms, geometry.coords, label=False)
        add_bonds(plotter, geometry.atoms, geometry.coords)

        # Orbital isosurface
        values = evaluate_orbital(
            diagnostics.mol, diagnostics.mo_coeff,
            orb_idx, grid_points, grid_shape,
        )
        iso = compute_isosurface_level(values, iso_fraction)
        add_orbital_isosurface(
            plotter, values, x, y, z,
            isovalue=iso, orbital_type="active",
        )

        # Label
        ang = orbital_angular_momentum_label(diagnostics.mol, diagnostics.mo_coeff, orb_idx)
        energy_str = ""
        if diagnostics.orbital_energies is not None and orb_idx < len(diagnostics.orbital_energies):
            energy_str = f"  E={diagnostics.orbital_energies[orb_idx]:.3f} Ha"
        occ_str = ""
        if diagnostics.natural_occupations is not None and orb_idx < len(diagnostics.natural_occupations):
            occ_str = f"  occ={diagnostics.natural_occupations[orb_idx]:.3f}"

        plotter.add_text(
            f"MO {orb_idx} [{ang}]{energy_str}{occ_str}",
            position="upper_left", font_size=9, color="white",
        )

    # Fill empty panels
    for panel_idx in range(n_active, n_rows * n_cols):
        row = panel_idx // n_cols
        col = panel_idx % n_cols
        plotter.subplot(row, col)
        plotter.set_background("#1a1a2e")

    # Title
    plotter.subplot(0, 0)
    n_e = diagnostics.proposed_n_active_electrons
    n_o = diagnostics.proposed_n_active_orbitals
    plotter.add_text(
        f"{geometry.name.upper()} — Active Space ({n_e}e, {n_o}o) — {n_o * 2} qubits",
        position="upper_edge", font_size=12, color="white",
    )

    # Output path
    if save_path is None:
        out_dir = Path(__file__).parent / "output"
        out_dir.mkdir(exist_ok=True)
        save_path = str(out_dir / f"{geometry.name}_active_orbitals.png")

    plotter.show(screenshot=save_path, auto_close=True)
    print(f"Saved active orbital summary: {save_path}")
    return save_path


# ---------- convenience ----------

def save_for_molecule(
    name: str,
    basis: str = "cc-pvdz",
    save_path: Optional[str] = None,
) -> str:
    """One-liner: load, analyse, save active orbital image."""
    print(f"Loading {name}...")
    geom = get_geometry(name)
    print(f"Running orbital analysis ({geom.n_atoms} atoms, basis={basis})...")
    diag = run_orbital_analysis(geom, basis=basis)
    print(diag.summary())
    return save_active_orbital_summary(geom, diag, save_path=save_path)


# ---------- CLI ----------

def main():
    args = sys.argv[1:]
    if not args or args[0] in ("-h", "--help"):
        available = list_available()
        print("Usage: python -m framework.visualize_molecules.save_active_orbitals <molecule> [--output path.png]")
        print(f"\nAvailable: {', '.join(available)}")
        sys.exit(0)

    mol_name = args[0]
    save_path = None
    if "--output" in args:
        idx = args.index("--output")
        if idx + 1 < len(args):
            save_path = args[idx + 1]

    save_for_molecule(mol_name, save_path=save_path)


if __name__ == "__main__":
    main()
