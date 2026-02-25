"""3D molecule visualization with orbital classification.

Renders an interactive 3D model showing:
- Atoms as colored spheres (CPK colors)
- Bonds as gray cylinders
- Orbital lobes color-coded:
    RED   = frozen core orbitals
    GREEN = active (NOT frozen) orbitals
    GRAY  = virtual (outside active space)
"""

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from typing import List, Optional, Dict, Tuple
import os
import sys

# Allow running both as module and standalone
_parent = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _parent not in sys.path:
    sys.path.insert(0, _parent)

from active_space_truncation.geometry import MoleculeGeometry, CORE_ELECTRONS
from active_space_truncation.step1_orbitals import OrbitalDiagnostics

# CPK colors
ATOM_COLORS = {
    "H": "#FFFFFF", "C": "#505050", "N": "#3050F8", "O": "#FF0D0D",
    "F": "#90E050", "S": "#FFFF30", "P": "#FF8000", "Li": "#CC80FF",
}
ATOM_RADII = {
    "H": 0.25, "C": 0.76, "N": 0.71, "O": 0.66,
    "F": 0.57, "S": 1.05, "P": 1.07, "Li": 1.28,
}
BOND_THRESHOLD = 1.85


def visualize_molecule_orbitals(
    geometry: MoleculeGeometry,
    diagnostics: OrbitalDiagnostics,
    save_path: Optional[str] = None,
    show: bool = False,
    figsize: Tuple[int, int] = (14, 10),
) -> str:
    """Create 3D visualization with orbital classification.

    Returns path to saved image.
    """
    fig = plt.figure(figsize=figsize, facecolor="white")
    ax = fig.add_subplot(111, projection="3d", computed_zorder=False)

    coords = geometry.coords
    atoms = geometry.atoms

    # Draw bonds
    for i in range(len(atoms)):
        for j in range(i + 1, len(atoms)):
            if np.linalg.norm(coords[i] - coords[j]) < BOND_THRESHOLD:
                ax.plot(*zip(coords[i], coords[j]), color="#888888", linewidth=3, zorder=1)

    # Draw atoms
    for i, (atom, coord) in enumerate(zip(atoms, coords)):
        color = ATOM_COLORS.get(atom, "#808080")
        radius = ATOM_RADII.get(atom, 0.5)
        ax.scatter(*coord, s=radius * 800, c=color, edgecolors="black",
                   linewidth=1.0, alpha=0.95, zorder=5, depthshade=True)
        ax.text(coord[0], coord[1], coord[2] + radius * 0.6, atom,
                fontsize=9, ha="center", va="bottom", fontweight="bold", zorder=10)

    # Assign orbitals to atoms and draw lobes
    n_core = diagnostics.n_core_orbitals
    active_set = set(diagnostics.proposed_active_indices)
    n_mo = diagnostics.n_molecular_orbitals
    atom_orbs = _assign_orbitals_to_atoms(geometry, diagnostics)

    for atom_idx, orb_list in atom_orbs.items():
        coord = coords[atom_idx]
        n_orbs = len(orb_list)
        if n_orbs == 0:
            continue
        for k, (orb_idx, orb_type) in enumerate(orb_list):
            angle = 2 * np.pi * k / max(n_orbs, 1)
            lobe_pos = coord + np.array([0.45 * np.cos(angle), 0.45 * np.sin(angle), 0.0])
            if orb_type == "core":
                color, alpha, size = "#D32F2F", 0.7, 80
            elif orb_type == "active":
                color, alpha, size = "#2E7D32", 0.85, 120
            else:
                color, alpha, size = "#9E9E9E", 0.3, 40
            ax.scatter(*lobe_pos, s=size, c=color, alpha=alpha, marker="o",
                       edgecolors="none", zorder=3, depthshade=True)

    # Styling
    ax.set_xlabel("X (A)", fontsize=10)
    ax.set_ylabel("Y (A)", fontsize=10)
    ax.set_zlabel("Z (A)", fontsize=10)

    n_virtual = n_mo - n_core - diagnostics.proposed_n_active_orbitals
    ax.set_title(
        f"{geometry.name.upper()} ({geometry.formula})  —  "
        f"Core frozen: {n_core} orbs (red)  |  "
        f"Active: {diagnostics.proposed_n_active_orbitals} orbs (green)  |  "
        f"Virtual: {n_virtual} orbs (gray)",
        fontsize=12, fontweight="bold", pad=20,
    )

    legend_elements = [
        Line2D([0], [0], marker="o", color="w", markerfacecolor="#D32F2F",
               markersize=10, label=f"Frozen core ({n_core} orbitals)"),
        Line2D([0], [0], marker="o", color="w", markerfacecolor="#2E7D32",
               markersize=12, label=f"Active / NOT frozen ({diagnostics.proposed_n_active_orbitals} orbitals)"),
        Line2D([0], [0], marker="o", color="w", markerfacecolor="#9E9E9E",
               markersize=8, label=f"Virtual ({n_virtual} orbitals)"),
    ]
    ax.legend(handles=legend_elements, loc="upper left", fontsize=10, framealpha=0.9)

    _set_equal_aspect(ax, coords)
    ax.view_init(elev=20, azim=135)
    ax.grid(True, alpha=0.3)
    plt.tight_layout()

    if save_path is None:
        out_dir = os.path.join(os.path.dirname(__file__), "output")
        os.makedirs(out_dir, exist_ok=True)
        save_path = os.path.join(out_dir, f"{geometry.name}_orbitals.png")
    plt.savefig(save_path, dpi=150, bbox_inches="tight", facecolor="white")
    print(f"  Saved: {save_path}")

    if show:
        plt.show()
    plt.close(fig)
    return save_path


def _assign_orbitals_to_atoms(
    geometry: MoleculeGeometry,
    diagnostics: OrbitalDiagnostics,
) -> Dict[int, List[Tuple[int, str]]]:
    """Assign each MO to the atom it's most localized on via |C|^2."""
    n_core = diagnostics.n_core_orbitals
    active_set = set(diagnostics.proposed_active_indices)
    mo_coeff = diagnostics.mo_coeff
    mol = diagnostics.mol

    atom_orbs: Dict[int, List[Tuple[int, str]]] = {i: [] for i in range(geometry.n_atoms)}

    if mo_coeff is None or mol is None:
        return _fallback_assignment(geometry, diagnostics)

    ao_labels = mol.ao_labels(fmt=False)
    n_ao = mo_coeff.shape[0]
    n_mo = min(mo_coeff.shape[1], diagnostics.n_molecular_orbitals)

    # Show: all core, all active, up to 5 virtual
    orbs_to_show = list(range(n_core)) + diagnostics.proposed_active_indices
    virtual_start = max(diagnostics.proposed_active_indices) + 1 if diagnostics.proposed_active_indices else n_core
    for v in range(virtual_start, min(virtual_start + 5, n_mo)):
        orbs_to_show.append(v)

    for orb_idx in orbs_to_show:
        if orb_idx >= n_mo:
            continue
        atom_weights = np.zeros(geometry.n_atoms)
        for ao_idx in range(n_ao):
            atom_weights[ao_labels[ao_idx][0]] += mo_coeff[ao_idx, orb_idx] ** 2
        best_atom = int(np.argmax(atom_weights))
        if orb_idx < n_core:
            orb_type = "core"
        elif orb_idx in active_set:
            orb_type = "active"
        else:
            orb_type = "virtual"
        atom_orbs[best_atom].append((orb_idx, orb_type))

    return atom_orbs


def _fallback_assignment(geometry, diagnostics):
    atom_orbs = {i: [] for i in range(geometry.n_atoms)}
    orb_counter = 0
    for i, atom in enumerate(geometry.atoms):
        n_core_for_atom = CORE_ELECTRONS.get(atom, 0) // 2
        for _ in range(n_core_for_atom):
            atom_orbs[i].append((orb_counter, "core"))
            orb_counter += 1
    heavy = [i for i, a in enumerate(geometry.atoms) if a != "H"] or list(range(geometry.n_atoms))
    for k, idx in enumerate(diagnostics.proposed_active_indices):
        atom_orbs[heavy[k % len(heavy)]].append((idx, "active"))
    return atom_orbs


def _set_equal_aspect(ax, coords):
    max_range = (coords.max(axis=0) - coords.min(axis=0)).max() / 2.0
    mid = coords.mean(axis=0)
    margin = max(max_range * 1.3, 1.0)
    ax.set_xlim(mid[0] - margin, mid[0] + margin)
    ax.set_ylim(mid[1] - margin, mid[1] + margin)
    ax.set_zlim(mid[2] - margin, mid[2] + margin)


def visualize_all_molecules(show: bool = False) -> List[str]:
    """Run orbital analysis + visualization for all available molecules."""
    from active_space_truncation.geometry import list_available, get_geometry
    from active_space_truncation.step1_orbitals import run_orbital_analysis

    saved = []
    for mol_name in list_available():
        print(f"\n--- Visualizing {mol_name} ---")
        try:
            geom = get_geometry(mol_name)
            diag = run_orbital_analysis(geom)
            path = visualize_molecule_orbitals(geom, diag, show=show)
            saved.append(path)
        except Exception as e:
            print(f"  ERROR: {e}")
    return saved


if __name__ == "__main__":
    paths = visualize_all_molecules(show=False)
    print(f"\nSaved {len(paths)} visualizations:")
    for p in paths:
        print(f"  {p}")
