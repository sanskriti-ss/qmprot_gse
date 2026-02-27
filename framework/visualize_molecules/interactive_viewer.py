"""Interactive 3D orbital viewer.

Main entry point that ties together orbital evaluation and PyVista
rendering. Supports viewing individual orbitals, cycling through
them with keyboard controls, and batch rendering.

Usage
-----
    python -m framework.visualize_molecules.interactive_viewer ala

Controls
--------
    Mouse drag   : rotate
    Scroll       : zoom
    Right-drag   : pan
    N / P        : next / previous orbital
    T            : toggle orbital type filter (all → core → active → virtual)
    R            : reset camera
    Q            : quit
"""

import sys
import os
import numpy as np
from typing import Optional, List

# Path setup for standalone execution
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
    create_plotter,
    add_atoms,
    add_bonds,
    add_orbital_isosurface,
    ORBITAL_COLORS,
)


class OrbitalViewer:
    """Interactive viewer that lets you browse orbitals one at a time.

    Parameters
    ----------
    geometry : MoleculeGeometry
    diagnostics : OrbitalDiagnostics
    resolution : grid spacing (Angstroms). Smaller = finer but slower.
    padding : extra grid extent around the molecule (Angstroms).
    iso_fraction : isosurface level as fraction of peak amplitude.
    """

    TYPE_CYCLE = ["all", "core", "active", "virtual"]

    def __init__(
        self,
        geometry: MoleculeGeometry,
        diagnostics: OrbitalDiagnostics,
        resolution: float = 0.25,
        padding: float = 4.0,
        iso_fraction: float = 0.03,
    ):
        self.geometry = geometry
        self.diag = diagnostics
        self.resolution = resolution
        self.padding = padding
        self.iso_fraction = iso_fraction

        # Classify every orbital
        self._build_orbital_list()

        # Precompute grid (shared across all orbitals)
        self.grid_points, self.grid_shape, self.x, self.y, self.z = build_grid(
            geometry.coords, padding=padding, resolution=resolution,
        )

        # State
        self._current_idx = 0
        self._type_filter_idx = 0  # index into TYPE_CYCLE
        self._plotter: Optional[object] = None

        # Cache evaluated orbitals to avoid recomputation
        self._cache = {}

    # ---- orbital index bookkeeping ----

    def _build_orbital_list(self):
        """Create a flat list of (orbital_index, type_str) for browsing."""
        n_core = self.diag.n_core_orbitals
        active_set = set(self.diag.proposed_active_indices)
        n_mo = self.diag.n_molecular_orbitals

        self._orbitals = []
        for i in range(n_mo):
            if i < n_core:
                otype = "core"
            elif i in active_set:
                otype = "active"
            else:
                otype = "virtual"
            self._orbitals.append((i, otype))

    def _filtered_orbitals(self) -> List[int]:
        """Indices into self._orbitals matching the current type filter."""
        filt = self.TYPE_CYCLE[self._type_filter_idx]
        if filt == "all":
            return list(range(len(self._orbitals)))
        return [i for i, (_, t) in enumerate(self._orbitals) if t == filt]

    # ---- orbital evaluation (cached) ----

    def _get_orbital_data(self, orb_idx: int) -> np.ndarray:
        if orb_idx not in self._cache:
            self._cache[orb_idx] = evaluate_orbital(
                self.diag.mol, self.diag.mo_coeff,
                orb_idx, self.grid_points, self.grid_shape,
            )
        return self._cache[orb_idx]

    # ---- rendering ----

    def _render_current(self):
        """Clear and re-render the scene for the current orbital."""
        filtered = self._filtered_orbitals()
        if not filtered:
            return

        self._current_idx = self._current_idx % len(filtered)
        list_idx = filtered[self._current_idx]
        orb_idx, orb_type = self._orbitals[list_idx]

        # Evaluate
        values = self._get_orbital_data(orb_idx)
        iso = compute_isosurface_level(values, self.iso_fraction)
        ang_label = orbital_angular_momentum_label(
            self.diag.mol, self.diag.mo_coeff, orb_idx,
        )

        # Rebuild scene
        self._plotter.clear()

        add_atoms(self._plotter, self.geometry.atoms, self.geometry.coords)
        add_bonds(self._plotter, self.geometry.atoms, self.geometry.coords)

        add_orbital_isosurface(
            self._plotter, values, self.x, self.y, self.z,
            isovalue=iso, orbital_type=orb_type,
            label=f"MO {orb_idx}",
        )

        # HUD text
        filt_name = self.TYPE_CYCLE[self._type_filter_idx]
        n_filt = len(filtered)
        energy_str = ""
        if self.diag.orbital_energies is not None and orb_idx < len(self.diag.orbital_energies):
            energy_str = f"  |  E = {self.diag.orbital_energies[orb_idx]:.4f} Ha"

        occ_str = ""
        if self.diag.natural_occupations is not None and orb_idx < len(self.diag.natural_occupations):
            occ_str = f"  |  occ = {self.diag.natural_occupations[orb_idx]:.4f}"

        title = (
            f"{self.geometry.name.upper()} ({self.geometry.formula})\n"
            f"MO {orb_idx}  [{orb_type}]  character: {ang_label}"
            f"{energy_str}{occ_str}\n"
            f"Filter: {filt_name} ({self._current_idx + 1}/{n_filt})  "
            f"|  N/P: next/prev  |  T: toggle filter  |  R: reset cam  |  Q: quit"
        )
        self._plotter.add_text(title, position="upper_left", font_size=10, color="white")

        self._plotter.render()

    # ---- keyboard callbacks ----

    def _next_orbital(self):
        self._current_idx += 1
        self._render_current()

    def _prev_orbital(self):
        self._current_idx -= 1
        filtered = self._filtered_orbitals()
        if filtered:
            self._current_idx = self._current_idx % len(filtered)
        self._render_current()

    def _toggle_filter(self):
        self._type_filter_idx = (self._type_filter_idx + 1) % len(self.TYPE_CYCLE)
        self._current_idx = 0
        self._render_current()

    def _reset_camera(self):
        self._plotter.reset_camera()
        self._plotter.render()

    # ---- public API ----

    def show(self):
        """Launch the interactive window."""
        self._plotter = create_plotter(
            title=f"Orbital Viewer — {self.geometry.name.upper()}",
        )

        self._plotter.add_key_event("n", lambda: self._next_orbital())
        self._plotter.add_key_event("p", lambda: self._prev_orbital())
        self._plotter.add_key_event("t", lambda: self._toggle_filter())
        self._plotter.add_key_event("r", lambda: self._reset_camera())

        # Start on the first active orbital if possible
        active_indices = self._filtered_orbitals()
        active_orbital_positions = [
            i for i, (_, t) in enumerate(self._orbitals) if t == "active"
        ]
        if active_orbital_positions:
            self._current_idx = active_orbital_positions[0]
            # Switch to "all" filter but point at first active
            for idx, (list_pos, _) in enumerate(
                [(j, self._orbitals[j]) for j in self._filtered_orbitals()]
            ):
                if list_pos == active_orbital_positions[0]:
                    self._current_idx = idx
                    break

        self._render_current()
        self._plotter.show()

    def screenshot(self, orbital_index: int, save_path: str) -> str:
        """Render a single orbital off-screen and save to file."""
        self._plotter = create_plotter(
            title=f"{self.geometry.name} MO {orbital_index}",
        )
        self._plotter.off_screen = True

        add_atoms(self._plotter, self.geometry.atoms, self.geometry.coords)
        add_bonds(self._plotter, self.geometry.atoms, self.geometry.coords)

        values = self._get_orbital_data(orbital_index)
        iso = compute_isosurface_level(values, self.iso_fraction)
        orb_type = self._orbitals[orbital_index][1] if orbital_index < len(self._orbitals) else "active"

        add_orbital_isosurface(
            self._plotter, values, self.x, self.y, self.z,
            isovalue=iso, orbital_type=orb_type,
        )

        self._plotter.show(screenshot=save_path, auto_close=True)
        return save_path


# ---------- convenience functions ----------

def view_molecule(
    name: str,
    basis: str = "cc-pvdz",
    resolution: float = 0.25,
):
    """One-liner: load molecule, run analysis, open viewer."""
    print(f"Loading {name}...")
    geom = get_geometry(name)
    print(f"Running orbital analysis ({geom.n_atoms} atoms, basis={basis})...")
    diag = run_orbital_analysis(geom, basis=basis)
    print(diag.summary())
    print("Launching interactive viewer...")
    viewer = OrbitalViewer(geom, diag, resolution=resolution)
    viewer.show()


# ---------- CLI ----------

def main():
    args = sys.argv[1:]

    if not args or args[0] in ("-h", "--help"):
        available = list_available()
        print("Usage: python -m framework.visualize_molecules.interactive_viewer <molecule>")
        print(f"\nAvailable molecules: {', '.join(available)}")
        print("\nOptions:")
        print("  --basis <name>       Basis set (default: cc-pvdz)")
        print("  --resolution <float> Grid spacing in Angstroms (default: 0.25)")
        sys.exit(0)

    mol_name = args[0]
    basis = "cc-pvdz"
    resolution = 0.25

    i = 1
    while i < len(args):
        if args[i] == "--basis" and i + 1 < len(args):
            basis = args[i + 1]
            i += 2
        elif args[i] == "--resolution" and i + 1 < len(args):
            resolution = float(args[i + 1])
            i += 2
        else:
            i += 1

    view_molecule(mol_name, basis=basis, resolution=resolution)


if __name__ == "__main__":
    main()
