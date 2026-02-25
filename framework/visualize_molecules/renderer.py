"""PyVista-based 3D renderer for molecules and orbital isosurfaces.

Handles atom spheres, bond cylinders, and translucent orbital lobes.
All rendering is done through a single PyVista Plotter instance.
"""

import numpy as np
import pyvista as pv
from typing import List, Tuple, Optional, Dict

# ---------- visual constants ----------

ATOM_COLORS: Dict[str, str] = {
    "H": "#FFFFFF", "C": "#404040", "N": "#3050F8", "O": "#FF0D0D",
    "F": "#90E050", "S": "#FFFF30", "P": "#FF8000", "Li": "#CC80FF",
}

ATOM_RADII: Dict[str, float] = {
    "H": 0.30, "C": 0.65, "N": 0.60, "O": 0.55,
    "F": 0.50, "S": 0.90, "P": 0.90, "Li": 1.10,
}

BOND_CUTOFF = 1.85  # Angstroms
BOND_RADIUS = 0.08

# Orbital lobe colors: positive / negative phase
ORBITAL_COLORS = {
    "core":    ("#d32f2f", "#b71c1c"),   # red tones
    "active":  ("#1b5e20", "#4caf50"),   # green tones
    "virtual": ("#616161", "#9e9e9e"),   # gray tones
}

ORBITAL_OPACITY = {
    "core": 0.20,
    "active": 0.35,
    "virtual": 0.15,
}


# ---------- molecule rendering ----------

def add_atoms(
    plotter: pv.Plotter,
    atoms: List[str],
    coords: np.ndarray,
    label: bool = True,
) -> None:
    """Add atom spheres to the plotter."""
    for i, (symbol, pos) in enumerate(zip(atoms, coords)):
        radius = ATOM_RADII.get(symbol, 0.5)
        color = ATOM_COLORS.get(symbol, "#808080")
        sphere = pv.Sphere(radius=radius, center=pos, theta_resolution=24, phi_resolution=24)
        plotter.add_mesh(sphere, color=color, opacity=0.45, smooth_shading=True)
        if label:
            plotter.add_point_labels(
                [pos + np.array([0, 0, radius + 0.15])],
                [symbol],
                font_size=14,
                text_color="white",
                bold=True,
                shape=None,
                render_points_as_spheres=False,
                always_visible=True,
            )


def add_bonds(
    plotter: pv.Plotter,
    atoms: List[str],
    coords: np.ndarray,
    cutoff: float = BOND_CUTOFF,
) -> None:
    """Draw bond cylinders between nearby atoms."""
    n = len(atoms)
    for i in range(n):
        for j in range(i + 1, n):
            dist = np.linalg.norm(coords[i] - coords[j])
            if dist < cutoff:
                direction = coords[j] - coords[i]
                center = (coords[i] + coords[j]) / 2.0
                cyl = pv.Cylinder(
                    center=center,
                    direction=direction,
                    radius=BOND_RADIUS,
                    height=dist,
                    resolution=16,
                )
                plotter.add_mesh(cyl, color="#888888", smooth_shading=True)


# ---------- orbital isosurface rendering ----------

def add_orbital_isosurface(
    plotter: pv.Plotter,
    orbital_values: np.ndarray,
    x: np.ndarray,
    y: np.ndarray,
    z: np.ndarray,
    isovalue: float,
    orbital_type: str = "active",
    label: str = "",
) -> None:
    """Add positive and negative isosurface lobes for one orbital.

    Parameters
    ----------
    orbital_values : (nx, ny, nz) volumetric data
    x, y, z : 1D grid axes (Angstroms)
    isovalue : threshold for the isosurface
    orbital_type : "core", "active", or "virtual" (determines color/opacity)
    label : text annotation (shown in legend)
    """
    grid = pv.ImageData(
        dimensions=orbital_values.shape,
        spacing=(x[1] - x[0], y[1] - y[0], z[1] - z[0]),
        origin=(x[0], y[0], z[0]),
    )
    grid.point_data["orbital"] = orbital_values.ravel(order="F")

    color_pos, color_neg = ORBITAL_COLORS.get(orbital_type, ORBITAL_COLORS["active"])
    opacity = ORBITAL_OPACITY.get(orbital_type, 0.3)

    # Positive lobe
    try:
        iso_pos = grid.contour([isovalue], scalars="orbital")
        if iso_pos.n_points > 0:
            plotter.add_mesh(
                iso_pos, color=color_pos, opacity=opacity,
                smooth_shading=True, label=f"+{label}" if label else None,
            )
    except Exception:
        pass

    # Negative lobe
    try:
        iso_neg = grid.contour([-isovalue], scalars="orbital")
        if iso_neg.n_points > 0:
            plotter.add_mesh(
                iso_neg, color=color_neg, opacity=opacity,
                smooth_shading=True, label=f"-{label}" if label else None,
            )
    except Exception:
        pass


# ---------- plotter factory ----------

def create_plotter(
    title: str = "Orbital Viewer",
    bg_color: str = "#1a1a2e",
    window_size: Tuple[int, int] = (1400, 900),
) -> pv.Plotter:
    """Create a styled PyVista plotter with sensible defaults."""
    plotter = pv.Plotter(window_size=window_size, title=title)
    plotter.set_background(bg_color)
    plotter.enable_anti_aliasing("ssaa")
    return plotter
