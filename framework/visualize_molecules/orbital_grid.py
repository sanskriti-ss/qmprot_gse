"""Orbital wavefunction evaluation on 3D grids.

Evaluates molecular orbitals on a uniform grid using PySCF basis
functions, producing volumetric data suitable for isosurface rendering.
"""

import numpy as np
from typing import Tuple, Optional


def build_grid(
    coords: np.ndarray,
    padding: float = 4.0,
    resolution: float = 0.2,
) -> Tuple[np.ndarray, Tuple[int, int, int], np.ndarray, np.ndarray, np.ndarray]:
    """Create a uniform 3D grid enclosing the molecule.

    Parameters
    ----------
    coords : (n_atoms, 3) array in Angstroms
    padding : extra space around the molecule in each direction
    resolution : grid spacing in Angstroms

    Returns
    -------
    grid_points : (N, 3) flat array of grid coordinates (Bohr)
    grid_shape : (nx, ny, nz)
    x, y, z : 1D arrays along each axis (Angstroms, for PyVista)
    """
    ANG_TO_BOHR = 1.8897259886

    mins = coords.min(axis=0) - padding
    maxs = coords.max(axis=0) + padding

    x = np.arange(mins[0], maxs[0] + resolution, resolution)
    y = np.arange(mins[1], maxs[1] + resolution, resolution)
    z = np.arange(mins[2], maxs[2] + resolution, resolution)

    grid_shape = (len(x), len(y), len(z))

    # Meshgrid for PySCF evaluation (needs Bohr)
    xx, yy, zz = np.meshgrid(x, y, z, indexing="ij")
    grid_points = np.column_stack([
        xx.ravel() * ANG_TO_BOHR,
        yy.ravel() * ANG_TO_BOHR,
        zz.ravel() * ANG_TO_BOHR,
    ])

    return grid_points, grid_shape, x, y, z


def evaluate_orbital(
    mol,
    mo_coeff: np.ndarray,
    orbital_index: int,
    grid_points: np.ndarray,
    grid_shape: Tuple[int, int, int],
) -> np.ndarray:
    """Evaluate a single molecular orbital on the grid.

    Parameters
    ----------
    mol : PySCF Mole object
    mo_coeff : (n_ao, n_mo) MO coefficient matrix
    orbital_index : which MO to evaluate
    grid_points : (N, 3) in Bohr
    grid_shape : (nx, ny, nz)

    Returns
    -------
    orbital_values : (nx, ny, nz) array
    """
    ao_values = mol.eval_gto("GTOval_sph", grid_points)  # (N, n_ao)
    mo_values = ao_values @ mo_coeff[:, orbital_index]     # (N,)
    return mo_values.reshape(grid_shape)


def compute_isosurface_level(
    orbital_values: np.ndarray,
    fraction: float = 0.03,
) -> float:
    """Pick an isovalue that encloses `fraction` of the max |psi|.

    A good default is ~3% of the peak amplitude, which captures
    the visually meaningful orbital lobes without noise.
    """
    peak = np.abs(orbital_values).max()
    if peak < 1e-12:
        return 1e-6
    return fraction * peak


def orbital_angular_momentum_label(
    mol, mo_coeff: np.ndarray, orbital_index: int,
) -> str:
    """Heuristic label for the dominant angular momentum character.

    Inspects AO contributions to classify as s / p / d / mixed.
    """
    ao_labels = mol.ao_labels(fmt=False)
    coeffs = mo_coeff[:, orbital_index]
    weights = {"s": 0.0, "p": 0.0, "d": 0.0, "f": 0.0}

    for i, (_, _, shell_type, _) in enumerate(ao_labels):
        key = shell_type[0].lower()
        if key in weights:
            weights[key] += coeffs[i] ** 2

    total = sum(weights.values())
    if total < 1e-12:
        return "?"

    dominant = max(weights, key=weights.get)
    frac = weights[dominant] / total
    return dominant if frac > 0.5 else "mixed"
