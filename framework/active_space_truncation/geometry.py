# ABOUTME: Loads atomic coordinates and molecule identifiers from QMProt geometry datasets.
# ABOUTME: Reads plain or gzip-compressed HDF5 inputs used by the active-space pipeline.
"""Geometry loading for molecular systems.

Supports loading from QMProt H5 files (the primary path)
and hardcoded fallbacks for testing.
"""

from dataclasses import dataclass
import gzip
import shutil
import sys
import tempfile
from contextlib import contextmanager
from pathlib import Path
from typing import Iterator, List, Tuple

import numpy as np

# Align with framework config (DATASETS_DIR env / default) for molecule discovery
_FW = Path(__file__).resolve().parent.parent
if str(_FW) not in sys.path:
    sys.path.insert(0, str(_FW))
from config import DATASETS_DIR  # noqa: E402


@dataclass
class MoleculeGeometry:
    """Molecular geometry specification."""
    atoms: List[str]
    coords: np.ndarray       # (n_atoms, 3) in Angstroms
    charge: int = 0
    spin: int = 0
    name: str = ""
    formula: str = ""
    abbreviation: str = ""

    ATOMIC_NUMBERS = {
        "H": 1, "He": 2, "Li": 3, "Be": 4, "B": 5, "C": 6,
        "N": 7, "O": 8, "F": 9, "Ne": 10,
    }

    @property
    def n_atoms(self) -> int:
        return len(self.atoms)

    @property
    def n_electrons(self) -> int:
        from pyscf.data.elements import charge
        return sum(charge(atom) for atom in self.atoms) - self.charge

    @property
    def multiplicity(self) -> int:
        return self.spin + 1

    def to_pyscf_atom_list(self) -> List[Tuple[str, Tuple[float, float, float]]]:
        return [(atom, tuple(coord)) for atom, coord in zip(self.atoms, self.coords)]

    def to_openfermion_geometry(self) -> List[Tuple[str, Tuple[float, float, float]]]:
        return self.to_pyscf_atom_list()


@contextmanager
def _materialized_h5_path(path: str) -> Iterator[Path]:
    """Yield an HDF5 path, temporarily decompressing ``.h5.gz`` inputs."""
    source = Path(path)
    if source.suffix != ".gz":
        yield source
        return

    with tempfile.TemporaryDirectory(prefix="qmprot_h5_") as temp_dir:
        materialized = Path(temp_dir) / source.name[:-3]
        with gzip.open(source, "rb") as compressed, materialized.open("wb") as output:
            shutil.copyfileobj(compressed, output)
        yield materialized


def load_geometry_from_h5(path: str) -> MoleculeGeometry:
    """Load molecular geometry from a QMProt-format H5 file.

    QMProt H5 structure:
        /symbols/{0,1,...}  - element symbols (bytes)
        /coordinates/{0,1,...}/{0,1,2}  - x,y,z per atom (floats)
        /name, /mf, /charge, /spin, /n_atoms  - metadata
    """
    import h5py

    with _materialized_h5_path(path) as h5_path:
        with h5py.File(h5_path, "r") as f:
            n_atoms = int(f["n_atoms"][()])
            fallback_name = h5_path.stem
            name = f["name"][()].decode() if "name" in f else fallback_name
            formula = f["mf"][()].decode() if "mf" in f else ""
            charge = int(f["charge"][()]) if "charge" in f else 0
            spin = int(f["spin"][()]) if "spin" in f else 0

            atoms = []
            coords = []
            for i in range(n_atoms):
                sym = f["symbols"][str(i)][()]
                atoms.append(sym.decode() if isinstance(sym, bytes) else str(sym))

                x = float(f["coordinates"][str(i)]["0"][()])
                y = float(f["coordinates"][str(i)]["1"][()])
                z = float(f["coordinates"][str(i)]["2"][()])
                coords.append([x, y, z])

    source_name = Path(path).name
    if source_name.endswith(".gz"):
        source_name = source_name[:-3]

    return MoleculeGeometry(
        atoms=atoms,
        coords=np.array(coords),
        charge=charge,
        spin=spin,
        name=name,
        formula=formula,
        abbreviation=Path(source_name).stem,
    )


# --- Built-in molecule registry (loads from H5 when available) ---

# Core electrons per atom (for freeze-core)
CORE_ELECTRONS = {
    "H": 0, "He": 0,
    "Li": 2, "Be": 2, "B": 2, "C": 2, "N": 2, "O": 2, "F": 2, "Ne": 2,
    "S": 10,
}

# All amino acid abbreviations with H5 datasets
AMINO_ACIDS = ["ala", "arg", "asn", "asp", "cys", "gly", "pro", "ser", "tyr", "val"]


def get_geometry(name: str) -> MoleculeGeometry:
    """Load a molecule by abbreviation from the datasets/ H5 files."""
    key = name.lower()
    h5_path = DATASETS_DIR / key / f"{key}.h5"
    if h5_path.exists():
        return load_geometry_from_h5(str(h5_path))
    raise ValueError(
        f"Unknown molecule '{name}'. Available: {list_available()}"
    )


def list_available() -> List[str]:
    """List all molecules with H5 datasets."""
    available = []
    if DATASETS_DIR.exists():
        for d in sorted(DATASETS_DIR.iterdir()):
            if d.is_dir() and (d / f"{d.name}.h5").exists():
                available.append(d.name)
    return available
