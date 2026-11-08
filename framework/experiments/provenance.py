# ABOUTME: Saves ordered molecular Hamiltonians, reference states and source code with experiment results.
# ABOUTME: Restores the exact solver input independently of orbital selection and subspace search.
from dataclasses import asdict
import hashlib
from importlib.metadata import PackageNotFoundError, version
from io import BytesIO
import json
from pathlib import Path
import platform
import subprocess
import sys
from zipfile import ZipFile, ZipInfo, ZIP_DEFLATED

import numpy as np

from core.hamiltonian_loader import Molecule, QubitHamiltonian


def framework_source():
    root = Path(__file__).resolve().parents[1]
    source = [root / "config.py"]
    for folder in ("core", "algorithms", "experiments", "active_space_truncation",
                   "contextual_subspace", "campaigns", "visualize_molecules"):
        source.extend(sorted((root / folder).glob("*.py")))
    archive = BytesIO()
    with ZipFile(archive, "w", compression=ZIP_DEFLATED) as output:
        for path in source:
            entry = ZipInfo(path.relative_to(root).as_posix())
            output.writestr(entry, path.read_bytes(), compress_type=ZIP_DEFLATED)
        driver = root / "run_barren_plateau_part_b_all_molecules.py"
        if not driver.is_file():
            driver = root.parent / driver.name
        if driver.is_file():
            output.writestr(ZipInfo("drivers/" + driver.name), driver.read_bytes(), compress_type=ZIP_DEFLATED)
    return archive.getvalue()


def record_runtime(directory):
    directory = Path(directory)
    directory.mkdir(parents=True, exist_ok=True)
    root = Path(__file__).resolve().parents[1]
    archive = directory / "framework-source.zip"
    archive.write_bytes(framework_source())
    revision = subprocess.run(["git", "-C", str(root), "rev-parse", "HEAD"], capture_output=True,
                              text=True) if (root.parent / ".git").exists() else None
    packages = {}
    for name in ("numpy", "scipy", "pyscf", "openfermion", "pennylane", "pennylane-lightning",
                 "autoray", "h5py", "scikit-optimize", "scikit-learn", "matplotlib", "pandas", "seaborn"):
        try:
            packages[name] = version(name)
        except PackageNotFoundError:
            packages[name] = None
    (directory / "runtime.json").write_text(json.dumps(dict(
        git_commit=revision.stdout.strip() if revision and revision.returncode == 0 else None,
        source_sha256=hashlib.sha256(archive.read_bytes()).hexdigest(), packages=packages,
        python=sys.version, platform=platform.platform(), argv=sys.argv,
    ), indent=2) + "\n")


def save_problem(loaded, directory):
    directory = Path(directory)
    directory.mkdir(parents=True, exist_ok=True)
    ham = loaded.hamiltonian
    state = np.asarray(ham.cs_initial_state) if ham.cs_initial_state is not None else np.array([], dtype=complex)
    path = directory / f"{loaded.molecule}.npz"
    np.savez_compressed(path, dataset=loaded.molecule, basis=loaded.basis,
                        preparation=json.dumps(loaded.preparation), cs_metadata=json.dumps(loaded.cs_metadata),
                        coefficients=ham.coefficients, pauli_strings=np.asarray(ham.pauli_strings),
                        initial_state=state, molecule=json.dumps(asdict(ham.molecule)),
                        n_qubits=ham.n_qubits, n_qubits_active=loaded.n_qubits_active,
                        cs_reduced=loaded.cs_reduced, molecular_hf=float(loaded.hf_energy), casci=float(loaded.casci_energy),
                        encoded_reference_energy=(float(loaded.encoded_reference_energy)
                                                  if loaded.encoded_reference_energy is not None else np.nan))
    return path


def load_problem(path):
    """Restore the mapped operator and the state the solvers prepare."""
    from experiments._common import LoadedHamiltonian

    with np.load(path, allow_pickle=False) as data:
        molecule = Molecule(**json.loads(str(data["molecule"])))
        coefficients = data["coefficients"]
        state = data["initial_state"]
        ham = QubitHamiltonian(molecule=molecule, coefficients=coefficients, pauli_strings=data["pauli_strings"].tolist(),
                                n_qubits=int(data["n_qubits"]), n_terms=len(coefficients),
                                cs_initial_state=state if state.size else None)
        return LoadedHamiltonian(hamiltonian=ham, molecule=str(data["dataset"]), basis=str(data["basis"]),
                                 n_qubits_active=int(data["n_qubits_active"]), n_qubits_final=ham.n_qubits,
                                 cs_reduced=bool(data["cs_reduced"]), cs_metadata=json.loads(str(data["cs_metadata"])),
                                 hf_energy=float(data["molecular_hf"]), casci_energy=float(data["casci"]),
                                 encoded_reference_energy=(float(data["encoded_reference_energy"])
                                     if "encoded_reference_energy" in data
                                     and np.isfinite(data["encoded_reference_energy"]) else None),
                                 preparation=json.loads(str(data["preparation"])))
