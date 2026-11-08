# ABOUTME: Compares canonical and MP2 natural orbital choices on closed-shell molecular geometries.
# ABOUTME: Checks the mapped Hamiltonians against CASCI in the same electron and spin-projection sector.
import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
from pyscf import mp

from active_space_truncation.geometry import load_geometry_from_h5
from active_space_truncation.step1_orbitals import run_orbital_analysis
from active_space_truncation.step2_active_space import validate_active_space
from active_space_truncation.step3_hamiltonian import build_qubit_hamiltonian


def sector_ground_energy(hamiltonian, n_electrons):
    """Sum the stored Pauli words in the closed-shell determinant basis."""
    n = hamiltonian.n_qubits
    alpha = sum(1 << i for i in range(0, n, 2))
    beta = ((1 << n) - 1) ^ alpha
    basis = np.array([i for i in range(1 << n)
                      if (i & alpha).bit_count() == (i & beta).bit_count() == n_electrons // 2])
    indices = np.full(1 << n, -1, dtype=int)
    indices[basis] = np.arange(len(basis))
    parity = np.array([(-1) ** i.bit_count() for i in range(1 << n)])
    matrix = np.zeros((len(basis), len(basis)), dtype=complex)
    for coeff, word in zip(hamiltonian.coefficients, hamiltonian.pauli_strings):
        flip = int(''.join('1' if p in 'XY' else '0' for p in word), 2)
        sign = int(''.join('1' if p in 'YZ' else '0' for p in word), 2)
        destinations = indices[basis ^ flip]
        columns = np.flatnonzero(destinations >= 0)
        matrix[destinations[columns], columns] += coeff * 1j**word.count('Y') * parity[basis[columns] & sign]
    if not np.allclose(matrix, matrix.conj().T, atol=1e-11, rtol=0):
        raise ValueError("Mapped Hamiltonian is not Hermitian in the selected sector")
    return float(np.linalg.eigvalsh(matrix)[0])


def compare(path, orbital_cap):
    geometry = load_geometry_from_h5(str(path))
    if geometry.spin != 0:
        raise ValueError(f"{path} requires an open-shell reference")
    diagnostics = run_orbital_analysis(geometry, basis="sto-3g", run_basis_comparison=False,
                                       max_active_orbitals=orbital_cap)
    if not diagnostics.hf_converged:
        raise ValueError(f"HF failed for {path}")
    canonical = diagnostics.mf.mo_coeff
    occupations, rotation = np.linalg.eigh(mp.MP2(diagnostics.mf).run().make_rdm1())
    if not np.allclose(occupations[::-1], diagnostics.natural_occupations, rtol=0, atol=1e-10):
        raise ValueError(f"Natural occupation ranks disagree for {path}")
    natural = canonical @ rotation[:, ::-1]
    results = {}
    for name, orbitals in (("canonical", canonical), ("natural", natural)):
        active = validate_active_space(geometry, diagnostics, orbital_coefficients=orbitals)
        if not active.casci_converged:
            raise ValueError(f"CASCI failed for {path} in {name} space")
        ham = build_qubit_hamiltonian(geometry, active, diagnostics, orbital_coefficients=orbitals).qubit_hamiltonian
        energy = sector_ground_energy(ham, active.n_active_electrons)
        if abs(energy - active.casci_energy) > 1e-8:
            raise ValueError(f"Sector ground energy disagrees with CASCI for {path} in {name} space")
        digest = hashlib.sha256()
        for word, coefficient in zip(ham.pauli_strings, ham.coefficients):
            digest.update(word.encode('ascii') + np.asarray(coefficient, dtype='<f8').tobytes())
        results[name] = dict(casci_ha=active.casci_energy, sector_energy_ha=energy,
                             qubits=ham.n_qubits, terms=ham.n_terms, hamiltonian_sha256=digest.hexdigest())
    return dict(molecule=path.stem, geometry_sha256=hashlib.sha256(path.read_bytes()).hexdigest(),
                basis="sto-3g", orbital_cap=orbital_cap, hf_ha=diagnostics.hf_energy,
                active_indices=diagnostics.proposed_active_indices, electrons=diagnostics.proposed_n_active_electrons,
                natural_minus_canonical_mha=1000*(results["natural"]["casci_ha"] - results["canonical"]["casci_ha"]),
                models=results)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("geometries", type=Path, nargs="+")
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--orbital-caps", type=int, nargs="+", choices=range(1, 7), default=[6])
    args = parser.parse_args()
    args.output.write_text(json.dumps([compare(path, cap) for path in args.geometries for cap in args.orbital_caps], indent=2) + "\n")
