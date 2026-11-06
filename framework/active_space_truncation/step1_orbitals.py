"""Step 1: Hartree-Fock + MP2 orbital analysis and diagnostics.

Runs HF and MP2, computes natural orbital occupations,
and proposes an active space capped at MAX_ACTIVE_ORBITALS.
"""

from dataclasses import dataclass, field
from typing import List, Optional
import numpy as np

from .geometry import MoleculeGeometry, CORE_ELECTRONS

# Active space config
MAX_ACTIVE_ORBITALS = 6
OCCUPATION_LOWER = 0.02
OCCUPATION_UPPER = 1.98
HARTREE_TO_EV = 27.211386245988


@dataclass
class OrbitalDiagnostics:
    """Results from orbital analysis."""
    hf_energy: float = 0.0
    hf_converged: bool = False
    n_basis_functions: int = 0
    n_molecular_orbitals: int = 0
    orbital_energies: Optional[np.ndarray] = None

    mp2_energy: float = 0.0
    mp2_correlation: float = 0.0
    natural_occupations: Optional[np.ndarray] = None

    homo_lumo_gap_ev: float = 0.0
    dipole_moment_debye: float = 0.0

    n_core_orbitals: int = 0
    core_orbital_indices: List[int] = field(default_factory=list)
    proposed_active_indices: List[int] = field(default_factory=list)
    proposed_n_active_electrons: int = 0
    proposed_n_active_orbitals: int = 0

    hf_energy_minimal: Optional[float] = None

    # PySCF objects (for later steps)
    mol: object = None
    mf: object = None
    mo_coeff: Optional[np.ndarray] = None

    @property
    def n_qubits_proposed(self) -> int:
        return 2 * self.proposed_n_active_orbitals

    def summary(self) -> str:
        lines = [
            "=" * 60,
            "ORBITAL ANALYSIS DIAGNOSTICS",
            "=" * 60,
            f"Basis functions:       {self.n_basis_functions}",
            f"Molecular orbitals:    {self.n_molecular_orbitals}",
            "",
            f"HF energy:             {self.hf_energy:.10f} Ha",
            f"HF converged:          {self.hf_converged}",
            f"MP2 total energy:      {self.mp2_energy:.10f} Ha",
            f"MP2 correlation:       {self.mp2_correlation:.10f} Ha",
            "",
            f"HOMO-LUMO gap:         {self.homo_lumo_gap_ev:.4f} eV",
            f"Dipole moment:         {self.dipole_moment_debye:.4f} Debye",
            "",
            f"Core orbitals frozen:  {self.n_core_orbitals}",
            f"Proposed active space: ({self.proposed_n_active_electrons}e, {self.proposed_n_active_orbitals}o)",
            f"Qubits needed:         {self.n_qubits_proposed}",
        ]
        if self.natural_occupations is not None:
            lines.append("")
            lines.append("Natural orbital occupations (active region):")
            for i in self.proposed_active_indices:
                if i < len(self.natural_occupations):
                    lines.append(f"  MO {i:3d}: {self.natural_occupations[i]:.6f}")
        if self.hf_energy_minimal is not None:
            lines.append("")
            lines.append(f"STO-3G HF energy:      {self.hf_energy_minimal:.10f} Ha")
        lines.append("=" * 60)
        return "\n".join(lines)


def run_orbital_analysis(
    geometry: MoleculeGeometry,
    basis: str = "cc-pvdz",
    run_basis_comparison: bool = False,
) -> OrbitalDiagnostics:
    """Run HF + MP2 and propose active space (capped at MAX_ACTIVE_ORBITALS)."""
    from pyscf import gto, scf, mp as pyscf_mp

    diag = OrbitalDiagnostics()

    mol = gto.Mole()
    mol.atom = geometry.to_pyscf_atom_list()
    mol.basis = basis
    mol.charge = geometry.charge
    mol.spin = geometry.spin
    mol.build()

    diag.n_basis_functions = mol.nao_nr()
    diag.n_molecular_orbitals = mol.nao_nr()
    diag.mol = mol

    # RHF
    mf = scf.RHF(mol)
    mf.kernel()
    diag.hf_energy = float(mf.e_tot)
    diag.hf_converged = bool(mf.converged)
    diag.orbital_energies = mf.mo_energy
    diag.mo_coeff = mf.mo_coeff
    diag.mf = mf

    if not mf.converged:
        print("WARNING: HF did not converge!")

    # HOMO-LUMO gap
    n_occ = mol.nelectron // 2
    diag.homo_lumo_gap_ev = float((mf.mo_energy[n_occ] - mf.mo_energy[n_occ - 1]) * HARTREE_TO_EV)

    # Dipole
    dip = mf.dip_moment(verbose=0)
    diag.dipole_moment_debye = float(np.linalg.norm(dip))

    # MP2
    mp2_obj = pyscf_mp.MP2(mf)
    mp2_obj.kernel()
    diag.mp2_energy = float(mp2_obj.e_tot)
    diag.mp2_correlation = float(mp2_obj.e_corr)

    # Natural orbital occupations.
    # Open-shell MP2 can return spin-resolved blocks; collapse to a spatial 1-RDM first.
    rdm1_mo = mp2_obj.make_rdm1()
    if isinstance(rdm1_mo, (tuple, list)) and len(rdm1_mo) == 2:
        rdm1_spatial = np.asarray(rdm1_mo[0]) + np.asarray(rdm1_mo[1])
    else:
        rdm1_arr = np.asarray(rdm1_mo)
        if rdm1_arr.ndim == 3 and rdm1_arr.shape[0] == 2:
            rdm1_spatial = rdm1_arr[0] + rdm1_arr[1]
        elif rdm1_arr.ndim == 2:
            rdm1_spatial = rdm1_arr
        else:
            raise ValueError(f"Unexpected MP2 RDM1 shape: {rdm1_arr.shape}")

    nat_occ = np.linalg.eigvalsh(rdm1_spatial)
    nat_occ = np.asarray(nat_occ, dtype=float)[::-1]  # descending
    diag.natural_occupations = nat_occ

    # Freeze core
    n_core = sum(CORE_ELECTRONS.get(a, 0) // 2 for a in geometry.atoms)
    diag.n_core_orbitals = n_core
    diag.core_orbital_indices = list(range(n_core))

    # Active space: rank by correlation importance, cap at MAX_ACTIVE_ORBITALS
    candidates = list(range(n_core, len(nat_occ)))
    #candidates = [i for i in range(n_core, len(nat_occ))
    #              if OCCUPATION_LOWER < nat_occ[i] < OCCUPATION_UPPER]

    #if not candidates:
    #    start = max(n_core, n_occ - 3)
    #    end = min(len(nat_occ), n_occ + 3)
    #    candidates = list(range(start, end))

     # Score: distance from nearest integer occupation (2.0 or 0.0)
    def _score(idx):
        return min(abs(nat_occ[idx] - 2.0), abs(nat_occ[idx] - 0.0))
    candidates.sort(key=_score, reverse=True)
    candidates = sorted(candidates[:MAX_ACTIVE_ORBITALS])
    
    diag.proposed_active_indices = candidates
    diag.proposed_n_active_orbitals = len(candidates)

    # Estimate active electrons from natural occupations, then enforce parity with
    # the molecule electron count so CASCI can assign a valid closed-shell core.
    est_active_electrons = int(np.rint(float(np.sum(nat_occ[candidates])))) if candidates else 0
    est_active_electrons = int(np.clip(est_active_electrons, 0, 2 * len(candidates)))
    target_parity = mol.nelectron % 2
    if est_active_electrons % 2 != target_parity:
        if est_active_electrons < 2 * len(candidates):
            est_active_electrons += 1
        elif est_active_electrons > 0:
            est_active_electrons -= 1
    diag.proposed_n_active_electrons = est_active_electrons

    # Optional basis comparison
    if run_basis_comparison:
        mol_min = gto.Mole()
        mol_min.atom = geometry.to_pyscf_atom_list()
        mol_min.basis = "sto-3g"
        mol_min.charge = geometry.charge
        mol_min.spin = geometry.spin
        mol_min.build()
        mf_min = scf.RHF(mol_min)
        mf_min.kernel()
        diag.hf_energy_minimal = float(mf_min.e_tot)

    return diag
