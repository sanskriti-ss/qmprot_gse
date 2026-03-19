"""Step 3: Build qubit Hamiltonian using OpenFermion.

Produces a QubitHamiltonian compatible with the existing VQE framework
(framework/core/hamiltonian_loader.py).
"""

from dataclasses import dataclass, field
from typing import Dict, List

from .geometry import MoleculeGeometry
from .step1_orbitals import OrbitalDiagnostics
from .step2_active_space import ActiveSpaceResult


@dataclass
class PipelineHamiltonian:
    """Result of Hamiltonian construction."""
    qubit_hamiltonian: object = None   # framework's QubitHamiltonian
    openfermion_qubit_op: object = None
    n_qubits: int = 0
    n_terms: int = 0
    terms: Dict[str, complex] = field(default_factory=dict)
    core_energy: float = 0.0          # frozen core + nuclear repulsion (identity term)

    def summary(self) -> str:
        lines = [
            "=" * 60,
            "QUBIT HAMILTONIAN",
            "=" * 60,
            f"Number of qubits:  {self.n_qubits}",
            f"Number of terms:   {self.n_terms}",
            f"Core energy:       {self.core_energy:.10f} Ha  (frozen core + nuclear repulsion)",
        ]
        if self.n_terms <= 20:
            lines.append("\nTerms (showing all):")
            for pauli, coeff in sorted(self.terms.items(), key=lambda x: -abs(x[1])):
                lines.append(f"  {coeff:+.8f}  {pauli}")
        else:
            lines.append("\nTop 10 terms by magnitude:")
            for pauli, coeff in sorted(self.terms.items(), key=lambda x: -abs(x[1]))[:10]:
                lines.append(f"  {coeff:+.8f}  {pauli}")
            lines.append(f"  ... and {self.n_terms - 10} more terms")
        lines.append("=" * 60)
        return "\n".join(lines)


def build_qubit_hamiltonian(
    geometry: MoleculeGeometry,
    active_space: ActiveSpaceResult,
    diagnostics: OrbitalDiagnostics,
) -> PipelineHamiltonian:
    """Build qubit Hamiltonian via PySCF active-space integrals + OpenFermion.

    Uses PySCF's CASCI ao2mo.full() to compute only (ncas)^4 ERIs, avoiding
    the (n_basis)^4 allocation that OOMs for large molecules like ARG (238^4 = 23.9 GiB).
    Spin-orbital expansion follows OpenFermion's spinorb_from_spatial() exactly.
    """
    from pyscf import mcscf
    from pyscf import ao2mo as pyscf_ao2mo
    from openfermion import InteractionOperator, jordan_wigner
    import numpy as np
    import sys, os
    sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    from core.hamiltonian_loader import Molecule, QubitHamiltonian

    result = PipelineHamiltonian()

    mf = diagnostics.mf
    ncas = active_space.n_active_orbitals
    nelecas = active_space.n_active_electrons

    # Set up CASCI with the same active space as step 2.
    mc = mcscf.CASCI(mf, ncas, nelecas)
    # PySCF sort_mo expects 1-based orbital indices.
    cas_list = [i + 1 for i in diagnostics.proposed_active_indices]
    mo = mcscf.sort_mo(mc, mf.mo_coeff, cas_list)

    # 1-electron integrals for active space.
    # h1eff includes the frozen-core Fock contribution;
    # e_core = nuclear repulsion + frozen-core electron energy.
    h1eff, e_core = mc.h1e_for_cas(mo)        # shape (ncas, ncas)

    # 2-electron integrals for active space only: (ncas^4) NOT (n_basis^4).
    # ao2mo.full returns chemist notation: (ij|kl)
    mo_cas = mo[:, mc.ncore:mc.ncore + ncas]  # (n_ao, ncas)
    h2e_chem = pyscf_ao2mo.full(
        mf.mol, mo_cas, compact=False
    ).reshape(ncas, ncas, ncas, ncas)

    # FINALLY FIX: The energies were wrong because we were feeding (pr|qs)_chem where (ps|qr)_chem was expected: a consistent index swap that shifts all 2-body matrix elements by ~20 Ha.
    h2e_phys = h2e_chem.transpose(0, 2, 3, 1) # shape (ncas, ncas, ncas, ncas)

    # Expand to spin-orbital basis using the EXACT convention of
    # OpenFermion's spinorb_from_spatial() (openfermion/chem/molecular_data.py).
    n_so = 2 * ncas
    h1_so = np.zeros((n_so, n_so))
    h2_so = np.zeros((n_so, n_so, n_so, n_so))

    for p in range(ncas):
        for q in range(ncas):
            h1_so[2 * p,     2 * q    ] = h1eff[p, q]   # alpha -> alpha
            h1_so[2 * p + 1, 2 * q + 1] = h1eff[p, q]  # beta  -> beta
            for r in range(ncas):
                for s in range(ncas):
                    val = h2e_phys[p, q, r, s]
                    # Mixed spin 
                    h2_so[2*p,   2*q+1, 2*r+1, 2*s  ] = val
                    h2_so[2*p+1, 2*q,   2*r,   2*s+1] = val
                    # Same spin 
                    h2_so[2*p,   2*q,   2*r,   2*s  ] = val
                    h2_so[2*p+1, 2*q+1, 2*r+1, 2*s+1] = val

    # Factor 1/2 on two-body matches get_molecular_hamiltonian() convention.
    molecular_hamiltonian = InteractionOperator(float(e_core), h1_so, 0.5 * h2_so)
    qubit_op = jordan_wigner(molecular_hamiltonian)

    # Extract terms
    n_qubits = 2 * active_space.n_active_orbitals
    terms = {}
    for term, coeff in qubit_op.terms.items():
        if abs(coeff) < 1e-12:
            continue
        pauli_str = _term_to_pauli_string(term, n_qubits)
        terms[pauli_str] = complex(coeff)

    # Extract core energy: the identity term coefficient contains
    # nuclear repulsion + frozen core electron energy.
    identity_key = "I" * n_qubits
    core_energy = terms.get(identity_key, 0.0).real

    result.openfermion_qubit_op = qubit_op
    result.n_qubits = n_qubits
    result.n_terms = len(terms)
    result.terms = terms
    result.core_energy = core_energy

    # Build framework-compatible QubitHamiltonian
    coefficients = np.array([c.real for c in terms.values()])
    pauli_strings = list(terms.keys())

    # Reference energy = CASCI energy (the best classical answer for this
    # active space).  VQE should approach this; error = VQE - CASCI.
    molecule = Molecule(
        abbreviation=geometry.name[:3],
        name=geometry.name,
        n_qubits=n_qubits,
        n_coefficients=len(terms),
        reference_energy=active_space.casci_energy,
        hamiltonian_file=f"pipeline_{geometry.name}",
        n_electrons=active_space.n_active_electrons,
        n_orbitals=active_space.n_active_orbitals,
        charge=geometry.charge,
        spin=geometry.spin,
        basis=diagnostics.mol.basis if diagnostics.mol else "cc-pvdz",
        molecular_formula=geometry.formula,
        core_energy=core_energy,
    )

    qh = QubitHamiltonian(
        molecule=molecule,
        coefficients=coefficients,
        pauli_strings=pauli_strings,
        n_qubits=n_qubits,
        n_terms=len(terms),
    )
    result.qubit_hamiltonian = qh

    return result


def _term_to_pauli_string(term: tuple, n_qubits: int) -> str:
    pauli_list = ["I"] * n_qubits
    for qubit_idx, pauli in term:
        if qubit_idx < n_qubits:
            pauli_list[qubit_idx] = pauli
    return "".join(pauli_list)
