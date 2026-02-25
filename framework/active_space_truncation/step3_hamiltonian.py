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

    def summary(self) -> str:
        lines = [
            "=" * 60,
            "QUBIT HAMILTONIAN",
            "=" * 60,
            f"Number of qubits:  {self.n_qubits}",
            f"Number of terms:   {self.n_terms}",
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
    """Build qubit Hamiltonian via OpenFermion + PySCF.

    Follows the same pattern as generate_gln_hamiltonian_active_space.py.
    Output is compatible with the existing framework's QubitHamiltonian.
    """
    from openfermion import MolecularData, jordan_wigner
    from openfermionpyscf import run_pyscf
    import numpy as np
    import sys, os
    sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    from core.hamiltonian_loader import Molecule, QubitHamiltonian

    result = PipelineHamiltonian()

    mol_data = MolecularData(
        geometry=geometry.to_openfermion_geometry(),
        basis=diagnostics.mol.basis if diagnostics.mol else "cc-pvdz",
        multiplicity=geometry.multiplicity,
        charge=geometry.charge,
        description=f"{geometry.name}_active_space",
    )

    mol_data = run_pyscf(mol_data, run_scf=True, run_mp2=True, run_fci=False)

    occupied_indices = diagnostics.core_orbital_indices
    active_indices = diagnostics.proposed_active_indices

    molecular_hamiltonian = mol_data.get_molecular_hamiltonian(
        occupied_indices=occupied_indices,
        active_indices=active_indices,
    )

    qubit_op = jordan_wigner(molecular_hamiltonian)

    # Extract terms
    n_qubits = 2 * active_space.n_active_orbitals
    terms = {}
    for term, coeff in qubit_op.terms.items():
        if abs(coeff) < 1e-12:
            continue
        pauli_str = _term_to_pauli_string(term, n_qubits)
        terms[pauli_str] = complex(coeff)

    result.openfermion_qubit_op = qubit_op
    result.n_qubits = n_qubits
    result.n_terms = len(terms)
    result.terms = terms

    # Build framework-compatible QubitHamiltonian
    coefficients = np.array([c.real for c in terms.values()])
    pauli_strings = list(terms.keys())

    molecule = Molecule(
        abbreviation=geometry.name[:3],
        name=geometry.name,
        n_qubits=n_qubits,
        n_coefficients=len(terms),
        reference_energy=diagnostics.hf_energy,
        hamiltonian_file=f"pipeline_{geometry.name}",
        n_electrons=active_space.n_active_electrons,
        n_orbitals=active_space.n_active_orbitals,
        charge=geometry.charge,
        spin=geometry.spin,
        basis=diagnostics.mol.basis if diagnostics.mol else "cc-pvdz",
        molecular_formula=geometry.formula,
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
