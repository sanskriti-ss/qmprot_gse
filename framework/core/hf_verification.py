# ABOUTME: Evaluates the molecular or projected reference state against the encoded Hamiltonian.
# ABOUTME: Compares reference energies from matrix and circuit methods.
"""
Hartree-Fock Energy Verification Module

Computes ⟨HF|H|HF⟩ for a given Hamiltonian and reference state to verify
that the Hartree-Fock energy is correct before running VQE optimisation.

Usage:
    from core.hf_verification import compute_hf_energy, verify_hf_energy

    hf_energy = compute_hf_energy(hamiltonian)
    ok, info   = verify_hf_energy(hamiltonian, expected_hf_energy=-1.117)
"""

import numpy as np
import logging
from typing import Optional, Tuple, Dict

from .hamiltonian_loader import QubitHamiltonian, parity_signs, pauli_masks

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Core helpers
# ---------------------------------------------------------------------------

def _parse_pauli_string(pauli_str_raw, n_qubits):
    """
    Convert a Pauli-string representation to simple IXYZ format.

    Supported inputs:
        - Simple IXYZ string: 'ZXIIIIII' → returned as-is (truncated/padded
          to n_qubits)
        - OpenFermion style:  'Z(0) X(2)' or 'Z(0) @ X(2)'
        - PennyLane style:    'PauliZ(0)' / 'Identity(0)'
        - list/tuple of the above
    """
    if isinstance(pauli_str_raw, (list, tuple)):
        pauli_str_raw = ' '.join(map(str, pauli_str_raw))
    word = str(pauli_str_raw)
    if '(' in word or word == 'Identity':
        return QubitHamiltonian._openfermion_to_pauli_string(word, n_qubits)
    return word[:n_qubits].ljust(n_qubits, 'I')


def _hf_bitstring(n_qubits: int, n_electrons: int) -> np.ndarray:
    """
    Return the computational-basis statevector for the Hartree-Fock state.

    Convention (Jordan-Wigner): the first *n_electrons* spin-orbitals are
    occupied → qubit register |1…1 0…0⟩ (big-endian, qubit-0 is leftmost).

    Returns:
        1-D complex array of length 2^n_qubits with a single 1.0 entry.
    """
    n_occ = min(n_electrons, n_qubits)
    # Build the integer index of |1…1 0…0⟩
    # qubit-0 is the most-significant bit → index has the top n_occ bits set
    index = 0
    for q in range(n_occ):
        index |= 1 << (n_qubits - 1 - q)

    state = np.zeros(2 ** n_qubits, dtype=complex)
    state[index] = 1.0 + 0.0j
    return state



def _build_hamiltonian_matrix(hamiltonian: QubitHamiltonian, *, row: Optional[int] = None) -> np.ndarray:
    """
    Build the full 2^n × 2^n Hamiltonian matrix or a single computational-basis row.
    The full matrix is only practical for n_qubits ≤ ~14.
    """
    n_qubits = hamiltonian.n_qubits
    dim = 2 ** n_qubits

    pauli_mats = {
        "I": np.eye(2, dtype=complex),
        "X": np.array([[0, 1], [1, 0]], dtype=complex),
        "Y": np.array([[0, -1j], [1j, 0]], dtype=complex),
        "Z": np.array([[1, 0], [0, -1]], dtype=complex),
    }

    if row is not None and not 0 <= row < dim:
        raise IndexError(f"Hamiltonian row {row} is outside dimension {dim}")
    rows = np.arange(dim) if row is None else np.array([row])
    output_rows = rows if row is None else np.array([0])
    H = np.zeros((len(rows), dim), dtype=complex)
    signs = parity_signs(n_qubits)

    for coeff, pauli_str_raw in zip(hamiltonian.coefficients, hamiltonian.pauli_strings):
        if abs(coeff) < 1e-15:
            continue
            
        # Parse the PennyLane format to simple format
        pauli_str = _parse_pauli_string(pauli_str_raw, n_qubits)
        if n_qubits and np.isfinite(coeff) and all(p in pauli_mats for p in pauli_str):
            flip, parity = pauli_masks(pauli_str)
            values = ((-1j) ** pauli_str.count("Y")) * signs[rows & parity]
            H[output_rows, rows ^ flip] += coeff * values
            continue
        
        # Tensor product of single-qubit Pauli matrices
        op = pauli_mats[pauli_str[0]]
        for ch in pauli_str[1:]:
            op = np.kron(op, pauli_mats[ch])
        H += coeff * (op if row is None else op[rows])

    return H if row is None else H[0]


# ---------------------------------------------------------------------------
# Public API — matrix method (exact, for small systems)
# ---------------------------------------------------------------------------

def compute_hf_energy(
    hamiltonian: QubitHamiltonian,
    n_electrons: Optional[int] = None,
) -> float:
    """
    Compute ⟨HF|H|HF⟩ for the given qubit Hamiltonian.

    A computational-basis reference uses the ordered I/Z contributions to
    its matrix element. A contextual reference uses its prepared state.

    Args:
        hamiltonian: A QubitHamiltonian instance.
        n_electrons: Number of occupied spin-orbitals.  If *None* the
                     value stored in ``hamiltonian.molecule.n_electrons``
                     is used (falling back to n_qubits // 2).

    Returns:
        The Hartree-Fock energy (float).
    """
    cs_state = getattr(hamiltonian, "cs_initial_state", None)
    if cs_state is not None and np.linalg.norm(cs_state) > 1e-10:
        return compute_hf_energy_pennylane(hamiltonian, n_electrons=n_electrons)
    try:
        n_qubits = hamiltonian.n_qubits
        if n_electrons is None:
            n_electrons = hamiltonian.molecule.n_electrons or n_qubits // 2

        # Debug: examine a few pauli strings to understand the format
        logger.debug(f"HF energy computation: n_qubits={n_qubits}, n_electrons={n_electrons}")
        if len(hamiltonian.pauli_strings) > 0:
            logger.debug(f"First pauli string: {repr(hamiltonian.pauli_strings[0])}")
            logger.debug(f"First coefficient: {hamiltonian.coefficients[0]}")

        small = n_qubits <= 14
        coeffs = np.asarray(hamiltonian.coefficients)
        if small and (not n_qubits or coeffs.dtype.kind not in "fiu" or coeffs.dtype.itemsize > 8
                      or not np.isfinite(coeffs).all()
                      or np.max(np.abs(coeffs), initial=0.) > np.finfo(float).max / max(1, coeffs.size)):
            return _compute_hf_energy_matrix(hamiltonian, n_electrons)
        return _compute_hf_energy_diagonal(hamiltonian, max(0, n_electrons) if small else n_electrons,
                                           strict=small)
    except Exception as e:
        logger.error(f"Error in compute_hf_energy: {e}")
        logger.debug(f"Hamiltonian type: {type(hamiltonian)}")
        logger.debug(f"Pauli strings type: {type(hamiltonian.pauli_strings)}")
        if len(hamiltonian.pauli_strings) > 0:
            logger.debug(f"First pauli string repr: {repr(hamiltonian.pauli_strings[0])}")
        raise


def _compute_hf_energy_matrix(
    hamiltonian: QubitHamiltonian,
    n_electrons: int,
) -> float:
    """Exact matrix method for small qubit counts."""
    H_mat = _build_hamiltonian_matrix(hamiltonian)
    hf_state = _hf_bitstring(hamiltonian.n_qubits, n_electrons)
    energy = np.real(hf_state.conj() @ H_mat @ hf_state)
    return float(energy)


def _compute_hf_energy_diagonal(
    hamiltonian: QubitHamiltonian,
    n_electrons: int,
    strict: bool = False,
) -> float:
    """
    Efficient diagonal-only evaluation for large qubit counts.

    Because the HF state is a single computational-basis state, only the
    *diagonal* element of H matters.  For a Pauli string that contains
    any X or Y operator, the diagonal element is zero.  For a string
    composed only of I and Z, the diagonal element is the product of the
    eigenvalues ±1 of each Z on the corresponding qubit.
    """
    n_qubits = hamiltonian.n_qubits
    n_occ = min(n_electrons, n_qubits)

    # Build the occupation vector: 1 for occupied, 0 for virtual
    occupation = np.zeros(n_qubits, dtype=int)
    occupation[:n_occ] = 1  # first n_occ qubits occupied

    energy = 0.0
    for coeff, pauli_str_raw in zip(hamiltonian.coefficients, hamiltonian.pauli_strings):
        try:
            if abs(coeff) < 1e-15:
                continue
            
            # Parse the PennyLane format to simple format
            pauli_str = _parse_pauli_string(pauli_str_raw, n_qubits)
            if strict:
                for letter in pauli_str:
                    if letter not in "IXYZ":
                        raise KeyError(letter)
            
            # Any X or Y → off-diagonal → skip
            if "X" in pauli_str or "Y" in pauli_str:
                continue
            
            # Product of eigenvalues for Z on occupied (|1⟩→-1) / virtual (|0⟩→+1)
            eigenvalue = 1.0
            for q, ch in enumerate(pauli_str):
                if q >= n_qubits:
                    break  # Safety check
                if ch == "Z":
                    eigenvalue *= (-1.0 if occupation[q] == 1 else 1.0)
                # 'I' contributes factor 1
            energy += float(coeff) * eigenvalue
            
        except Exception as e:
            if strict:
                raise
            # Log the problematic term and continue
            logger.debug(f"Skipping term due to parsing error: coeff={coeff}, pauli_str={pauli_str_raw}, error={e}")
            continue

    return float(energy)


# ---------------------------------------------------------------------------
# Public API — PennyLane circuit method
# ---------------------------------------------------------------------------

def compute_hf_energy_pennylane(
    hamiltonian: QubitHamiltonian,
    n_electrons: Optional[int] = None,
    device_name: str = "default.qubit",
) -> float:
    """
    Compute ⟨HF|H|HF⟩ using a PennyLane circuit (useful as a sanity
    check that the PennyLane Hamiltonian conversion is correct).

    Args:
        hamiltonian: A QubitHamiltonian instance.
        n_electrons: Number of occupied spin-orbitals.
        device_name: PennyLane device to use.

    Returns:
        The Hartree-Fock energy (float).
    """
    import pennylane as qml

    n_qubits = hamiltonian.n_qubits
    if n_electrons is None:
        n_electrons = hamiltonian.molecule.n_electrons or n_qubits // 2
    n_occ = min(n_electrons, n_qubits)

    H = hamiltonian.to_pennylane()
    dev = qml.device(device_name, wires=n_qubits)

    @qml.qnode(dev)
    def hf_circuit():
        state = getattr(hamiltonian, "cs_initial_state", None)
        if state is not None and np.linalg.norm(state) > 1e-10:
            qml.StatePrep(state, wires=range(n_qubits))
        else:
            # Prepare |1…1 0…0⟩ (Jordan-Wigner HF state)
            for i in range(n_occ):
                qml.PauliX(wires=i)
        return qml.expval(H)

    return float(hf_circuit())


# ---------------------------------------------------------------------------
# Public API — verification helper
# ---------------------------------------------------------------------------

def verify_hf_energy(
    hamiltonian: QubitHamiltonian,
    expected_hf_energy: Optional[float] = None,
    n_electrons: Optional[int] = None,
    atol: float = 1e-6,
    use_pennylane: bool = False,
) -> Tuple[bool, Dict]:
    """
    Compute the HF energy and optionally compare it to an expected value.

    Args:
        hamiltonian:         QubitHamiltonian to check.
        expected_hf_energy:  If provided, check that the computed HF energy
                             matches within *atol*.
        n_electrons:         Occupied spin-orbitals (auto-detected if None).
        atol:                Absolute tolerance for the comparison.
        use_pennylane:       Also run the PennyLane circuit check.

    Returns:
        (passed, info_dict)
        *passed* is True when every requested comparison agrees within
        the absolute tolerance. *info_dict* contains all computed values.
    """
    n_qubits = hamiltonian.n_qubits
    if n_electrons is None:
        n_electrons = hamiltonian.molecule.n_electrons or n_qubits // 2

    info: Dict = {
        "n_qubits": n_qubits,
        "n_electrons": n_electrons,
    }

    # --- Compute via fast diagonal / matrix method ---
    state = hamiltonian.cs_initial_state
    if use_pennylane and state is not None and np.linalg.norm(state) > 1e-10:
        matrix = hamiltonian._extend_sparse_matrix(None, hamiltonian.coefficients, hamiltonian.pauli_strings, n_qubits)
        hf_energy = float(np.vdot(state, matrix @ state).real)
    else:
        hf_energy = compute_hf_energy(hamiltonian, n_electrons)
    info["hf_energy"] = hf_energy

    # --- Optionally cross-check with PennyLane ---
    if use_pennylane:
        hf_energy_pl = compute_hf_energy_pennylane(hamiltonian, n_electrons)
        info["hf_energy_pennylane"] = hf_energy_pl
        info["pennylane_match"] = bool(np.isclose(hf_energy, hf_energy_pl, atol=atol, rtol=0))
        if not info["pennylane_match"]:
            logger.warning(
                f"PennyLane HF energy ({hf_energy_pl:.8f}) differs from "
                f"matrix HF energy ({hf_energy:.8f}) by "
                f"{abs(hf_energy - hf_energy_pl):.2e}"
            )

    # --- Compare to expected value ---
    passed = True
    if expected_hf_energy is not None:
        info["expected_hf_energy"] = expected_hf_energy
        info["hf_error"] = hf_energy - expected_hf_energy
        passed = bool(np.isclose(hf_energy, expected_hf_energy, atol=atol, rtol=0))
        info["hf_matches_expected"] = passed
        if not passed:
            logger.warning(
                f"HF energy mismatch!  Computed {hf_energy:.8f}, "
                f"expected {expected_hf_energy:.8f} "
                f"(Δ = {hf_energy - expected_hf_energy:.2e})"
            )
    else:
        info["hf_matches_expected"] = None  # no expected value supplied

    # Log summary
    logger.info(
        f"HF verification for {hamiltonian.molecule.abbreviation}: "
        f"⟨HF|H|HF⟩ = {hf_energy:.8f} Ha  "
        f"(n_qubits={n_qubits}, n_electrons={n_electrons})"
    )

    return passed and info.get("pennylane_match", True), info
