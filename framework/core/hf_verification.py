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

from .hamiltonian_loader import QubitHamiltonian

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
    import re

    # ── Fast path: simple IXYZ string ────────────────────────────────
    if isinstance(pauli_str_raw, str) and '(' not in pauli_str_raw:
        ps = pauli_str_raw
        if len(ps) >= n_qubits:
            return ps[:n_qubits]
        return ps + 'I' * (n_qubits - len(ps))

    # ── General path: parse operator(qubit) patterns ─────────────────
    pauli_chars = ['I'] * n_qubits

    if isinstance(pauli_str_raw, str):
        pauli_ops = [pauli_str_raw]
    elif isinstance(pauli_str_raw, (list, tuple)):
        pauli_ops = pauli_str_raw
    else:
        pauli_ops = [str(pauli_str_raw)]

    for op_str in pauli_ops:
        op_str = str(op_str).strip()

        # Extract operator type and qubit index
        if 'Identity(' in op_str:
            op_type = 'I'
        elif 'PauliX(' in op_str:
            op_type = 'X'
        elif 'PauliY(' in op_str:
            op_type = 'Y'
        elif 'PauliZ(' in op_str:
            op_type = 'Z'
        else:
            op_type = op_str[0] if op_str and op_str[0] in 'IXYZ' else 'I'

        try:
            match = re.search(r'\((\d+)\)', op_str)
            if match:
                qubit_idx = int(match.group(1))
                if 0 <= qubit_idx < n_qubits:
                    pauli_chars[qubit_idx] = op_type
        except (ValueError, IndexError):
            pass

    return ''.join(pauli_chars)

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



def _build_hamiltonian_matrix(hamiltonian: QubitHamiltonian) -> np.ndarray:
    """
    Build the full 2^n × 2^n Hamiltonian matrix (exact diagonalisation
    approach).  Only practical for n_qubits ≤ ~14.
    """
    n_qubits = hamiltonian.n_qubits
    dim = 2 ** n_qubits

    pauli_mats = {
        "I": np.eye(2, dtype=complex),
        "X": np.array([[0, 1], [1, 0]], dtype=complex),
        "Y": np.array([[0, -1j], [1j, 0]], dtype=complex),
        "Z": np.array([[1, 0], [0, -1]], dtype=complex),
    }

    H = np.zeros((dim, dim), dtype=complex)

    for coeff, pauli_str_raw in zip(hamiltonian.coefficients, hamiltonian.pauli_strings):
        if abs(coeff) < 1e-15:
            continue
            
        # Parse the PennyLane format to simple format
        pauli_str = _parse_pauli_string(pauli_str_raw, n_qubits)
        
        # Tensor product of single-qubit Pauli matrices
        op = pauli_mats[pauli_str[0]]
        for ch in pauli_str[1:]:
            op = np.kron(op, pauli_mats[ch])
        H += coeff * op

    return H


# ---------------------------------------------------------------------------
# Public API — matrix method (exact, for small systems)
# ---------------------------------------------------------------------------

def compute_hf_energy(
    hamiltonian: QubitHamiltonian,
    n_electrons: Optional[int] = None,
) -> float:
    """
    Compute ⟨HF|H|HF⟩ for the given qubit Hamiltonian.

    For small systems (n_qubits ≤ 14) this builds the full Hamiltonian
    matrix.  For larger systems it falls back to evaluating each Pauli
    term individually on the HF bitstring (no matrix needed).

    Args:
        hamiltonian: A QubitHamiltonian instance.
        n_electrons: Number of occupied spin-orbitals.  If *None* the
                     value stored in ``hamiltonian.molecule.n_electrons``
                     is used (falling back to n_qubits // 2).

    Returns:
        The Hartree-Fock energy (float).
    """
    try:
        n_qubits = hamiltonian.n_qubits
        if n_electrons is None:
            n_electrons = hamiltonian.molecule.n_electrons or n_qubits // 2

        # Debug: examine a few pauli strings to understand the format
        logger.debug(f"HF energy computation: n_qubits={n_qubits}, n_electrons={n_electrons}")
        if len(hamiltonian.pauli_strings) > 0:
            logger.debug(f"First pauli string: {repr(hamiltonian.pauli_strings[0])}")
            logger.debug(f"First coefficient: {hamiltonian.coefficients[0]}")

        if n_qubits <= 14:
            return _compute_hf_energy_matrix(hamiltonian, n_electrons)
        else:
            return _compute_hf_energy_diagonal(hamiltonian, n_electrons)
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
        *passed* is True when no expected value was given **or** when the
        computed energy matches.  *info_dict* contains all computed values.
    """
    n_qubits = hamiltonian.n_qubits
    if n_electrons is None:
        n_electrons = hamiltonian.molecule.n_electrons or n_qubits // 2

    info: Dict = {
        "n_qubits": n_qubits,
        "n_electrons": n_electrons,
    }

    # --- Compute via fast diagonal / matrix method ---
    hf_energy = compute_hf_energy(hamiltonian, n_electrons)
    info["hf_energy"] = hf_energy

    # --- Optionally cross-check with PennyLane ---
    if use_pennylane:
        hf_energy_pl = compute_hf_energy_pennylane(hamiltonian, n_electrons)
        info["hf_energy_pennylane"] = hf_energy_pl
        info["pennylane_match"] = bool(np.isclose(hf_energy, hf_energy_pl, atol=atol))
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
        passed = bool(np.isclose(hf_energy, expected_hf_energy, atol=atol))
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

    return passed, info
