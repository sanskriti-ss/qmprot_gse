"""
Contextual Subspace Reduction

Reduces the qubit count of a QubitHamiltonian by finding a noncontextual
sub-Hamiltonian and projecting out the classically-approximated qubits.
This produces a smaller Hamiltonian suitable for any VQE algorithm.

Reference: Kirby et al., https://arxiv.org/abs/2011.10027
"""

import logging
import time
from copy import deepcopy
from dataclasses import replace
from typing import Dict, List, Optional, Tuple

import numpy as np

from core.hamiltonian_loader import Molecule, QubitHamiltonian
from contextual_subspace.core import (
    contextualQ_ham,
    diagonalize_epistemic,
    apply_rotation,
    energy_function_form,
    find_gs_noncon,
    greedy_dfs,
    pauli_to_sparse,
    quasi_model,
)

logger = logging.getLogger(__name__)


def _build_reduced_hamiltonian(
    ham: Dict[str, float],
    model,
    fn_form,
    ep_state,
    n_generators_to_drop: int,
) -> Dict[str, float]:
    """Rotate the full Hamiltonian into the diagonal-generator basis and
    project out stabilized qubits, dropping ``n_generators_to_drop``
    generators so the result lives on more qubits.

    Returns the reduced Hamiltonian dict for the target qubit count.
    """
    rotations, diagonal_set, vals = diagonalize_epistemic(model, fn_form, ep_state)
    n_q = len(diagonal_set[0])
    vals = list(vals)

    # Drop generators 0..n_generators_to_drop-1 from the stabilizer set
    active_set = list(diagonal_set)[n_generators_to_drop:]
    active_vals = vals[n_generators_to_drop:]

    # Rotate the full Hamiltonian into the diagonal-generator basis
    ham_rotated: Dict[str, float] = dict(ham)
    for r in rotations:
        ham_next: Dict[str, float] = {}
        for t in ham_rotated:
            t_set_next = apply_rotation(r, t)
            for t_next, coeff in t_set_next.items():
                ham_next[t_next] = ham_next.get(t_next, 0) + coeff * ham_rotated[t]
        ham_rotated = ham_next

    # Find the Z-index for each remaining generator
    z_indices: List[int] = []
    for d in active_set:
        for idx in range(n_q):
            if d[idx] == "Z":
                z_indices.append(idx)
                break

    # Project out stabilized qubits
    ham_red: Dict[str, float] = {}
    for t in ham_rotated:
        sgn = 1.0
        for j in range(len(active_set)):
            z_index = z_indices[j]
            if t[z_index] == "Z":
                sgn *= active_vals[j]
            elif t[z_index] != "I":
                sgn = 0.0
                break
        if sgn != 0:
            t_red = "".join(t[idx] for idx in range(n_q) if idx not in z_indices)
            ham_red[t_red] = ham_red.get(t_red, 0) + ham_rotated[t] * sgn

    # Drop near-zero terms
    ham_red = {k: float(np.real(v)) for k, v in ham_red.items() if abs(v) > 1e-12}

    return ham_red


def _compute_cs_initial_state(
    n_qubits: int,
    n_electrons: Optional[int],
    model,
    fn_form,
    ep_state,
    n_generators_to_drop: int,
) -> np.ndarray:
    """Rotate the HF state into the CS basis and project out stabilized qubits.

    Returns the reduced statevector suitable for ``qml.StatePrep``.
    """
    from scipy.sparse import eye as sp_eye

    rotations, diagonal_set, vals = diagonalize_epistemic(model, fn_form, ep_state)
    n_q = len(diagonal_set[0])
    vals_list = list(vals)
    dim = 2 ** n_q

    # 1. Build HF statevector  |1...1 0...0>
    n_el = min(n_electrons or n_qubits // 2, n_qubits)
    state = np.zeros(dim, dtype=complex)
    hf_idx = sum(1 << (n_q - 1 - i) for i in range(n_el))
    state[hf_idx] = 1.0

    # 2. Apply the same rotations used on the Hamiltonian
    identity = sp_eye(dim, format="csr")
    for rotation in rotations:
        angle, gen_str = rotation
        G = pauli_to_sparse(gen_str)
        if angle == "pi/2":
            R = (identity - 1j * G) / np.sqrt(2)
        else:
            R = np.cos(angle / 2) * identity - 1j * np.sin(angle / 2) * G
        state = R @ state

    # 3. Project out stabilized qubits
    active_set = list(diagonal_set)[n_generators_to_drop:]
    active_vals = vals_list[n_generators_to_drop:]

    z_indices: List[int] = []
    for d in active_set:
        for idx in range(n_q):
            if d[idx] == "Z":
                z_indices.append(idx)
                break

    free_qubits = [q for q in range(n_q) if q not in z_indices]
    reduced_dim = 2 ** len(free_qubits)
    reduced_state = np.zeros(reduced_dim, dtype=complex)

    for full_idx in range(dim):
        keep = True
        for j, z_idx in enumerate(z_indices):
            bit = (full_idx >> (n_q - 1 - z_idx)) & 1
            expected_bit = 0 if active_vals[j] > 0 else 1
            if bit != expected_bit:
                keep = False
                break
        if keep and abs(state[full_idx]) > 1e-15:
            red_idx = 0
            for k, q in enumerate(free_qubits):
                bit = (full_idx >> (n_q - 1 - q)) & 1
                red_idx |= bit << (len(free_qubits) - 1 - k)
            reduced_state[red_idx] += state[full_idx]

    norm = np.linalg.norm(reduced_state)
    if norm > 1e-15:
        reduced_state /= norm

    return reduced_state


def apply_contextual_subspace_reduction(
    hamiltonian: QubitHamiltonian,
    target_qubits: int,
    dfs_cutoff_seconds: float = 60.0,
    dfs_criterion: str = "weight",
) -> Tuple[QubitHamiltonian, dict]:
    """Reduce a QubitHamiltonian via contextual subspace projection.

    Parameters
    ----------
    hamiltonian : QubitHamiltonian
        The input Hamiltonian (after active-space truncation).
    target_qubits : int
        Desired number of qubits in the reduced Hamiltonian.
    dfs_cutoff_seconds : float
        Time budget (seconds) for the greedy DFS search for the
        noncontextual sub-Hamiltonian.
    dfs_criterion : str
        Criterion for greedy DFS (``'weight'`` or ``'size'``).

    Returns
    -------
    reduced_hamiltonian : QubitHamiltonian
        A new QubitHamiltonian with ``target_qubits`` qubits.
    metadata : dict
        Diagnostic information about the reduction.
    """
    n_qubits = hamiltonian.n_qubits
    ham_dict = hamiltonian.to_dict()

    if target_qubits >= n_qubits:
        logger.info("target_qubits >= n_qubits; no CS reduction applied.")
        return hamiltonian, {"original_qubits": n_qubits, "reduced": False}

    if target_qubits < 1:
        raise ValueError("target_qubits must be >= 1")

    # --- 1. Find noncontextual sub-Hamiltonian ---
    logger.info(
        f"CS reduction: searching for noncontextual sub-Hamiltonian "
        f"(cutoff={dfs_cutoff_seconds}s, criterion={dfs_criterion})..."
    )
    t0 = time.time()
    best_guesses = greedy_dfs(ham_dict, dfs_cutoff_seconds, criterion=dfs_criterion)
    dfs_elapsed = time.time() - t0

    if not best_guesses or not best_guesses[-1]:
        logger.warning("No noncontextual sub-Hamiltonian found; returning original.")
        return hamiltonian, {"original_qubits": n_qubits, "reduced": False}

    terms_noncon = best_guesses[-1]
    ham_noncon = {t: ham_dict[t] for t in terms_noncon}
    noncon_weight = sum(abs(ham_noncon[t]) for t in ham_noncon)
    total_weight = sum(abs(ham_dict[t]) for t in ham_dict)
    logger.info(
        f"Noncontextual sub-Hamiltonian: {len(ham_noncon)}/{len(ham_dict)} terms "
        f"({noncon_weight:.4f}/{total_weight:.4f} total weight, "
        f"DFS took {dfs_elapsed:.1f}s)"
    )

    if contextualQ_ham(ham_noncon):
        logger.warning("Selected sub-Hamiltonian is contextual; returning original.")
        return hamiltonian, {"original_qubits": n_qubits, "reduced": False}

    # --- 2. Build quasi-quantized model & find noncontextual ground state ---
    logger.info("Building quasi-quantized model and solving noncontextual ground state...")
    model = quasi_model(ham_noncon)
    fn_form = energy_function_form(ham_noncon, model)
    gs_noncon, all_ep_candidates = find_gs_noncon(
        ham_noncon, method="differential_evolution", return_all=True
    )
    ep_state = gs_noncon[1]
    noncon_energy = gs_noncon[0]
    logger.info(f"Noncontextual ground state energy: {noncon_energy:.8f} Ha")

    # --- 3. Determine how many generators to keep / drop ---
    rotations, diagonal_set, vals = diagonalize_epistemic(model, fn_form, ep_state)
    n_generators = len(diagonal_set)
    min_qubits = n_qubits - n_generators

    if target_qubits < min_qubits:
        logger.warning(
            f"target_qubits={target_qubits} < minimum achievable={min_qubits}. "
            f"Using minimum ({min_qubits} qubits)."
        )
        target_qubits = min_qubits

    n_generators_to_drop = target_qubits - min_qubits
    n_generators_to_keep = n_generators - n_generators_to_drop

    logger.info(
        f"CS reduction: {n_qubits} -> {target_qubits} qubits "
        f"({n_generators_to_keep} generators stabilized, "
        f"{n_generators_to_drop} released to quantum)"
    )

    # --- 4. Build the reduced Hamiltonian and rotated initial state ---
    ham_red = _build_reduced_hamiltonian(
        ham_dict, model, fn_form, ep_state, n_generators_to_drop
    )

    cs_initial_state = _compute_cs_initial_state(
        n_qubits,
        hamiltonian.molecule.n_electrons,
        model, fn_form, ep_state,
        n_generators_to_drop,
    )

    # If the minimum-energy noncontextual sector doesn't contain the HF state
    # (zero norm after projection), search other sectors ordered by energy until
    # we find one that is HF-compatible.  This ensures VQE starts in the correct
    # physical sector rather than a spuriously low-energy wrong sector.
    if np.linalg.norm(cs_initial_state) < 1e-10 and len(all_ep_candidates) > 1:
        logger.info(
            "Default noncontextual sector has zero HF overlap; "
            f"searching for HF-compatible sector among {len(all_ep_candidates)} candidates..."
        )
        for alt_candidate in all_ep_candidates[1:]:
            alt_ep = alt_candidate[1]
            alt_state = _compute_cs_initial_state(
                n_qubits,
                hamiltonian.molecule.n_electrons,
                model, fn_form, alt_ep,
                n_generators_to_drop,
            )
            if np.linalg.norm(alt_state) > 1e-10:
                logger.info(
                    f"Found HF-compatible noncontextual sector "
                    f"(energy {alt_candidate[0]:.6f} Ha vs min {noncon_energy:.6f} Ha); "
                    f"rebuilding reduced Hamiltonian..."
                )
                ep_state = alt_ep
                cs_initial_state = alt_state
                ham_red = _build_reduced_hamiltonian(
                    ham_dict, model, fn_form, ep_state, n_generators_to_drop
                )
                break
        else:
            logger.warning(
                "No noncontextual sector contains the HF state; "
                "using default sector (VQE initial state will be naive HF fallback)."
            )

    logger.info(
        f"Rotated HF state: {np.sum(np.abs(cs_initial_state) > 1e-8)} non-zero amplitudes "
        f"in {len(cs_initial_state)}-dim Hilbert space"
    )

    if not ham_red:
        logger.warning("Reduced Hamiltonian is empty; returning original.")
        return hamiltonian, {"original_qubits": n_qubits, "reduced": False}

    # Verify qubit count
    sample_key = next(iter(ham_red))
    actual_qubits = len(sample_key) if sample_key else 0
    if actual_qubits == 0:
        energy_scalar = float(np.real(ham_red.get("", 0.0)))
        logger.info(f"Fully reduced to scalar energy: {energy_scalar}")

    # --- 5. Package as QubitHamiltonian ---
    # n_electrons/n_orbitals are cleared because the reduced qubits no longer
    # correspond to spin-orbitals under Jordan-Wigner.  Algorithms that need
    # an electron count will fall back to n_qubits // 2.
    reduced_molecule = Molecule(
        abbreviation=hamiltonian.molecule.abbreviation,
        name=hamiltonian.molecule.name,
        n_qubits=actual_qubits,
        n_coefficients=len(ham_red),
        reference_energy=hamiltonian.molecule.reference_energy,
        hamiltonian_file=hamiltonian.molecule.hamiltonian_file,
        n_electrons=None,
        n_orbitals=None,
        charge=hamiltonian.molecule.charge,
        spin=hamiltonian.molecule.spin,
        basis=hamiltonian.molecule.basis,
        coordinates=hamiltonian.molecule.coordinates,
        molecular_formula=hamiltonian.molecule.molecular_formula,
        core_energy=hamiltonian.molecule.core_energy,
    )

    reduced_hamiltonian = QubitHamiltonian.from_dict(ham_red, reduced_molecule)
    reduced_hamiltonian.cs_initial_state = cs_initial_state

    metadata = {
        "original_qubits": n_qubits,
        "reduced_qubits": actual_qubits,
        "reduced": True,
        "n_generators_total": n_generators,
        "n_generators_stabilized": n_generators_to_keep,
        "n_generators_released": n_generators_to_drop,
        "noncontextual_terms": len(ham_noncon),
        "total_terms": len(ham_dict),
        "reduced_terms": len(ham_red),
        "noncontextual_ground_state_energy": noncon_energy,
        "min_achievable_qubits": min_qubits,
    }

    return reduced_hamiltonian, metadata
