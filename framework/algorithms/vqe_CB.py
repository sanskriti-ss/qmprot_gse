"""
Classically-Boosted VQE (CB-VQE) Implementation

Implements the classically-boosted VQE algorithm from:
    M. D. Radin, "Classically-Boosted Variational Quantum Eigensolver",
    arXiv:2106.04755 (2021).

Reference tutorial:
    https://pennylane.ai/qml/demos/tutorial_classically_boosted_vqe

The idea: solve a generalized eigenvalue problem in a 2D subspace spanned by
    |phi_cl> = Hartree-Fock state  (classical, single Slater determinant)
    |phi_q>  = VQE-optimised state (quantum, ansatz output)

This reduces measurement overhead vs. standard VQE while improving energy
estimates, especially for noisy / shot-limited settings.
"""

import math
import numpy as np
from typing import Optional, Any, List
from scipy import linalg
import logging
import time

import sys
sys.path.append('..')
from core.base_vqe import BaseVQE, VQEResult
from core.hamiltonian_loader import QubitHamiltonian
from core.hf_verification import _build_hamiltonian_matrix, _parse_pauli_string

logger = logging.getLogger(__name__)


def _hf_index(n_qubits: int, n_electrons: int) -> int:
    """
    Return the computational-basis index for the HF state |1…1 0…0⟩.
    Qubit-0 is the most-significant bit (big-endian / Jordan-Wigner).
    """
    n_occ = min(n_electrons, n_qubits)
    idx = 0
    for q in range(n_occ):
        idx |= 1 << (n_qubits - 1 - q)
    return idx


def _single_double_excitations(n_electrons: int, n_qubits: int):
    """
    Generate all single- and double-excitation Slater determinants
    from the HF reference (first n_electrons orbitals occupied).

    Returns a list of binary arrays of length n_qubits, each
    representing a computational-basis state.
    """
    n_occ = min(n_electrons, n_qubits)
    occupied = list(range(n_occ))
    virtual = list(range(n_occ, n_qubits))

    states = []

    # HF state itself (always included as the first basis state)
    hf = np.zeros(n_qubits, dtype=int)
    hf[:n_occ] = 1
    states.append(hf.copy())

    # Singles: move one electron from occupied -> virtual
    for i in occupied:
        for a in virtual:
            s = hf.copy()
            s[i] = 0
            s[a] = 1
            states.append(s)

    # Doubles: move two electrons from occupied -> virtual
    for i_idx, i in enumerate(occupied):
        for j in occupied[i_idx + 1:]:
            for a_idx, a in enumerate(virtual):
                for b in virtual[a_idx + 1:]:
                    d = hf.copy()
                    d[i] = 0
                    d[j] = 0
                    d[a] = 1
                    d[b] = 1
                    states.append(d)

    return states


# ---------------------------------------------------------------------------
# CB-VQE class
# ---------------------------------------------------------------------------

class ClassicallyBoostedVQE(BaseVQE):
    """
    Classically-Boosted VQE.

    1. Run standard VQE to get |phi_q> and E_VQE.
    2. Build the 2x2 projected Hamiltonian **H** and overlap matrix **S**
       in the {|phi_cl>, |phi_q>} subspace.
    3. Solve the generalized eigenvalue problem  H c = E S c.

    The classical state |phi_cl> is the Hartree-Fock determinant.
    Cross-terms are computed via the Hadamard test (quantum) combined
    with classically extractable matrix elements.
    """

    def __init__(
        self,
        hamiltonian: QubitHamiltonian,
        n_layers: int = 2,
        **kwargs,
    ):
        super().__init__(hamiltonian, **kwargs)

        self.name = "cb_vqe"
        self.description = "Classically-Boosted VQE"
        self.n_layers = n_layers

        # Effective n_electrons for the (possibly truncated) active space.
        # When the full-molecule electron count exceeds the qubit count the
        # Hamiltonian has been projected into an active space – use half-
        # filling so the HF state has both occupied and virtual orbitals.
        n_el_raw = self.hamiltonian.molecule.n_electrons or self.n_qubits // 2
        if n_el_raw >= self.n_qubits:
            self._eff_n_electrons = max(1, self.n_qubits // 2)
        else:
            self._eff_n_electrons = n_el_raw

        # Set during build / run
        self.device = None
        self.cost_fn = None
        self._vqe_energy: Optional[float] = None
        self._cb_energy: Optional[float] = None

    # ------------------------------------------------------------------
    # HF verification with effective n_electrons for truncated system
    # ------------------------------------------------------------------

    def _perform_hf_verification(self) -> None:
        """Compute HF energy using the effective active-space electron count."""
        from core.hf_verification import compute_hf_energy
        try:
            self.hf_energy = compute_hf_energy(
                self.hamiltonian, n_electrons=self._eff_n_electrons
            )
            logger.info(
                f"HF energy (truncated, n_el={self._eff_n_electrons}) = "
                f"{self.hf_energy:.8f} Ha"
            )
        except Exception as exc:
            logger.warning(f"Could not compute HF energy: {exc}")
            self.hf_energy = None

    # ------------------------------------------------------------------
    # Ansatz (same lightweight hardware-efficient ansatz as VanillaVQE)
    # ------------------------------------------------------------------

    def build_ansatz(self) -> Any:
        """Build parameterised ansatz circuit and cost QNode."""
        import pennylane as qml
        from core.backend_manager import create_device

        n_qubits = self.n_qubits
        n_layers = self.n_layers
        self.n_parameters = n_qubits * 3 * n_layers

        self.device = create_device(self.backend_config)
        H_pl = self.hamiltonian.to_pennylane()
        insert_noise = self.noise_inserter

        @qml.qnode(self.device)
        def circuit(params):
            params = params.reshape(n_layers, n_qubits, 3)

            self._prepare_initial_state()

            for layer in range(n_layers):
                for qubit in range(n_qubits):
                    qml.RX(params[layer, qubit, 0], wires=qubit)
                    qml.RY(params[layer, qubit, 1], wires=qubit)
                    qml.RZ(params[layer, qubit, 2], wires=qubit)

                for i in range(0, n_qubits - 1, 2):
                    qml.CNOT(wires=[i, i + 1])
                for i in range(1, n_qubits - 1, 2):
                    qml.CNOT(wires=[i, i + 1])

                insert_noise()

            return qml.expval(H_pl)

        self.cost_fn = circuit
        logger.info(
            f"CB-VQE ansatz: {self.n_parameters} params, "
            f"{n_layers} layers, backend={self.backend_config.label}"
        )
        return circuit

    def cost_function(self, parameters: np.ndarray) -> float:
        if self.cost_fn is None:
            raise RuntimeError("Must call build_ansatz() first")
        return float(self.cost_fn(parameters))

    def get_initial_parameters(self) -> np.ndarray:
        """All zeros --> HF reference state (rotations = identity)."""
        return np.zeros(self.n_parameters)

    # ------------------------------------------------------------------
    # Ansatz unitary matrix (needed for Hadamard test)
    # ------------------------------------------------------------------

    def _ansatz_unitary(self, params: np.ndarray) -> np.ndarray:
        """Return the unitary matrix of the ansatz circuit for given params."""
        import pennylane as qml

        n_qubits = self.n_qubits
        n_layers = self.n_layers
        wire_order = list(range(n_qubits))

        def ansatz_circuit(params, wires):
            params = params.reshape(n_layers, n_qubits, 3)
            self._prepare_initial_state()
            for layer in range(n_layers):
                for qubit in range(n_qubits):
                    qml.RX(params[layer, qubit, 0], wires=qubit)
                    qml.RY(params[layer, qubit, 1], wires=qubit)
                    qml.RZ(params[layer, qubit, 2], wires=qubit)
                for i in range(0, n_qubits - 1, 2):
                    qml.CNOT(wires=[i, i + 1])
                for i in range(1, n_qubits - 1, 2):
                    qml.CNOT(wires=[i, i + 1])

        U = qml.matrix(ansatz_circuit, wire_order=wire_order)(params, wire_order)
        return np.array(U)

    def _basis_state_unitary(self, state: np.ndarray) -> np.ndarray:
        """Return the unitary that maps |0…0⟩ -> |state⟩ (product of X gates)."""
        import pennylane as qml

        n_qubits = self.n_qubits
        wire_order = list(range(n_qubits))

        def basis_circuit(state):
            qml.BasisState(state, wires=range(n_qubits))

        U = qml.matrix(basis_circuit, wire_order=wire_order)(state)
        return np.array(U)

    # ------------------------------------------------------------------
    # Hadamard test
    # ------------------------------------------------------------------

    def _hadamard_test(self, Uq: np.ndarray, Ucl: np.ndarray) -> float:
        """
        Compute Re(⟨0|Uq† Ucl|0⟩) via the Hadamard test.

        Uses one ancilla qubit (wire 0) and the system register (wires 1..n).
        Returns 2*p(0) - 1 where p(0) is the probability of measuring
        the ancilla in |0⟩.
        """
        import pennylane as qml

        n_qubits = self.n_qubits
        total_wires = n_qubits + 1
        wires = list(range(total_wires))

        from config import PENNYLANE_DEVICE
        dev = qml.device(PENNYLANE_DEVICE, wires=total_wires)

        # Controlled unitary: Uq† @ Ucl
        controlled_U = Uq.conj().T @ Ucl

        @qml.qnode(dev)
        def hadamard_circuit():
            qml.Hadamard(wires=0)
            # PennyLane >= 0.44: wires = [control, *targets]
            qml.ControlledQubitUnitary(
                controlled_U,
                wires=[0] + list(range(1, total_wires)),
            )
            qml.Hadamard(wires=0)
            return qml.probs(wires=[0])

        probs = hadamard_circuit()
        # Re(⟨0|U|0⟩) = 2 * P(ancilla=0) - 1
        return float(2.0 * probs[0] - 1.0)

    # ------------------------------------------------------------------
    # Classical boosting step
    # ------------------------------------------------------------------

    def _classical_boost(self, optimal_params: np.ndarray, vqe_energy: float) -> float:
        """
        Perform the classical boosting post-processing.

        Builds the 2×2 projected Hamiltonian **H** and overlap **S**
        in the {|HF⟩, |VQE⟩} subspace and returns the lowest eigenvalue
        of the generalised eigenvalue problem.
        """
        n_qubits = self.n_qubits
        n_electrons = self._eff_n_electrons

        logger.info(
            f"CB-VQE: computing classical boost ... "
            f"(n_electrons={n_electrons}, n_qubits={n_qubits})"
        )

        # ── Full Hamiltonian matrix (fermionic representation) ────────
        H_mat = _build_hamiltonian_matrix(self.hamiltonian)
        hf_idx = _hf_index(n_qubits, n_electrons)

        # ── H11: classical entry ⟨HF|H|HF⟩ ──────────────────────────
        H11 = float(np.real(H_mat[hf_idx, hf_idx]))
        S11 = 1.0
        logger.info(f"  H11 Classical Entry(HF energy)     = {H11:.8f}")

        # ── H22: quantum entry = VQE energy ──────────────────────────
        H22 = vqe_energy
        S22 = 1.0
        logger.info(f"  H22 Quantum Entry (VQE energy)    = {H22:.8f}")

        # ── Cross terms H12, S12 via Hadamard test ────────────────────
        # Relevant basis states: HF + single & double excitations
        basis_states = _single_double_excitations(n_electrons, n_qubits)
        logger.info(f"  Relevant basis states: {len(basis_states)}")

        # Get ansatz unitary
        Uq = self._ansatz_unitary(optimal_params)

        H12 = 0.0 + 0.0j
        S12 = 0.0

        for j, basis_state in enumerate(basis_states):
            # Unitary that prepares |basis_state⟩ from |0…0⟩
            Ucl = self._basis_state_unitary(basis_state)

            # Re(⟨phi_q | basis_state⟩) via Hadamard test
            y = self._hadamard_test(Uq, Ucl)

            # ⟨basis_state|H|HF⟩ from the fermionic Hamiltonian matrix
            binary_string = "".join(str(b) for b in basis_state)
            idx = int(binary_string, 2)
            overlap_H = H_mat[hf_idx, idx]

            H12 += y * overlap_H

            # The first basis state is the HF state itself →
            #   y0 = Re(⟨phi_q | HF⟩) → used for overlap S12
            if j == 0:
                S12 = y

        H21 = np.conjugate(H12)
        S21 = np.conjugate(S12)

        logger.info(f"  H12 (cross-term)    = {H12}")
        logger.info(f"  S12 (overlap)       = {S12}")

        # ── Solve generalised eigenvalue problem ──────────────────────
        H_proj = np.array([[H11, H12], [H21, H22]], dtype=complex)
        S_proj = np.array([[S11, S12], [S21, S22]], dtype=complex)

        try:
            eigenvalues = linalg.eigvals(H_proj, S_proj)
            real_evals = np.real(eigenvalues[np.isfinite(eigenvalues)])
            if len(real_evals) == 0:
                logger.warning("Generalised eigenvalue problem returned no finite eigenvalues; "
                               "falling back to VQE energy.")
                return vqe_energy
            cb_energy = float(np.min(real_evals))
        except linalg.LinAlgError as exc:
            logger.warning(f"Generalised eigenvalue problem failed ({exc}); "
                           f"falling back to VQE energy.")
            cb_energy = vqe_energy

        logger.info(f"  CB-VQE energy       = {cb_energy:.8f}")
        return cb_energy

    # ------------------------------------------------------------------
    # Override run() to add the classical-boosting post-processing
    # ------------------------------------------------------------------

    def run(self) -> VQEResult:
        """
        Run CB-VQE:
            1. Standard VQE optimisation → optimal params & E_VQE
            2. Classical boosting        → E_CB  (≤ E_VQE)
        """
        logger.info(f"Running {self.name} on {self.hamiltonian.molecule.name}")
        logger.info(f"Backend: {self.backend_config.label}")

        # ── HF verification (truncated active-space electron count) ─
        self._perform_hf_verification()

        # ── Progress bar ──────────────────────────────────────────────
        from tqdm import tqdm
        self.progress_bar = tqdm(
            total=self.max_iterations,
            desc=f"{self.name} on {self.hamiltonian.molecule.abbreviation}",
            unit="iter",
        )

        # ── Step 1: standard VQE ──────────────────────────────────────
        start_time = time.time()
        self.build_ansatz()
        optimal_params, vqe_energy = self.optimize()
        self._vqe_energy = vqe_energy

        # ── Step 2: classical boost ───────────────────────────────────
        try:
            cb_energy = self._classical_boost(optimal_params, vqe_energy)
        except Exception as exc:
            logger.warning(f"Classical boost failed ({exc}); using VQE energy.")
            cb_energy = vqe_energy
        self._cb_energy = cb_energy

        runtime = time.time() - start_time

        # ── Choose the best energy ────────────────────────────────────
        best_energy = min(vqe_energy, cb_energy)

        # ── Build result ──────────────────────────────────────────────
        ref_energy = self.hamiltonian.molecule.reference_energy
        error = best_energy - ref_energy
        relative_error = abs(error / ref_energy) if ref_energy != 0 else 0.0

        converged = (
            len(self.convergence_history) > 1
            and abs(self.convergence_history[-1] - self.convergence_history[-2])
            < self.convergence_threshold
        )

        result = VQEResult(
            molecule_abbrev=self.hamiltonian.molecule.abbreviation,
            molecule_name=self.hamiltonian.molecule.name,
            algorithm_name=self.name,
            calculated_energy=best_energy,
            reference_energy=ref_energy,
            error=error,
            relative_error=relative_error,
            n_iterations=self.iteration_count,
            n_qubits=self.n_qubits,
            n_parameters=self.n_parameters,
            runtime_seconds=runtime,
            convergence_history=self.convergence_history,
            optimal_parameters=optimal_params,
            converged=converged,
            metadata={
                "optimizer": self.optimizer_name,
                "max_iterations": self.max_iterations,
                "n_shots": self.n_shots,
                "random_seed": self.random_seed,
                "n_layers": self.n_layers,
                "vqe_energy": float(vqe_energy),
                "cb_energy": float(cb_energy),
                "improvement": float(vqe_energy - cb_energy),
            },
            backend_type=self.backend_config.backend_type,
            noise_model=self.backend_config.noise_model,
            noise_strength=self.backend_config.noise_strength,
            hf_energy=self.hf_energy,
        )

        logger.info(
            f"Completed {self.name}: "
            f"VQE={vqe_energy:.8f}, CB={cb_energy:.8f}, "
            f"Best={best_energy:.8f}, Error={error:.8f}, "
            f"Runtime={runtime:.2f}s"
        )

        if self.progress_bar:
            self.progress_bar.close()

        if self.hamiltonian.molecule.truncated_ground_state_energy is not None:
            logger.info(f"Full system reference energy: {ref_energy:.8f} Hartree")
            logger.info(
                f"Truncated system ground state: "
                f"{self.hamiltonian.molecule.truncated_ground_state_energy:.8f} Hartree"
            )

        return result
