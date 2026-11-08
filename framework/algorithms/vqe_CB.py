# ABOUTME: Optimizes a variational circuit and measures its overlap with classical determinants.
# ABOUTME: Solves the projected energy problem for the Hartree–Fock and variational states.
"""
Classically-Boosted VQE (CB-VQE) Implementation

Implements the classically-boosted VQE algorithm from:
    M. D. Radin, "Classically-Boosted Variational Quantum Eigensolver",
    arXiv:2106.04755 (2021).

Reference tutorial:
    https://pennylane.ai/qml/demos/tutorial_classically_boosted_vqe

The idea: solve a generalized eigenvalue problem in a 2D subspace spanned by
    |phi_cl> = encoded Hartree-Fock reference (including its CS projection)
    |phi_q>  = VQE-optimised state (quantum, ansatz output)

The implemented complex Rayleigh–Ritz projection uses an analytic pure-state
simulator for both the variational energy and the measured cross terms.
"""

import numpy as np
from functools import lru_cache
from typing import Optional, Any
from scipy import linalg
import logging
import time

import sys
sys.path.append('..')
from core.base_vqe import BaseVQE, VQEResult, incumbent_history
from core.hamiltonian_loader import QubitHamiltonian
from core.hf_verification import _build_hamiltonian_matrix

logger = logging.getLogger(__name__)


@lru_cache(maxsize=2)
def _contextual_unitary(n_qubits, dtype, state_bytes):
    """Retain the two most recent small native preparation matrices by their complete input."""
    import pennylane as qml

    state = np.frombuffer(state_bytes, dtype=dtype)
    wires = tuple(range(n_qubits))
    matrix = np.asarray(qml.matrix(qml.StatePrep(state, wires=wires), wire_order=wires))
    matrix.flags.writeable = False
    return matrix


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

    The classical state |phi_cl> is the encoded reference, including its CS rotation.
    Cross-terms are computed via the Hadamard test (quantum) combined
    with classically extractable matrix elements.
    """

    parameter_period = 2 * np.pi

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

        if self.backend_config.backend_type != "statevector" or self.backend_config.n_shots != 0:
            raise ValueError("Classically boosted Ritz energies require an analytic pure statevector backend")

        # Set during build / run
        self.device = None
        self.cost_fn = None
        self._vqe_energy: Optional[float] = None
        self._cb_energy: Optional[float] = None

    # ------------------------------------------------------------------
    # Ansatz (same lightweight hardware-efficient ansatz as VanillaVQE)
    # ------------------------------------------------------------------

    def build_ansatz(self) -> Any:
        """Build parameterised ansatz circuit and cost QNode."""
        import pennylane as qml
        from core.backend_manager import create_device, fixed_circuit_energy

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

        circuit.func.rotation_layout = self._rotation_layout
        self.cost_fn = circuit
        self._energy = fixed_circuit_energy(circuit, H_pl)
        logger.info(
            f"CB-VQE ansatz: {self.n_parameters} params, "
            f"{n_layers} layers, backend={self.backend_config.label}"
        )
        return circuit

    def cost_function(self, parameters: np.ndarray) -> float:
        if self.cost_fn is None:
            raise RuntimeError("Must call build_ansatz() first")
        return float(self._energy(parameters))

    def get_initial_parameters(self) -> np.ndarray:
        """Initialize rotation angles at zero."""
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

        tape = qml.tape.make_qscript(ansatz_circuit)(params, wire_order)
        if tape.operations and type(tape.operations[0]) is qml.StatePrep and n_qubits <= 9:
            preparation = tape.operations[0]
            state = preparation.data[0]
            if (type(state) is np.ndarray and state.shape == (1 << n_qubits,)
                    and state.dtype in (np.float64, np.complex128) and tuple(preparation.wires) == tuple(wire_order)):
                matrix = _contextual_unitary(n_qubits, state.dtype.str, state.tobytes())
                first = qml.QubitUnitary(matrix, wires=wire_order)
                tape = qml.tape.QuantumScript([first, *tape.operations[1:]])
        U = qml.matrix(tape, wire_order=wire_order)
        return np.array(U)

    # ------------------------------------------------------------------
    # Hadamard test
    # ------------------------------------------------------------------

    def _hadamard_test(self, unitary: np.ndarray, index: int) -> complex:
        """Measure both quadratures of ⟨index|U|0⟩ on a Hadamard-test ancilla."""
        import pennylane as qml

        n_qubits = self.n_qubits
        total_wires = n_qubits + 1
        wires = list(range(total_wires))

        from config import PENNYLANE_DEVICE
        dev = qml.device(PENNYLANE_DEVICE, wires=total_wires)

        # The product of X gates preparing the basis state permutes the unitary rows.
        controlled_U = unitary[np.arange(1 << n_qubits) ^ index]

        @qml.qnode(dev)
        def hadamard_circuit():
            qml.Hadamard(wires=0)
            qml.ctrl(qml.QubitUnitary, control=0)(controlled_U, wires=wires[1:])
            return qml.expval(qml.PauliX(0)), qml.expval(qml.PauliY(0))

        real, imaginary = hadamard_circuit()
        return complex(real, imaginary)

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
        import pennylane as qml

        n_qubits = self.n_qubits
        device = qml.device("default.qubit", wires=n_qubits)

        @qml.qnode(device)
        def reference_state():
            self._prepare_initial_state()
            return qml.state()

        reference = np.asarray(reference_state())
        support = np.flatnonzero(reference)
        matrix = None
        if support.size == 1:
            idx = support[0]
            h_row = reference[idx].conj() * _build_hamiltonian_matrix(self.hamiltonian, row=idx)
        else:
            matrix = _build_hamiltonian_matrix(self.hamiltonian)
            h_row = reference.conj() @ matrix
        ref_energy = float(np.real(h_row @ reference))

        unitary = self._ansatz_unitary(optimal_params)
        h_cross = 0j
        overlap = 0j
        # Both off-diagonal elements use the same measured amplitude ⟨j|ψ⟩.
        for idx in np.flatnonzero((reference != 0) | (h_row != 0)):
            amplitude = self._hadamard_test(unitary, idx)
            h_cross += h_row[idx] * amplitude
            overlap += reference[idx].conj() * amplitude

        h_proj = np.array([[ref_energy, h_cross], [h_cross.conjugate(), vqe_energy]])
        s_proj = np.array([[1., overlap], [overlap.conjugate(), 1.]])
        if 1. - abs(overlap) < np.sqrt(np.finfo(float).eps):
            # The Gram matrix squares the condition number of the state pair.
            # Resolve a small independent direction directly on the analytic simulator.
            basis, triangular = np.linalg.qr(np.column_stack((reference, unitary[:, 0])))
            if abs(triangular[1, 1]) <= np.finfo(float).eps * max(basis.shape):
                return min(ref_energy, vqe_energy)
            if matrix is None:
                matrix = _build_hamiltonian_matrix(self.hamiltonian)
            cb_energy = float(linalg.eigvalsh(basis.conj().T @ matrix @ basis)[0])
        else:
            cb_energy = float(linalg.eigvalsh(h_proj, s_proj)[0])
        logger.info(f"  CB-VQE energy       = {cb_energy:.8f}")
        return cb_energy

    # ------------------------------------------------------------------
    # Override run() to add the classical-boosting post-processing
    # ------------------------------------------------------------------

    def run(self, initial_parameters: Optional[np.ndarray] = None) -> VQEResult:
        """
        Run CB-VQE:
            1. Standard VQE optimisation → optimal params & E_VQE
            2. Classical boosting        → E_CB  (≤ E_VQE)
        """
        logger.info(f"Running {self.name} on {self.hamiltonian.molecule.name}")
        logger.info(f"Backend: {self.backend_config.label}")

        # ── Reference energy ─
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
        try:
            self.build_ansatz()
            optimal_params, vqe_energy = self.optimize(initial_parameters=initial_parameters)
            self._vqe_energy = vqe_energy

            # ── Step 2: classical boost ───────────────────────────────────
            cb_energy = self._classical_boost(optimal_params, vqe_energy)
        finally:
            self.progress_bar.close()
        self._cb_energy = cb_energy

        runtime = time.time() - start_time

        # ── Choose the best energy ────────────────────────────────────
        best_energy = min(vqe_energy, cb_energy)

        # ── Build result ──────────────────────────────────────────────
        ref_energy = self.hamiltonian.molecule.reference_energy
        error = best_energy - ref_energy
        relative_error = abs(error / ref_energy) if ref_energy != 0 else 0.0

        converged = bool(self.optimizer_outcome["optimizer_success"])

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
            best_so_far_history=incumbent_history(self.convergence_history),
            optimal_parameters=optimal_params,
            converged=converged,
            metadata={
                **self.fit_diagnostics(),
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

        if self.hamiltonian.molecule.truncated_ground_state_energy is not None:
            logger.info(f"Full system reference energy: {ref_energy:.8f} Hartree")
            logger.info(
                f"Truncated system ground state: "
                f"{self.hamiltonian.molecule.truncated_ground_state_energy:.8f} Hartree"
            )

        return result
