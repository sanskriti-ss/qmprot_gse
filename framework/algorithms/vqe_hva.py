"""
Hamiltonian Variational Ansatz (HVA) VQE Implementation

Implements the Hamiltonian Variational Ansatz from:
    [1] D. Wecker et al., "Progress towards practical quantum variational
        algorithms", Phys. Rev. A 92, 042303 (2015).
    [2] R. Wiersema et al., "Exploring entanglement and optimization within
        the Hamiltonian Variational Ansatz", arXiv:2008.02941 (2020).

Qiskit reference:
    https://quantum.cloud.ibm.com/docs/en/api/qiskit/
    qiskit.circuit.library.hamiltonian_variational_ansatz

The idea:  Given H = sum_k H_k  where each H_k is a group of mutually
commuting Pauli terms (but different groups do NOT commute with each other),
the HVA ansatz is:

    |psi(theta)> = prod_{r=1}^{R} prod_{k} exp(-i theta_{k,r} H_k) |HF>

Each exp(-i theta H_k) is implemented exactly via PennyLane's
`qml.CommutingEvolution`.  Because the ansatz mirrors the Hamiltonian
structure, it is a physics-informed ansatz that can be more parameter-
efficient than generic hardware-efficient circuits.
"""

import numpy as np
from typing import Optional, Any, List, Tuple
import logging

import sys
sys.path.append('..')
from core.base_vqe import BaseVQE, VQEResult
from core.hamiltonian_loader import QubitHamiltonian

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Helper: group Pauli terms into commuting sets
# ---------------------------------------------------------------------------

def _pauli_str_to_simple(pauli_str_raw: str, n_qubits: int) -> str:
    """
    Convert an OpenFermion-style string like 'X(0) Z(2)' or a simple
    'IXZI' string into a fixed-length IXYZ string of length n_qubits.
    """
    if '(' not in str(pauli_str_raw):
        # Already in simple format
        ps = str(pauli_str_raw)
        if len(ps) >= n_qubits:
            return ps[:n_qubits]
        return ps + 'I' * (n_qubits - len(ps))

    import re
    chars = ['I'] * n_qubits
    for m in re.finditer(r'([IXYZ])\((\d+)\)', str(pauli_str_raw)):
        op, idx = m.group(1), int(m.group(2))
        if idx < n_qubits:
            chars[idx] = op
    return ''.join(chars)


def _commutes(ps_a: str, ps_b: str) -> bool:
    """
    Check if two Pauli strings (simple IXYZ format, same length) commute.

    Two Pauli strings commute iff the number of qubit positions where
    both are non-identity AND different is even.
    """
    anti = 0
    for a, b in zip(ps_a, ps_b):
        if a != 'I' and b != 'I' and a != b:
            anti += 1
    return anti % 2 == 0


def group_commuting_terms(
    coefficients: np.ndarray,
    pauli_strings: List[str],
    n_qubits: int,
) -> List[List[Tuple[float, str]]]:
    """
    Partition Hamiltonian terms into groups of mutually commuting terms.

    Uses a greedy algorithm: iterate over terms and place each term into
    the first existing group with which it commutes; if none, start a new
    group.

    Returns:
        List of groups, where each group is a list of (coeff, simple_pauli_str).
    """
    simple_strings = [_pauli_str_to_simple(ps, n_qubits) for ps in pauli_strings]

    groups: List[List[Tuple[float, str]]] = []

    for coeff, sps in zip(coefficients, simple_strings):
        if abs(coeff) < 1e-15:
            continue
        placed = False
        for group in groups:
            # Check commutativity with every term already in the group
            if all(_commutes(sps, existing_ps) for _, existing_ps in group):
                group.append((float(coeff), sps))
                placed = True
                break
        if not placed:
            groups.append([(float(coeff), sps)])

    return groups


# ---------------------------------------------------------------------------
# HVA VQE class
# ---------------------------------------------------------------------------

class HamiltonianVariationalVQE(BaseVQE):
    """
    VQE with a Hamiltonian Variational Ansatz (HVA).

    The ansatz applies Trotterised time-evolution layers built directly
    from the groups of commuting terms in the Hamiltonian.  One variational
    parameter theta_{k,r} per commuting group per repetition.

    References:
        [1] Wecker et al., Phys. Rev. A 92 042303 (2015)
        [2] Wiersema et al., arXiv:2008.02941 (2020)
    """

    def __init__(
        self,
        hamiltonian: QubitHamiltonian,
        n_layers: int = 2,
        **kwargs,
    ):
        """
        Args:
            hamiltonian: QubitHamiltonian object.
            n_layers:    Number of Trotter repetitions (reps).
            **kwargs:    Passed to BaseVQE (optimizer, max_iterations, …).
        """
        super().__init__(hamiltonian, **kwargs)

        self.name = "hva_vqe"
        self.description = "Hamiltonian Variational Ansatz VQE"
        self.n_layers = n_layers

        # Effective n_electrons for the (possibly truncated) active space.
        n_el_raw = self.hamiltonian.molecule.n_electrons or self.n_qubits // 2
        if n_el_raw >= self.n_qubits:
            self._eff_n_electrons = max(1, self.n_qubits // 2)
        else:
            self._eff_n_electrons = n_el_raw

        # Built during build_ansatz
        self.device = None
        self.cost_fn = None
        self._commuting_groups: List[List[Tuple[float, str]]] = []

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
    # Ansatz construction
    # ------------------------------------------------------------------

    def build_ansatz(self) -> Any:
        """
        Build the HVA circuit.

        Steps:
            1. Partition the Hamiltonian into commuting groups {H_k}.
            2. For each rep r and each group k, apply exp(-i theta_{k,r} H_k).
            3. One parameter per (group, rep) pair.
        """
        import pennylane as qml
        from core.backend_manager import create_device

        n_qubits = self.n_qubits
        n_layers = self.n_layers

        # ── 1. Group commuting terms ──────────────────────────────────
        self._commuting_groups = group_commuting_terms(
            self.hamiltonian.coefficients,
            self.hamiltonian.pauli_strings,
            n_qubits,
        )
        n_groups = len(self._commuting_groups)
        logger.info(
            f"HVA: {len(self.hamiltonian.coefficients)} terms -> "
            f"{n_groups} commuting groups"
        )

        # One parameter per group per layer
        self.n_parameters = n_groups * n_layers

        # ── 2. Build PennyLane Hamiltonians for each group ────────────
        group_hamiltonians = []
        for group in self._commuting_groups:
            coeffs = []
            ops = []
            for coeff, sps in group:
                pauli_ops = []
                for i, p in enumerate(sps):
                    if p == 'X':
                        pauli_ops.append(qml.PauliX(i))
                    elif p == 'Y':
                        pauli_ops.append(qml.PauliY(i))
                    elif p == 'Z':
                        pauli_ops.append(qml.PauliZ(i))
                if pauli_ops:
                    op = pauli_ops[0]
                    for extra in pauli_ops[1:]:
                        op = op @ extra
                    ops.append(op)
                else:
                    ops.append(qml.Identity(0))
                coeffs.append(coeff)
            group_hamiltonians.append(qml.Hamiltonian(coeffs, ops))

        # ── 3. Create device ──────────────────────────────────────────
        self.device = create_device(self.backend_config)

        # Full Hamiltonian for expectation value
        H_full = self.hamiltonian.to_pennylane()
        insert_noise = self.noise_inserter

        # ── 4. QNode ──────────────────────────────────────────────────
        @qml.qnode(self.device)
        def circuit(params):
            params = params.reshape(n_layers, n_groups)

            self._prepare_initial_state()

            # HVA layers: for each rep, evolve under each group
            for r in range(n_layers):
                for k, H_k in enumerate(group_hamiltonians):
                    theta = params[r, k]
                    qml.CommutingEvolution(H_k, theta)

                # Noise insertion after each full Trotter step
                insert_noise()

            return qml.expval(H_full)

        self.cost_fn = circuit
        logger.info(
            f"HVA ansatz built: {self.n_parameters} params "
            f"({n_groups} groups x {n_layers} layers), "
            f"backend={self.backend_config.label}"
        )
        return circuit

    # ------------------------------------------------------------------
    # Cost / init
    # ------------------------------------------------------------------

    def cost_function(self, parameters: np.ndarray) -> float:
        if self.cost_fn is None:
            raise RuntimeError("Must call build_ansatz() first")
        return float(self.cost_fn(parameters))

    def get_initial_parameters(self) -> np.ndarray:
        """All zeros -> HF reference state (evolutions = identity)."""
        return np.zeros(self.n_parameters)

    # ------------------------------------------------------------------
    # Override run() to use HF energy of truncated system as reference
    # ------------------------------------------------------------------

    def run(self) -> VQEResult:
        """
        Run HVA-VQE.

        Reference energy = HF energy of the truncated system (params=0).
        The optimised energy should be *below* this reference (variational
        principle).
        """
        import time

        logger.info(f"Running {self.name} on {self.hamiltonian.molecule.name}")
        logger.info(f"Backend: {self.backend_config.label}")

        # ── HF verification ──────────────────────────────────────────
        self._perform_hf_verification()

        # ── Progress bar ──────────────────────────────────────────────
        from tqdm import tqdm
        self.progress_bar = tqdm(
            total=self.max_iterations,
            desc=f"{self.name} on {self.hamiltonian.molecule.abbreviation}",
            unit="iter",
        )

        # ── Build ansatz & optimise ───────────────────────────────────
        start_time = time.time()
        self.build_ansatz()
        optimal_params, optimal_energy = self.optimize()
        runtime = time.time() - start_time

        # ── Reference = HF energy (truncated system, params=0) ────────
        ref_energy = self.hamiltonian.molecule.reference_energy
        error = optimal_energy - ref_energy
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
            calculated_energy=optimal_energy,
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
                "n_commuting_groups": len(self._commuting_groups),
            },
            backend_type=self.backend_config.backend_type,
            noise_model=self.backend_config.noise_model,
            noise_strength=self.backend_config.noise_strength,
            hf_energy=self.hf_energy,
        )

        logger.info(
            f"Completed {self.name}: "
            f"Energy={optimal_energy:.8f}, HF ref={ref_energy:.8f}, "
            f"Error={error:.8f}, Runtime={runtime:.2f}s"
        )

        if self.progress_bar:
            self.progress_bar.close()

        if self.hamiltonian.molecule.truncated_ground_state_energy is not None:
            logger.info(
                f"Truncated system ground state: "
                f"{self.hamiltonian.molecule.truncated_ground_state_energy:.8f} Hartree"
            )

        return result
