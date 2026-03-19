"""
VQE-UCC (UCCSD) Implementation

Implements the Unitary Coupled Cluster Singles and Doubles (UCCSD) ansatz,
following the SPUCC symmetry-optimised variant described in:

    Guo Q and Chen P-X (2021) "Optimization of VQE-UCC Algorithm Based on
    Spin State Symmetry." Front. Phys. 9:735321.
    doi: 10.3389/fphy.2021.735321

The UCCSD trial state is:

    |psi(theta)> = exp(T(theta) - T†(theta)) |HF>

where the cluster operator T = T1 + T2 contains single- and double-excitation
terms that map electrons from occupied to virtual orbitals:

    T1 = sum_{i in occ, a in virt} t_ia (a†_a a_i)
    T2 = sum_{i<j in occ, a<b in virt} t_ijab (a†_a a†_b a_j a_i)

Each term is implemented as a Givens rotation via PennyLane's built-in gates:
    qml.SingleExcitation(theta, wires=[i, a])
    qml.DoubleExcitation(theta, wires=[i, j, b, a])

Key property: at theta=0 all gates are the identity, so the circuit outputs
the Hartree-Fock state and cost_function(zeros) == HF energy.

Spin-symmetry (SPUCC) optimisation: singles are paired by spin
(alpha, beta same amplitude) and doubles are restricted to pair excitations,
reducing the parameter count from O(N^4) to O(N^2) while maintaining
near-UCCSD accuracy for singlet ground states.
"""

import numpy as np
from itertools import combinations
from typing import Optional, Any, List, Tuple
import logging

import sys
sys.path.append('..')
from core.base_vqe import BaseVQE, VQEResult
from core.hamiltonian_loader import QubitHamiltonian

logger = logging.getLogger(__name__)


def _generate_uccsd_excitations(
    n_electrons: int,
    n_qubits: int,
    use_spin_symmetry: bool = True,
) -> Tuple[List[Tuple[int, int]], List[Tuple[int, int, int, int]]]:
    """
    Generate singles and doubles excitation indices for UCCSD.

    With spin-symmetry (SPUCC mode):
        Assumes Jordan-Wigner encoding with alternating spin ordering:
            qubit 2k   = spin-up of spatial orbital k
            qubit 2k+1 = spin-down of spatial orbital k
        Only spin-preserving excitations are included.
        Pair doubles: from spin-up/down pair of occupied orbital i
                      to spin-up/down pair of virtual orbital a.

    Without spin-symmetry: all occupied->virtual singles and doubles.

    Args:
        n_electrons: Number of electrons (occupied qubits under JW).
        n_qubits: Total number of qubits.
        use_spin_symmetry: If True, use SPUCC pairing (fewer params).

    Returns:
        (singles, doubles) where
            singles = list of (i, a) pairs
            doubles = list of (i, j, a, b) 4-tuples  (i<j, a<b in their sets)
    """
    occ = list(range(n_electrons))
    virt = list(range(n_electrons, n_qubits))

    if not use_spin_symmetry or n_qubits < 4:
        # ── Full UCCSD ───────────────────────────────────────────────
        singles = [(i, a) for i in occ for a in virt]
        doubles = [
            (i, j, a, b)
            for (i, j) in combinations(occ, 2)
            for (a, b) in combinations(virt, 2)
        ]
        return singles, doubles

    # ── SPUCC: spin-symmetric excitations ───────────────────────────
    # Paired singles: excite spin-up AND spin-down together with one param
    # Qubit ordering: 0↑, 1↓, 2↑, 3↓, ...  (or any arbitrary ordering - we
    # use the simpler "all occ --> virt" approach but pair by alpha/beta)

    # Even-indexed qubits = alpha spin, odd = beta spin (JW ordering)
    occ_alpha = [i for i in occ if i % 2 == 0]
    occ_beta  = [i for i in occ if i % 2 == 1]
    virt_alpha = [a for a in virt if a % 2 == 0]
    virt_beta  = [a for a in virt if a % 2 == 1]

    # Singlet singles: one parameter for each (occ_alpha, virt_alpha) pair
    # and its beta mirror — handled as two separate SingleExcitation gates
    # sharing the same angle.  Return the alpha pairs; beta pairs are derived.
    singles_alpha = [(i, a) for i in occ_alpha for a in virt_alpha]
    singles_beta  = [(i + 1, a + 1) for (i, a) in singles_alpha
                     if (i + 1) in occ_beta and (a + 1) in virt_beta]

    # Pair doubles: (i_alpha, i_beta) → (a_alpha, a_beta)
    # i.e., excite a pair of electrons from spatial orbital i to spatial orbital a
    occ_orbs  = [k for k in range(n_qubits // 2) if 2 * k in occ and 2 * k + 1 in occ]
    virt_orbs = [k for k in range(n_qubits // 2) if 2 * k in virt and 2 * k + 1 in virt]
    pair_doubles = [
        (2 * i, 2 * i + 1, 2 * a, 2 * a + 1)
        for i in occ_orbs
        for a in virt_orbs
    ]

    # Fall back to full UCCSD if symmetry pairing gave nothing
    if not singles_alpha and not pair_doubles:
        singles = [(i, a) for i in occ for a in virt]
        doubles = [
            (i, j, a, b)
            for (i, j) in combinations(occ, 2)
            for (a, b) in combinations(virt, 2)
        ]
        return singles, doubles

    return singles_alpha, pair_doubles


# ---------------------------------------------------------------------------
# UCCSD VQE class
# ---------------------------------------------------------------------------

class UCCSDVariationalVQE(BaseVQE):
    """
    VQE with UCCSD (Unitary Coupled Cluster Singles and Doubles) ansatz.

    Ansatz: |psi(θ)⟩ = prod_k G_k(θ_k) |HF⟩

    where each G_k is either a SingleExcitation or DoubleExcitation Givens
    rotation gate.  At θ=0 all gates are the identity, so the circuit starts
    at the HF reference state.

    An optional spin-symmetry reduction (SPUCC) pairs alpha and beta spin
    excitations under a shared amplitude, reducing O(N^4) → O(N^2) parameters.

    References:
        Guo Q and Chen P-X (2021) Front. Phys. 9:735321.
        doi: 10.3389/fphy.2021.735321
    """

    def __init__(
        self,
        hamiltonian: QubitHamiltonian,
        use_spin_symmetry: bool = True,
        **kwargs,
    ):
        """
        Args:
            hamiltonian:       QubitHamiltonian object.
            use_spin_symmetry: Use SPUCC spin pairing (default True, fewer params).
            **kwargs:          Passed to BaseVQE.
        """
        super().__init__(hamiltonian, **kwargs)

        self.name = "ucc_vqe"
        self.description = (
            "UCCSD VQE (SPUCC spin-symmetry variant, "
            "Guo & Chen, Front. Phys. 2021)"
        )
        self.use_spin_symmetry = use_spin_symmetry

        # Effective n_electrons for the (possibly truncated) active space.
        n_el_raw = self.hamiltonian.molecule.n_electrons or self.n_qubits // 2
        if n_el_raw >= self.n_qubits:
            self._eff_n_electrons = max(1, self.n_qubits // 2)
        else:
            self._eff_n_electrons = n_el_raw

        # Ensure even number of electrons (paired electrons)
        if self._eff_n_electrons % 2 != 0 and self._eff_n_electrons > 1:
            self._eff_n_electrons -= 1

        # Built during build_ansatz
        self.device = None
        self.cost_fn = None
        self._singles: List[Tuple[int, int]] = []
        self._doubles: List[Tuple[int, int, int, int]] = []
        self._beta_singles: List[Tuple[int, int]] = []

    # ------------------------------------------------------------------
    # HF verification with effective n_electrons
    # ------------------------------------------------------------------

    def _perform_hf_verification(self) -> None:
        """Compute HF energy using effective active-space electron count."""
        from core.hf_verification import compute_hf_energy
        try:
            self.hf_energy = compute_hf_energy(
                self.hamiltonian, n_electrons=self._eff_n_electrons
            )
            logger.info(
                f"HF energy (n_el={self._eff_n_electrons}) = "
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
        Build the UCCSD circuit.

        1. Prepare the HF state (fill first n_electrons qubits).
        2. Apply SingleExcitation gates for all (occ → virt) pairs.
        3. Apply DoubleExcitation gates for all (occ,occ → virt,virt) tuples.
        4. Measure ⟨H⟩.

        In SPUCC mode, alpha-spin singles have a paired beta-spin gate with
        the same parameter; pair doubles excite both spin components together.
        """
        import pennylane as qml
        from core.backend_manager import create_device

        n_qubits = self.n_qubits
        n_electrons = self._eff_n_electrons

        # Generate excitation indices
        singles, doubles = _generate_uccsd_excitations(
            n_electrons, n_qubits, self.use_spin_symmetry
        )
        self._singles = singles
        self._doubles = doubles

        # In SPUCC mode, derive paired beta singles from alpha singles
        if self.use_spin_symmetry:
            occ_beta  = [i for i in range(n_electrons) if i % 2 == 1]
            virt_beta = [a for a in range(n_electrons, n_qubits) if a % 2 == 1]
            self._beta_singles = [
                (i + 1, a + 1) for (i, a) in singles
                if (i + 1) in occ_beta and (a + 1) in virt_beta
            ]
        else:
            self._beta_singles = []

        # Total parameters = n_singles (one per alpha single in SPUCC) + n_doubles
        self.n_parameters = len(singles) + len(doubles)

        if self.n_parameters == 0:
            # Safety: fall back to at least one parameter
            logger.warning(
                f"No UCCSD excitations generated for n_el={n_electrons}, "
                f"n_qubits={n_qubits}. Using full UCCSD."
            )
            singles = [(i, a) for i in range(n_electrons)
                       for a in range(n_electrons, n_qubits)]
            doubles = [
                (i, j, a, b)
                for (i, j) in combinations(range(n_electrons), 2)
                for (a, b) in combinations(range(n_electrons, n_qubits), 2)
            ]
            self._singles = singles
            self._doubles = doubles
            self._beta_singles = []
            self.n_parameters = max(1, len(singles) + len(doubles))

        logger.info(
            f"UCCSD: n_electrons={n_electrons}, n_qubits={n_qubits}, "
            f"singles={len(singles)}, doubles={len(doubles)}, "
            f"beta_singles={len(self._beta_singles)}, "
            f"use_spin_symmetry={self.use_spin_symmetry}"
        )

        self.device = create_device(self.backend_config)

        H_full = self.hamiltonian.to_pennylane()
        insert_noise = self.noise_inserter
        _singles = self._singles
        _doubles = self._doubles
        _beta_singles = self._beta_singles
        _n_singles = len(singles)

        @qml.qnode(self.device)
        def circuit(params):
            # ── 1. Initial state (HF or CS-rotated HF if CS reduction applied) ──
            self._prepare_initial_state()

            # ── 2. Single excitations ─────────────────────────────────
            for k, (i, a) in enumerate(_singles):
                qml.SingleExcitation(params[k], wires=[i, a])
                # In SPUCC mode the beta mirror shares the same angle
            for (i, a) in _beta_singles:
                # Find matching alpha pair to get its parameter index
                try:
                    k = _singles.index((i - 1, a - 1))
                    qml.SingleExcitation(params[k], wires=[i, a])
                except ValueError:
                    pass

            # ── 3. Double excitations ─────────────────────────────────
            for m, (i, j, a, b) in enumerate(_doubles):
                # PennyLane's DoubleExcitation expects wires=[s0, s1, d0, d1]
                # where s0<s1 are "source" (occupied) and d0<d1 are "target" (virtual)
                qml.DoubleExcitation(params[_n_singles + m], wires=[i, j, a, b])

            insert_noise()
            return qml.expval(H_full)

        self.cost_fn = circuit
        logger.info(
            f"UCCSD ansatz built: {self.n_parameters} params, "
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
        """All-zeros → HF reference (all excitation gates = identity)."""
        return np.zeros(self.n_parameters)

    # ------------------------------------------------------------------
    # Override run() to use HF energy as reference
    # ------------------------------------------------------------------

    def run(self) -> VQEResult:
        """
        Run UCCSD-VQE.

        Reference energy = HF energy of the truncated system (params=0).
        The optimised energy should be *below* this reference.
        """
        import time

        logger.info(f"Running {self.name} on {self.hamiltonian.molecule.name}")
        logger.info(f"Backend: {self.backend_config.label}")

        # ── HF verification ───────────────────────────────────────────
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
                "n_singles": len(self._singles),
                "n_doubles": len(self._doubles),
                "use_spin_symmetry": self.use_spin_symmetry,
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
                f"{self.hamiltonian.molecule.truncated_ground_state_energy:.8f} Ha"
            )

        return result
