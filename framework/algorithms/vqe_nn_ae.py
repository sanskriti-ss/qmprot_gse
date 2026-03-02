"""
NN-AE-VQE: Neural Network Autoencoder Variational Quantum Eigensolver

Implements the quantum autoencoder VQE circuit topology described in:

    Mesman et al. (2024) "NN-AE-VQE: Neural network parameter prediction
    on autoencoded variational quantum eigensolvers."
    arXiv:2411.15667

Architecture
------------
The circuit has three stages arranged as a **bottleneck**:

    |HF⟩  ──[ Encoder ]──[ Latent Ansatz ]──[ Decoder ]──  ⟨H⟩

1. **Encoder** (n_enc_layers repetitions of RY + brick-wall CNOT):
   Parameterised rotations entangle all n_qubits, compressing information
   towards the first n_latent = n_qubits // 2 qubits.

2. **Latent Ansatz** (n_latent_layers hardware-efficient layers on n_latent
   qubits):
   The variational optimisation happens here, in the compressed representation.

3. **Decoder** (n_enc_layers repetitions of reverse-CNOT + RY):
   Mirrors the encoder to map the compressed state back to the full n_qubits
   space prior to measuring ⟨H⟩.

The bottleneck forces the circuit to learn a compact representation of the
ground state, similar to a classical autoencoder.  The paper combines this
topology with a classical neural network that warm-starts the parameters;
in this implementation we initialise all parameters to zero (θ=0 gives the
identity on every gate, so the circuit outputs the HF state and
cost_function(zeros) == E_HF).

Parameter count
---------------
    P = n_qubits * n_enc_layers          (encoder RY angles)
      + n_qubits * n_enc_layers          (decoder RY angles)
      + n_latent * n_latent_layers       (latent ansatz RY angles)

The encoder and decoder RY blocks are distinct (not tied), allowing the
circuit to learn an asymmetric compression.
"""

import numpy as np
from typing import Optional, Any
import logging

import sys
sys.path.append('..')
from core.base_vqe import BaseVQE, VQEResult
from core.hamiltonian_loader import QubitHamiltonian

logger = logging.getLogger(__name__)


class NeuralNetworkAutoEncoderVQE(BaseVQE):
    """
    VQE with a quantum autoencoder (bottleneck) circuit topology.

    The circuit compresses the n-qubit HF state into a latent subspace of
    n_latent = n_qubits // 2 qubits, applies a variational ansatz in that
    compressed space, then decodes back to the full space for Hamiltonian
    measurement.  This mirrors the architecture described in the NN-AE-VQE
    paper (Mesman et al., arXiv:2411.15667).

    At initialisation (all params = 0) every RY gate is identity, so the
    circuit passes through the HF state, giving E_HF as the starting point.

    References:
        Mesman et al. (2024) arXiv:2411.15667
    """

    def __init__(
        self,
        hamiltonian: QubitHamiltonian,
        n_enc_layers: int = 1,
        n_latent_layers: int = 2,
        **kwargs,
    ):
        """
        Args:
            hamiltonian:     QubitHamiltonian object.
            n_enc_layers:    Number of encoder/decoder repetition blocks.
            n_latent_layers: Number of variational layers in the latent space.
            **kwargs:        Passed to BaseVQE.
        """
        super().__init__(hamiltonian, **kwargs)

        self.name = "nn_ae_vqe"
        self.description = (
            "Neural Network Autoencoder VQE (Mesman et al., arXiv:2411.15667)"
        )
        self.n_enc_layers = n_enc_layers
        self.n_latent_layers = n_latent_layers

        # Latent dimension: compress to half the qubits
        self.n_latent = max(1, self.n_qubits // 2)

        # Effective n_electrons for the (possibly truncated) active space.
        n_el_raw = self.hamiltonian.molecule.n_electrons or self.n_qubits // 2
        if n_el_raw >= self.n_qubits:
            self._eff_n_electrons = max(1, self.n_qubits // 2)
        else:
            self._eff_n_electrons = n_el_raw

        # Compute total parameter count
        self.n_parameters = (
            self.n_qubits * n_enc_layers        # encoder RY block
            + self.n_qubits * n_enc_layers      # decoder RY block
            + self.n_latent * n_latent_layers   # latent ansatz RY block
        )

        # Built during build_ansatz
        self.device = None
        self.cost_fn = None

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
        Build the autoencoder VQE circuit.

        Circuit structure (per Mesman et al., adapted for statevector sim):

            [HF init]
            --> [Encoder: n_enc_layers × (RY wall + CNOT brickwork)]
            --> [Latent ansatz: n_latent_layers × (RY on n_latent + CZ chain)]
            --> [Decoder: n_enc_layers × (CNOT brickwork reversed + RY wall)]
            --> expval(H)

        All RY parameters initialised to 0 → identity → HF state → E_HF.
        """
        import pennylane as qml
        from core.backend_manager import create_device

        n_qubits = self.n_qubits
        n_latent = self.n_latent
        n_enc_layers = self.n_enc_layers
        n_latent_layers = self.n_latent_layers
        n_el = self._eff_n_electrons

        # Recompute parameter count in case it changed
        self.n_parameters = (
            n_qubits * n_enc_layers
            + n_qubits * n_enc_layers
            + n_latent * n_latent_layers
        )

        # Slice offsets
        enc_size = n_qubits * n_enc_layers
        latent_size = n_latent * n_latent_layers
        # dec slice starts at enc_size + latent_size

        self.device = create_device(self.backend_config)
        H_full = self.hamiltonian.to_pennylane()
        insert_noise = self.noise_inserter

        @qml.qnode(self.device)
        def circuit(params):
            p_enc = params[:enc_size].reshape(n_enc_layers, n_qubits)
            p_lat = params[enc_size: enc_size + latent_size].reshape(
                n_latent_layers, n_latent
            )
            p_dec = params[enc_size + latent_size:].reshape(n_enc_layers, n_qubits)

            # ── 1. Hartree-Fock reference state ───────────────────────
            for i in range(n_el):
                qml.PauliX(wires=i)

            # ── 2. Encoder ────────────────────────────────────────────
            for layer in range(n_enc_layers):
                # RY wall
                for q in range(n_qubits):
                    qml.RY(p_enc[layer, q], wires=q)
                # Even CNOT pairs
                for q in range(0, n_qubits - 1, 2):
                    qml.CNOT(wires=[q, q + 1])
                # Odd CNOT pairs (if enough qubits)
                if n_qubits > 2:
                    for q in range(1, n_qubits - 1, 2):
                        qml.CNOT(wires=[q, q + 1])

            # ── 3. Latent ansatz (first n_latent qubits only) ─────────
            for layer in range(n_latent_layers):
                for q in range(n_latent):
                    qml.RY(p_lat[layer, q], wires=q)
                for q in range(n_latent - 1):
                    qml.CZ(wires=[q, q + 1])

            # ── 4. Decoder (reverse of encoder) ───────────────────────
            for layer in range(n_enc_layers):
                # Reverse odd CNOT pairs
                if n_qubits > 2:
                    for q in range(n_qubits - 2, 0, -2):
                        qml.CNOT(wires=[q - 1, q])
                # Reverse even CNOT pairs
                for q in range(n_qubits - 2, -1, -2):
                    qml.CNOT(wires=[q, q + 1 if q + 1 < n_qubits else q])
                # RY wall
                for q in range(n_qubits):
                    qml.RY(p_dec[layer, q], wires=q)

            insert_noise()
            return qml.expval(H_full)

        self.cost_fn = circuit
        logger.info(
            f"NN-AE-VQE ansatz built: n_qubits={n_qubits}, "
            f"n_latent={n_latent}, n_enc_layers={n_enc_layers}, "
            f"n_latent_layers={n_latent_layers}, "
            f"n_parameters={self.n_parameters}, "
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
        """All zeros → HF reference state (all RY gates = identity)."""
        return np.zeros(self.n_parameters)

    # ------------------------------------------------------------------
    # Override run() to use HF energy as reference
    # ------------------------------------------------------------------

    def run(self) -> VQEResult:
        """
        Run NN-AE-VQE.

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

        # ── Reference = HF energy ─────────────────────────────────────
        ref_energy = (
            self.hf_energy
            if self.hf_energy is not None
            else self.hamiltonian.molecule.reference_energy
        )
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
                "n_latent": self.n_latent,
                "n_enc_layers": self.n_enc_layers,
                "n_latent_layers": self.n_latent_layers,
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
