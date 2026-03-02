"""
NN-AE-VQE: Neural Network Autoencoder Variational Quantum Eigensolver
(Extended with configurable architecture search)

Implements and extends the quantum autoencoder VQE topology described in:

    Mesman et al. (2024) "NN-AE-VQE: Neural network parameter prediction
    on autoencoded variational quantum eigensolvers."
    arXiv:2411.15667

Architecture
------------
The circuit has three stages arranged as a **bottleneck**:

    |HF⟩  ──[ Encoder ]──[ Latent Ansatz ]──[ Decoder ]──  ⟨H⟩

1. **Encoder**  (n_enc_layers × rotation-wall + entangling block):
   Parameterised rotations entangle all n_qubits, compressing information
   toward the first n_latent = n_qubits // 2 qubits.

2. **Latent Ansatz**  (n_latent_layers × rotation-wall + latent entanglement):
   Variational optimisation in the compressed subspace.

3. **Decoder**  (n_enc_layers × entangling block + rotation-wall):
   Maps the compressed state back to the full n_qubits space.

Configurable styles
-------------------
encoder_style / decoder_style:
  "ry_brick"    (default)  RY per qubit + brick-work CNOT (even then odd pairs)
  "ry_rz_brick"            RY+RZ per qubit + brick-work CNOT
  "rot_brick"              Rot(phi,theta,omega) per qubit + brick-work CNOT
  "ry_ring"                RY per qubit + ring CNOT (chain + wrap-around)

latent_style:
  "ry_cz"       (default)  RY per latent qubit + CZ chain
  "ry_rz_cnot"             RY+RZ per latent qubit + CNOT chain
  "rot_cnot"               Rot per latent qubit + CNOT chain
  "ry_cz_ry"               RY + CZ chain + RY  (sandwich; 2 params per qubit)

decoder_style:
  "mirror"      (default)  independent params, reversed CNOT order
  "adjoint"                true adjoint of encoder — shares enc params
                           (negated), no extra params; decoder layers run
                           in reverse layer order
  "independent"            independent params, same (forward) CNOT order

init_strategy:
  "zeros"       (default)  all zeros → identity → HF state at t=0
  "random_small"           U(-0.1, 0.1) — breaks symmetry, explores wider
  "hf_warm"                zeros for encoder & decoder, U(-0.1, 0.1) for latent

Parameter count
---------------
For non-adjoint decoder:
    P = n_qubits × enc_rpq × n_enc_layers        (encoder)
      + n_qubits × dec_rpq × n_enc_layers        (decoder; dec_rpq = enc_rpq)
      + n_latent × lat_rpq × n_latent_layers     (latent)

For adjoint decoder:
    P = n_qubits × enc_rpq × n_enc_layers        (encoder — shared with decoder)
      + n_latent × lat_rpq × n_latent_layers     (latent)

Architecture search
-------------------
Use ``NeuralNetworkAutoEncoderVQE.run_architecture_search(hamiltonian, ...)``
to perform a grid search (or random search) over style combinations and report
which configuration achieves the lowest energy.
"""

import itertools
import logging
import sys
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

sys.path.append('..')
from core.base_vqe import BaseVQE, VQEResult
from core.hamiltonian_loader import QubitHamiltonian

logger = logging.getLogger(__name__)


# ── Style registries ──────────────────────────────────────────────────────────

# Rotations-per-qubit-per-layer for each encoder/decoder style
_ENC_ROTS: Dict[str, int] = {
    "ry_brick":    1,   # RY only
    "ry_rz_brick": 2,   # RY + RZ
    "rot_brick":   3,   # Rot(phi, theta, omega)
    "ry_ring":     1,   # RY + ring CNOT
}

# Rotations-per-qubit-per-layer for each latent style
_LAT_ROTS: Dict[str, int] = {
    "ry_cz":      1,   # RY + CZ chain
    "ry_rz_cnot": 2,   # RY + RZ + CNOT chain
    "rot_cnot":   3,   # Rot + CNOT chain
    "ry_cz_ry":   2,   # RY + CZ + RY  (pre+post sandwich)
}

ENCODER_STYLES  = list(_ENC_ROTS.keys())
LATENT_STYLES   = list(_LAT_ROTS.keys())
DECODER_STYLES  = ["mirror", "adjoint", "independent"]
INIT_STRATEGIES = ["zeros", "random_small", "hf_warm"]


# ── Gate-list helpers ─────────────────────────────────────────────────────────

def _cnot_brick_gates(n_q: int) -> List[Tuple[int, int]]:
    """Even-then-odd brick-work CNOT gate list (control, target)."""
    gates: List[Tuple[int, int]] = []
    for q in range(0, n_q - 1, 2):
        gates.append((q, q + 1))
    if n_q > 2:
        for q in range(1, n_q - 1, 2):
            gates.append((q, q + 1))
    return gates


def _cnot_ring_gates(n_q: int) -> List[Tuple[int, int]]:
    """Linear chain + wrap-around ring CNOT gate list."""
    gates = [(q, q + 1) for q in range(n_q - 1)]
    if n_q > 1:
        gates.append((n_q - 1, 0))
    return gates


def _get_enc_cnot_gates(style: str, n_q: int) -> List[Tuple[int, int]]:
    if style == "ry_ring":
        return _cnot_ring_gates(n_q)
    return _cnot_brick_gates(n_q)


# ── Rotation-layer helpers ────────────────────────────────────────────────────

def _apply_single_qubit_rot(qml, style: str, params: np.ndarray, n_q: int) -> None:
    """
    Apply a wall of single-qubit rotations to qubits 0..n_q-1.

    ``params`` must have length ``n_q × _ENC_ROTS[style]``.
    """
    rpq = _ENC_ROTS.get(style, 1)
    for q in range(n_q):
        base = q * rpq
        if rpq == 1:
            qml.RY(params[base], wires=q)
        elif rpq == 2:
            qml.RY(params[base],     wires=q)
            qml.RZ(params[base + 1], wires=q)
        elif rpq == 3:
            # Rot(phi, theta, omega) = RZ(phi) RY(theta) RZ(omega)
            qml.Rot(params[base], params[base + 1], params[base + 2], wires=q)


def _apply_single_qubit_rot_adjoint(
    qml, style: str, params: np.ndarray, n_q: int
) -> None:
    """
    Apply the adjoint (dagger) of _apply_single_qubit_rot.

    RY(θ)† = RY(-θ);  Rot(φ,θ,ω)† = Rot(-ω, -θ, -φ).
    """
    rpq = _ENC_ROTS.get(style, 1)
    for q in range(n_q - 1, -1, -1):
        base = q * rpq
        if rpq == 1:
            qml.RY(-params[base], wires=q)
        elif rpq == 2:
            qml.RZ(-params[base + 1], wires=q)
            qml.RY(-params[base],     wires=q)
        elif rpq == 3:
            qml.Rot(-params[base + 2], -params[base + 1], -params[base], wires=q)


def _apply_latent_rot(qml, style: str, params: np.ndarray, n_lat: int) -> None:
    """
    Apply a single latent ansatz layer.

    ``params`` must have length ``n_lat × _LAT_ROTS[style]``.
    """
    if style == "ry_cz_ry":
        # Pre-rotation
        for q in range(n_lat):
            qml.RY(params[q], wires=q)
        # CZ chain
        for q in range(n_lat - 1):
            qml.CZ(wires=[q, q + 1])
        # Post-rotation
        for q in range(n_lat):
            qml.RY(params[n_lat + q], wires=q)

    elif style == "ry_cz":
        for q in range(n_lat):
            qml.RY(params[q], wires=q)
        for q in range(n_lat - 1):
            qml.CZ(wires=[q, q + 1])

    elif style == "ry_rz_cnot":
        rpq = 2
        for q in range(n_lat):
            qml.RY(params[q * rpq],     wires=q)
            qml.RZ(params[q * rpq + 1], wires=q)
        for q in range(n_lat - 1):
            qml.CNOT(wires=[q, q + 1])

    elif style == "rot_cnot":
        rpq = 3
        for q in range(n_lat):
            qml.Rot(
                params[q * rpq],
                params[q * rpq + 1],
                params[q * rpq + 2],
                wires=q,
            )
        for q in range(n_lat - 1):
            qml.CNOT(wires=[q, q + 1])


class NeuralNetworkAutoEncoderVQE(BaseVQE):
    """
    VQE with a configurable quantum autoencoder (bottleneck) circuit topology.

    The circuit compresses the n-qubit HF state into a latent subspace of
    n_latent = n_qubits // 2 qubits, applies a variational ansatz in that
    compressed space, then decodes back to the full space for Hamiltonian
    measurement.

    All encoder/latent/decoder styles, decoder coupling strategy, and
    parameter-initialisation scheme are configurable.  See the module
    docstring for the full option matrix.

    References
    ----------
    Mesman et al. (2024) arXiv:2411.15667
    """

    def __init__(
        self,
        hamiltonian: QubitHamiltonian,
        n_enc_layers:    int = 1,
        n_latent_layers: int = 2,
        encoder_style:   str = "ry_brick",
        latent_style:    str = "ry_cz",
        decoder_style:   str = "mirror",
        init_strategy:   str = "zeros",
        **kwargs,
    ):
        """
        Args:
            hamiltonian:     QubitHamiltonian object.
            n_enc_layers:    Number of encoder/decoder repetition blocks.
            n_latent_layers: Number of variational layers in the latent space.
            encoder_style:   Encoder rotation+entangling style.
                             One of: "ry_brick", "ry_rz_brick", "rot_brick", "ry_ring".
            latent_style:    Latent ansatz style.
                             One of: "ry_cz", "ry_rz_cnot", "rot_cnot", "ry_cz_ry".
            decoder_style:   Decoder topology.
                             One of: "mirror", "adjoint", "independent".
            init_strategy:   Parameter initialisation strategy.
                             One of: "zeros", "random_small", "hf_warm".
            **kwargs:        Forwarded to BaseVQE.
        """
        super().__init__(hamiltonian, **kwargs)

        # Validate
        if encoder_style not in ENCODER_STYLES:
            raise ValueError(f"encoder_style must be one of {ENCODER_STYLES}, got '{encoder_style}'")
        if latent_style not in LATENT_STYLES:
            raise ValueError(f"latent_style must be one of {LATENT_STYLES}, got '{latent_style}'")
        if decoder_style not in DECODER_STYLES:
            raise ValueError(f"decoder_style must be one of {DECODER_STYLES}, got '{decoder_style}'")
        if init_strategy not in INIT_STRATEGIES:
            raise ValueError(f"init_strategy must be one of {INIT_STRATEGIES}, got '{init_strategy}'")

        self.name = "nn_ae_vqe"
        self.description = (
            "Neural Network Autoencoder VQE (Mesman et al., arXiv:2411.15667) — "
            f"enc={encoder_style}, lat={latent_style}, dec={decoder_style}, "
            f"init={init_strategy}"
        )
        self.n_enc_layers    = n_enc_layers
        self.n_latent_layers = n_latent_layers
        self.encoder_style   = encoder_style
        self.latent_style    = latent_style
        self.decoder_style   = decoder_style
        self.init_strategy   = init_strategy

        # Latent dimension: half the qubits, at least 1
        self.n_latent = max(1, self.n_qubits // 2)

        # Effective electron count for HF init
        n_el_raw = self.hamiltonian.molecule.n_electrons or self.n_qubits // 2
        self._eff_n_electrons = (
            max(1, self.n_qubits // 2) if n_el_raw >= self.n_qubits else n_el_raw
        )

        # Pre-build gate lists (captured in qnode closure)
        self._enc_cnot_fwd = _get_enc_cnot_gates(encoder_style, self.n_qubits)
        self._enc_cnot_rev = list(reversed(self._enc_cnot_fwd))

        # Parameter block sizes
        enc_rpq = _ENC_ROTS[encoder_style]
        lat_rpq = _LAT_ROTS[latent_style]

        self._enc_block_size = self.n_qubits * enc_rpq
        self._lat_block_size = self.n_latent * lat_rpq
        self._dec_block_size = (
            0 if decoder_style == "adjoint"
            else self.n_qubits * enc_rpq
        )

        self._enc_size = self._enc_block_size * n_enc_layers
        self._lat_size = self._lat_block_size * n_latent_layers
        self._dec_size = self._dec_block_size * n_enc_layers

        self.n_parameters = self._enc_size + self._lat_size + self._dec_size

        self.device  = None
        self.cost_fn = None

    # ------------------------------------------------------------------
    # HF verification
    # ------------------------------------------------------------------

    def _perform_hf_verification(self) -> None:
        """Compute HF energy using effective active-space electron count."""
        from core.hf_verification import compute_hf_energy
        try:
            self.hf_energy = compute_hf_energy(
                self.hamiltonian, n_electrons=self._eff_n_electrons
            )
            logger.info(
                f"HF energy (n_el={self._eff_n_electrons}) = {self.hf_energy:.8f} Ha"
            )
        except Exception as exc:
            logger.warning(f"Could not compute HF energy: {exc}")
            self.hf_energy = None

    # ------------------------------------------------------------------
    # Ansatz construction
    # ------------------------------------------------------------------

    def build_ansatz(self) -> Any:
        """
        Build the configurable autoencoder VQE circuit.

        All parameters at zero give identity rotations → HF state → E_HF
        (for init_strategy="zeros").

        Decoder variants
        ----------------
        "mirror"      : CNOT-reversed, then independent rotation wall.
        "adjoint"     : Layers run in reverse; encoder params negated for each layer.
        "independent" : Same CNOT order as encoder, independent rotation wall.
        """
        import pennylane as qml
        from core.backend_manager import create_device

        n_qubits        = self.n_qubits
        n_latent        = self.n_latent
        n_enc_layers    = self.n_enc_layers
        n_latent_layers = self.n_latent_layers
        encoder_style   = self.encoder_style
        latent_style    = self.latent_style
        decoder_style   = self.decoder_style
        n_el            = self._eff_n_electrons

        enc_block = self._enc_block_size
        lat_block = self._lat_block_size
        dec_block = self._dec_block_size

        enc_cnot_fwd = self._enc_cnot_fwd
        enc_cnot_rev = self._enc_cnot_rev

        self.device  = create_device(self.backend_config)
        H_full       = self.hamiltonian.to_pennylane()
        insert_noise = self.noise_inserter

        # Capture sizes for closure
        enc_size = self._enc_size
        lat_size = self._lat_size

        @qml.qnode(self.device)
        def circuit(params):
            # ── Slice parameter vector ─────────────────────────────────
            p_enc = params[:enc_size]
            p_lat = params[enc_size : enc_size + lat_size]
            p_dec = params[enc_size + lat_size:] if decoder_style != "adjoint" else None

            # ── 1. Hartree-Fock reference state ───────────────────────
            for i in range(n_el):
                qml.PauliX(wires=i)

            # ── 2. Encoder ────────────────────────────────────────────
            for layer in range(n_enc_layers):
                enc_lp = p_enc[layer * enc_block : (layer + 1) * enc_block]
                _apply_single_qubit_rot(qml, encoder_style, enc_lp, n_qubits)
                for ctrl, tgt in enc_cnot_fwd:
                    qml.CNOT(wires=[ctrl, tgt])

            # ── 3. Latent ansatz ──────────────────────────────────────
            for layer in range(n_latent_layers):
                lat_lp = p_lat[layer * lat_block : (layer + 1) * lat_block]
                _apply_latent_rot(qml, latent_style, lat_lp, n_latent)

            # ── 4. Decoder ────────────────────────────────────────────
            if decoder_style == "adjoint":
                # True adjoint: reverse layer order; CNOT reversed; rot negated
                for layer in range(n_enc_layers - 1, -1, -1):
                    enc_lp = p_enc[layer * enc_block : (layer + 1) * enc_block]
                    for ctrl, tgt in enc_cnot_rev:
                        qml.CNOT(wires=[ctrl, tgt])
                    _apply_single_qubit_rot_adjoint(
                        qml, encoder_style, enc_lp, n_qubits
                    )

            elif decoder_style == "mirror":
                # Independent params; reversed CNOT topology
                for layer in range(n_enc_layers):
                    dec_lp = p_dec[layer * dec_block : (layer + 1) * dec_block]
                    for ctrl, tgt in enc_cnot_rev:
                        qml.CNOT(wires=[ctrl, tgt])
                    _apply_single_qubit_rot(qml, encoder_style, dec_lp, n_qubits)

            elif decoder_style == "independent":
                # Independent params; same (forward) CNOT topology as encoder
                for layer in range(n_enc_layers):
                    dec_lp = p_dec[layer * dec_block : (layer + 1) * dec_block]
                    _apply_single_qubit_rot(qml, encoder_style, dec_lp, n_qubits)
                    for ctrl, tgt in enc_cnot_fwd:
                        qml.CNOT(wires=[ctrl, tgt])

            insert_noise()
            return qml.expval(H_full)

        self.cost_fn = circuit
        logger.info(
            f"NN-AE-VQE ansatz built: n_qubits={n_qubits}, n_latent={n_latent}, "
            f"enc_layers={n_enc_layers}, lat_layers={n_latent_layers}, "
            f"enc={encoder_style}, lat={latent_style}, dec={decoder_style}, "
            f"n_params={self.n_parameters}, backend={self.backend_config.label}"
        )
        return circuit

    # ------------------------------------------------------------------
    # Cost function & initialisation
    # ------------------------------------------------------------------

    def cost_function(self, parameters: np.ndarray) -> float:
        if self.cost_fn is None:
            raise RuntimeError("Must call build_ansatz() first")
        return float(self.cost_fn(parameters))

    def get_initial_parameters(self) -> np.ndarray:
        """
        Generate initial parameters according to self.init_strategy.

        "zeros"        →  all zeros → identity rotations → HF start
        "random_small" →  U(-0.1, 0.1) — breaks parameter symmetry
        "hf_warm"      →  zeros for encoder & decoder, U(-0.1, 0.1) for latent
        """
        rng = np.random.default_rng(self.random_seed)

        if self.init_strategy == "random_small":
            return rng.uniform(-0.1, 0.1, self.n_parameters)

        elif self.init_strategy == "hf_warm":
            params = np.zeros(self.n_parameters)
            lat_start = self._enc_size
            lat_end   = self._enc_size + self._lat_size
            params[lat_start:lat_end] = rng.uniform(-0.1, 0.1, self._lat_size)
            return params

        else:  # "zeros"
            return np.zeros(self.n_parameters)

    # ------------------------------------------------------------------
    # Override run() to use HF energy as reference baseline
    # ------------------------------------------------------------------

    def run(self) -> VQEResult:
        """
        Run NN-AE-VQE.

        Reference energy = HF energy of active-space system (params=0).
        The optimised energy should be *below* this reference.
        """
        import time
        from tqdm import tqdm

        logger.info(f"Running {self.name} on {self.hamiltonian.molecule.name}")
        logger.info(
            f"Config: enc={self.encoder_style}, lat={self.latent_style}, "
            f"dec={self.decoder_style}, init={self.init_strategy}"
        )
        logger.info(f"Backend: {self.backend_config.label}")

        self._perform_hf_verification()

        self.progress_bar = tqdm(
            total=self.max_iterations,
            desc=(
                f"{self.name} | {self.hamiltonian.molecule.abbreviation} | "
                f"enc={self.encoder_style} lat={self.latent_style} dec={self.decoder_style}"
            ),
            unit="iter",
        )

        start_time = time.time()
        self.build_ansatz()
        optimal_params, optimal_energy = self.optimize()
        runtime = time.time() - start_time

        ref_energy = (
            self.hf_energy
            if self.hf_energy is not None
            else self.hamiltonian.molecule.reference_energy
        )
        error          = optimal_energy - ref_energy
        relative_error = abs(error / ref_energy) if ref_energy != 0 else 0.0

        converged = (
            len(self.convergence_history) > 1
            and abs(self.convergence_history[-1] - self.convergence_history[-2])
            < self.convergence_threshold
        )

        result = VQEResult(
            molecule_abbrev  = self.hamiltonian.molecule.abbreviation,
            molecule_name    = self.hamiltonian.molecule.name,
            algorithm_name   = self.name,
            calculated_energy= optimal_energy,
            reference_energy = ref_energy,
            error            = error,
            relative_error   = relative_error,
            n_iterations     = self.iteration_count,
            n_qubits         = self.n_qubits,
            n_parameters     = self.n_parameters,
            runtime_seconds  = runtime,
            convergence_history = self.convergence_history,
            optimal_parameters  = optimal_params,
            converged        = converged,
            metadata={
                "optimizer":        self.optimizer_name,
                "max_iterations":   self.max_iterations,
                "n_shots":          self.n_shots,
                "random_seed":      self.random_seed,
                "n_latent":         self.n_latent,
                "n_enc_layers":     self.n_enc_layers,
                "n_latent_layers":  self.n_latent_layers,
                "encoder_style":    self.encoder_style,
                "latent_style":     self.latent_style,
                "decoder_style":    self.decoder_style,
                "init_strategy":    self.init_strategy,
            },
            backend_type  = self.backend_config.backend_type,
            noise_model   = self.backend_config.noise_model,
            noise_strength= self.backend_config.noise_strength,
            hf_energy     = self.hf_energy,
        )

        logger.info(
            f"Completed {self.name}: Energy={optimal_energy:.8f}, "
            f"HF ref={ref_energy:.8f}, Error={error:.8f}, Runtime={runtime:.2f}s"
        )
        if self.progress_bar:
            self.progress_bar.close()

        if self.hamiltonian.molecule.truncated_ground_state_energy is not None:
            logger.info(
                f"Truncated system ground state: "
                f"{self.hamiltonian.molecule.truncated_ground_state_energy:.8f} Ha"
            )
        return result

    # ------------------------------------------------------------------
    # Architecture search (grid / random CV)
    # ------------------------------------------------------------------

    @classmethod
    def run_architecture_search(
        cls,
        hamiltonian: "QubitHamiltonian",
        encoder_styles:       Optional[List[str]] = None,
        latent_styles:        Optional[List[str]] = None,
        decoder_styles:       Optional[List[str]] = None,
        init_strategies:      Optional[List[str]] = None,
        n_enc_layers_list:    Optional[List[int]] = None,
        n_latent_layers_list: Optional[List[int]] = None,
        max_configs:          int = 24,
        random_sample:        bool = False,
        random_seed:          int = 42,
        save_csv:             Optional[str] = None,
        **vqe_kwargs,
    ) -> pd.DataFrame:
        """
        Grid (or random) search over NN-AE-VQE architecture hyper-parameters.

        Parameters
        ----------
        hamiltonian          : QubitHamiltonian to optimise against.
        encoder_styles       : List of encoder styles (default: all 4).
        latent_styles        : List of latent styles  (default: all 4).
        decoder_styles       : List of decoder styles (default: all 3).
        init_strategies      : List of init strategies (default: ["zeros","hf_warm"]).
        n_enc_layers_list    : n_enc_layers values  (default: [1, 2]).
        n_latent_layers_list : n_latent_layers values (default: [1, 2]).
        max_configs          : Maximum number of configurations to evaluate.
        random_sample        : If True, randomly sample max_configs from the
                               full Cartesian product instead of the first N.
        random_seed          : Seed for reproducibility.
        save_csv             : Optional path to save results as CSV.
        **vqe_kwargs         : Forwarded to NeuralNetworkAutoEncoderVQE
                               (optimizer, max_iterations, backend_config, …).

        Returns
        -------
        pandas.DataFrame  sorted by calculated_energy (ascending).
        Columns: encoder_style, latent_style, decoder_style, init_strategy,
                 n_enc_layers, n_latent_layers, n_parameters,
                 calculated_energy, reference_energy, error, relative_error,
                 n_iterations, converged, runtime_seconds, status.
        """
        import time

        enc_styles  = encoder_styles  or ENCODER_STYLES
        lat_styles  = latent_styles   or LATENT_STYLES
        dec_styles  = decoder_styles  or DECODER_STYLES
        init_strats = init_strategies or ["zeros", "hf_warm"]
        enc_layers  = n_enc_layers_list    or [1, 2]
        lat_layers  = n_latent_layers_list or [1, 2]

        all_combos = list(itertools.product(
            enc_styles, lat_styles, dec_styles, init_strats,
            enc_layers, lat_layers,
        ))

        rng = np.random.default_rng(random_seed)
        if random_sample and len(all_combos) > max_configs:
            idx    = rng.choice(len(all_combos), size=max_configs, replace=False)
            combos = [all_combos[i] for i in sorted(idx)]
        else:
            combos = all_combos[:max_configs]

        total = len(combos)
        logger.info(
            f"Architecture search: {total} configs on "
            f"{hamiltonian.molecule.name} ({hamiltonian.n_qubits} qubits)"
        )

        records = []
        for i, (enc, lat, dec, init, nel, nll) in enumerate(combos):
            tag = (
                f"[{i+1}/{total}] enc={enc} lat={lat} dec={dec} "
                f"init={init} nel={nel} nll={nll}"
            )
            logger.info(f"Running config {tag}")
            t0 = time.time()
            try:
                vqe = cls(
                    hamiltonian,
                    n_enc_layers    = nel,
                    n_latent_layers = nll,
                    encoder_style   = enc,
                    latent_style    = lat,
                    decoder_style   = dec,
                    init_strategy   = init,
                    **vqe_kwargs,
                )
                result = vqe.run()
                wall   = time.time() - t0
                records.append({
                    "encoder_style":     enc,
                    "latent_style":      lat,
                    "decoder_style":     dec,
                    "init_strategy":     init,
                    "n_enc_layers":      nel,
                    "n_latent_layers":   nll,
                    "n_parameters":      result.n_parameters,
                    "calculated_energy": result.calculated_energy,
                    "reference_energy":  result.reference_energy,
                    "error":             result.error,
                    "relative_error":    result.relative_error,
                    "n_iterations":      result.n_iterations,
                    "converged":         result.converged,
                    "runtime_seconds":   wall,
                    "status":            "ok",
                })
                logger.info(
                    f"  → E={result.calculated_energy:.8f}  "
                    f"err={result.error:+.6f}  {wall:.1f}s"
                )
            except Exception as exc:
                wall = time.time() - t0
                logger.warning(f"  Config {tag} FAILED: {exc}")
                records.append({
                    "encoder_style":     enc,
                    "latent_style":      lat,
                    "decoder_style":     dec,
                    "init_strategy":     init,
                    "n_enc_layers":      nel,
                    "n_latent_layers":   nll,
                    "n_parameters":      None,
                    "calculated_energy": None,
                    "reference_energy":  None,
                    "error":             None,
                    "relative_error":    None,
                    "n_iterations":      None,
                    "converged":         None,
                    "runtime_seconds":   wall,
                    "status":            f"error: {exc}",
                })

        df = pd.DataFrame(records)

        # Sort by energy (ascending); failed runs go to the bottom
        ok   = df[df["status"] == "ok"].sort_values("calculated_energy")
        fail = df[df["status"] != "ok"]
        df   = pd.concat([ok, fail], ignore_index=True)

        if save_csv:
            df.to_csv(save_csv, index=False)
            logger.info(f"Architecture search results saved to {save_csv}")

        if len(ok):
            logger.info(
                f"\nArchitecture search complete.  Best config:\n"
                f"{df.iloc[0].to_string()}"
            )
        else:
            logger.warning("All configurations failed.")

        return df

    # ------------------------------------------------------------------
    # Convenience summary
    # ------------------------------------------------------------------

    def __repr__(self) -> str:
        return (
            f"NeuralNetworkAutoEncoderVQE("
            f"n_qubits={self.n_qubits}, n_latent={self.n_latent}, "
            f"enc={self.encoder_style}, lat={self.latent_style}, "
            f"dec={self.decoder_style}, init={self.init_strategy}, "
            f"n_params={self.n_parameters})"
        )
