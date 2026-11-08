# ABOUTME: Selects simulation devices and noise channels for variational circuits.
# ABOUTME: Provides bounded reuse of repeated energies on deterministic analytic devices.
"""
Backend Manager Module

Provides a unified interface for creating PennyLane quantum devices with
support for both **statevector** (exact, noiseless) and **noisy**
(depolarising, bit-flip, amplitude-damping, custom) simulation backends.

Usage:
    from core.backend_manager import BackendConfig, create_device

    # Noiseless statevector
    cfg = BackendConfig.statevector(n_qubits=8)
    dev = create_device(cfg)

    # Noisy simulation
    cfg = BackendConfig.noisy(n_qubits=8, noise_model="depolarizing", noise_strength=0.01)
    dev = create_device(cfg)
"""

import logging
import os
from collections import OrderedDict

import numpy as np
from dataclasses import dataclass, field
from typing import Any, Dict, Optional

logger = logging.getLogger(__name__)


_LIGHTNING_AVAILABLE: Optional[bool] = None


def cached_analytic_energy(circuit):
    """Retain four recent binary parameter vectors for a fixed deterministic scalar circuit."""
    recent = OrderedDict()

    def evaluate(parameters):
        vector = np.asarray(parameters)
        if vector.dtype.hasobject:
            return float(circuit(parameters))
        key = (vector.dtype.str, vector.shape, vector.tobytes())
        if key in recent:
            recent.move_to_end(key)
        else:
            recent[key] = float(circuit(parameters))
            if len(recent) > 4:
                recent.popitem(last=False)
        return recent[key]

    return evaluate


def analytic_energy(circuit, observable=None, *, measure_state=None):
    """Measure a fixed molecular observable with Lightning's original device operations and reduction."""
    import pennylane as qml

    device = circuit.device
    if device.name != "lightning.qubit" or device.shots:
        return circuit
    if tuple(device.wires) != tuple(range(len(device.wires))) or device.c_dtype != np.complex128:
        return circuit

    from pennylane_lightning.core._serialize import QuantumScriptSerializer
    from pennylane_lightning.lightning_qubit_ops import MeasurementsC128

    # Keep the QNode's device-level decomposition, including its adjoint-specific gate expansions.
    state = device._statevector
    measurement = MeasurementsC128(state.state_vector)
    serialized_observable = serialized = None
    rotation_layout = getattr(circuit.func, "rotation_layout", None)
    if measure_state is None:
        measure_state = getattr(circuit.func, "state_measurement", None)
    rotations = None

    def evaluate(parameters):
        nonlocal serialized, serialized_observable, rotations
        if device.tracker.active or type(parameters) is not np.ndarray or parameters.ndim != 1:
            return circuit(parameters)
        layout = None
        if (rotation_layout and parameters.dtype == np.float64 and circuit.device is device
                and not circuit.transform_program and circuit.expansion_strategy == "gradient"
                and circuit.diff_method in (None, "best") and circuit.interface in (None, "auto")
                and getattr(circuit.func, "rotation_layout", None) is rotation_layout):
            reference = rotation_layout()
            if reference is not None:
                layout = (reference, parameters.size, circuit.diff_method, circuit.interface)
        if rotations is not None and layout == rotations[0]:
            _, template, rotation_indices, finish = rotations
            tape = template.bind_new_parameters(parameters, rotation_indices)
            circuit._tape = tape
            circuit._qfunc_output = tape.measurements[0]
        else:
            circuit.construct((parameters,), {})
            circuit._update_gradient_fn(tape=circuit.qtape)
            program = qml.workflow.get_transform_program(circuit, level="device")
            tapes, finish = program((circuit.qtape,))
            if len(tapes) != 1 or len(tapes[0].measurements) != 1:
                return circuit(parameters)
            tape = tapes[0]
            if layout is not None and tape is circuit.qtape and not tape.trainable_params:
                # The rotation-layout contract uses each vector element in one native RX, RY or RZ, in order.
                indices = [i for i, p in enumerate(tape.par_info) if type(p["op"]) in (qml.RX, qml.RY, qml.RZ)]
                if (len(indices) == len(parameters) and
                        all(tape.par_info[i]["op"].data[0].tobytes() == x.tobytes()
                            for i, x in zip(indices, parameters))):
                    rotations = layout, tape, indices, finish
        terminal = tape.measurements[0]
        if (type(terminal) is not qml.measurements.ExpectationMP
                or not isinstance(terminal.obs, qml.ops.LinearCombination)
                or (observable is not None and terminal.obs is not observable)):
            return circuit(parameters)
        if serialized_observable is not terminal.obs:
            serialized = QuantumScriptSerializer(device.name, False)._ob(terminal.obs)
            serialized_observable = terminal.obs
        state.reset_state()
        state.get_final_state(tape)
        if measure_state is None:
            energy = measurement.expval(serialized)
        else:
            energy = measure_state(state.state, measurement.expval, serialized)
        return finish((energy,))[0]

    return evaluate


def fixed_circuit_energy(circuit, observable):
    """Reuse deterministic energies of a fixed statevector ansatz across optimizer callbacks."""
    device = circuit.device
    if device.shots or device.name not in ("default.qubit", "lightning.qubit"):
        return circuit
    energy = analytic_energy(circuit, observable)
    if device.name == "default.qubit":
        return cached_analytic_energy(energy)
    from core.pauli_measurement import lightning_thread_count

    threads = lightning_thread_count()
    if threads is None or os.environ.get("OMP_NUM_THREADS") != "1":
        return energy
    cached = cached_analytic_energy(energy)
    return lambda parameters: cached(parameters) if threads() == 1 else energy(parameters)


def _detect_lightning_qubit() -> bool:
    """Detect whether ``lightning.qubit`` can be instantiated.

    Cached after first check to avoid repeated plugin probing.
    """
    global _LIGHTNING_AVAILABLE
    if _LIGHTNING_AVAILABLE is not None:
        return _LIGHTNING_AVAILABLE

    try:
        import pennylane as qml

        qml.device("lightning.qubit", wires=1)
        _LIGHTNING_AVAILABLE = True
    except Exception as exc:
        _LIGHTNING_AVAILABLE = False
        logger.info("lightning.qubit unavailable, using default.qubit (%s)", exc)

    return _LIGHTNING_AVAILABLE


def _preferred_statevector_device() -> str:
    return "lightning.qubit" if _detect_lightning_qubit() else "default.qubit"


# ── Noise helpers ─────────────────────────────────────────────────────────

def _add_noise_to_circuit(noise_model: str, noise_strength: float, n_qubits: int):
    """
    Return a PennyLane noise-insertion function suitable for use as a
    ``qml.transforms`` or manual insertion after each gate layer.

    The returned callable takes no arguments and applies the noise channel
    to every qubit.
    """
    import pennylane as qml

    def apply_noise():
        for q in range(n_qubits):
            if noise_model == "depolarizing":
                qml.DepolarizingChannel(noise_strength, wires=q)
            elif noise_model == "bitflip":
                qml.BitFlip(noise_strength, wires=q)
            elif noise_model == "phaseflip":
                qml.PhaseFlip(noise_strength, wires=q)
            elif noise_model == "amplitude_damping":
                qml.AmplitudeDamping(noise_strength, wires=q)
            elif noise_model == "phase_damping":
                qml.PhaseDamping(noise_strength, wires=q)
            else:
                raise ValueError(
                    f"Unknown noise model: {noise_model}. "
                    f"Supported: depolarizing, bitflip, phaseflip, "
                    f"amplitude_damping, phase_damping"
                )

    return apply_noise


# ── Configuration dataclass ───────────────────────────────────────────────

@dataclass
class BackendConfig:
    """
    Immutable configuration object describing *how* to create a PennyLane
    quantum device.

    Attributes:
        backend_type:    "statevector" or "noisy".
        device_name:     PennyLane device name (e.g. "lightning.qubit",
                         "default.mixed").
        n_qubits:        Number of qubits.
        n_shots:         Number of measurement shots.  0 → analytic.
        noise_model:     Noise channel name (only used when backend_type
                         == "noisy").
        noise_strength:  Single-qubit error probability / strength.
        extra:           Any additional keyword arguments forwarded to
                         ``qml.device()``.
    """

    backend_type: str = "statevector"
    device_name: str = "default.qubit"  # or "lightning.qubit" for statevector, "default.mixed" for noisy
    n_qubits: int = 0
    n_shots: int = 0
    noise_model: Optional[str] = None
    noise_strength: float = 0.0
    extra: Dict[str, Any] = field(default_factory=dict)

    # ── Convenience constructors ──────────────────────────────────────

    @classmethod
    def statevector(
        cls,
        n_qubits: int,
        device_name: Optional[str] = None,
        n_shots: int = 0,
        **extra,
    ) -> "BackendConfig":
        """Create a noiseless statevector configuration."""
        chosen_device = device_name or _preferred_statevector_device()
        return cls(
            backend_type="statevector",
            device_name=chosen_device,
            n_qubits=n_qubits,
            n_shots=n_shots,
            extra=extra,
        )

    @classmethod
    def noisy(
        cls,
        n_qubits: int,
        noise_model: str = "depolarizing",
        noise_strength: float = 0.01,
        device_name: str = "default.mixed",
        n_shots: int = 0,
        **extra,
    ) -> "BackendConfig":
        """Create a noisy-simulation configuration."""
        return cls(
            backend_type="noisy",
            device_name=device_name,
            n_qubits=n_qubits,
            n_shots=n_shots,
            noise_model=noise_model,
            noise_strength=noise_strength,
            extra=extra,
        )

    # ── Serialisation ─────────────────────────────────────────────────

    def to_dict(self) -> Dict[str, Any]:
        """Serialise to a plain dict (JSON-safe)."""
        return {
            "backend_type": self.backend_type,
            "device_name": self.device_name,
            "n_qubits": self.n_qubits,
            "n_shots": self.n_shots,
            "noise_model": self.noise_model,
            "noise_strength": self.noise_strength,
        }

    @property
    def label(self) -> str:
        """Short human-readable label for display in plots / CSV."""
        if self.backend_type == "noisy":
            base = f"noisy({self.noise_model}, p={self.noise_strength})"
        else:
            base = "statevector"
        if self.n_shots > 0:
            return f"{base}, shots={self.n_shots}"
        return base

    @property
    def is_noisy(self) -> bool:
        return self.backend_type == "noisy"


# ── Device factory ────────────────────────────────────────────────────────

def create_device(config: BackendConfig):
    """
    Instantiate and return a PennyLane device from a ``BackendConfig``.

    For noisy configs the device is ``default.mixed`` (density-matrix
    simulator).  The caller is responsible for inserting noise channels
    into the circuit — use ``get_noise_inserter()`` to obtain a callable
    that does this.
    """
    import pennylane as qml

    shots = config.n_shots if config.n_shots > 0 else None

    device_name = config.device_name
    try:
        if device_name == "default.mixed" and config.is_noisy and shots is None and not config.extra:
            from core.mixed_measurements import GroupedMixedState

            dev = qml.devices.LegacyDeviceFacade(GroupedMixedState(wires=config.n_qubits))
        else:
            dev = qml.device(
                device_name,
                wires=config.n_qubits,
                shots=shots,
                **config.extra,
            )
    except Exception as exc:
        if device_name == "lightning.qubit":
            logger.warning(
                "Failed to create lightning.qubit device (%s). Falling back to default.qubit.",
                exc,
            )
            device_name = "default.qubit"
            dev = qml.device(
                device_name,
                wires=config.n_qubits,
                shots=shots,
                **config.extra,
            )
        else:
            raise

    logger.info(
        f"Created device: {device_name} | "
        f"n_qubits={config.n_qubits} | shots={shots} | "
        f"type={config.backend_type}"
        + (f" | noise={config.noise_model}(p={config.noise_strength})"
           if config.is_noisy else "")
    )

    return dev


def get_noise_inserter(config: BackendConfig):
    """
    Return a callable that inserts noise channels on all qubits.

    Call this *after each layer* inside your QNode to simulate gate noise.
    If the config is statevector (no noise), returns a no-op.
    """
    if not config.is_noisy or config.noise_model is None:
        return lambda: None  # no-op

    return _add_noise_to_circuit(
        config.noise_model, config.noise_strength, config.n_qubits
    )


# ── Supported values (for CLI / validation) ──────────────────────────────

SUPPORTED_BACKEND_TYPES = ["statevector", "noisy"]

SUPPORTED_NOISE_MODELS = [
    "depolarizing",
    "bitflip",
    "phaseflip",
    "amplitude_damping",
    "phase_damping",
]

SUPPORTED_DEVICES = {
    "statevector": ["lightning.qubit", "default.qubit"],
    "noisy": ["default.mixed"],
}
