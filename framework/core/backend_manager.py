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
from dataclasses import dataclass, field
from typing import Any, Dict, Optional

logger = logging.getLogger(__name__)


_LIGHTNING_AVAILABLE: Optional[bool] = None


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
    device_name: str = "default.qubit" # or "lightning.qubit" for statevector, "default.mixed" for noisy
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
            return f"noisy({self.noise_model}, p={self.noise_strength})"
        return "statevector"

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
