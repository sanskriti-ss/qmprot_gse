"""
Shared helpers for ``experiments/*`` modules.

This module centralises three tasks the noise-resilience and barren-plateau
experiments both need:

1. **Loading small qubit Hamiltonians** from the ``datasets2/`` H5 store.

   We re-use the existing ``active_space_truncation`` pipeline (HF -> MP2 ->
   CASCI -> qubit Hamiltonian) and optionally fold in a contextual-subspace
   reduction so we can hit a tiny target qubit count (default: 4 qubits).
   This keeps every experiment cheap enough to iterate on a laptop.

2. **Building a VQE algorithm instance** with a chosen ``BackendConfig``
   (statevector or noisy ``default.mixed``) and a chosen ansatz depth.

3. **Generating reproducible initial parameters** under a few named
   initialisation strategies.  Both experiments need to compare the same
   ansatz initialised in different ways, and the noise-resilience experiment
   needs to use the *same* initial parameters across noise levels so any
   parameter drift is attributable to the noise channel and not to a
   different starting point.

All path handling is done via ``pathlib`` and uses ``framework/datasets2``
explicitly (per project convention) rather than the ``DATASETS_DIR`` env
variable, so these scripts run regardless of what ``DATASETS_DIR`` is set to.
"""

from __future__ import annotations

import logging
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

# ── Make the framework importable regardless of cwd ───────────────────────
_FW_DIR = Path(__file__).resolve().parent.parent
if str(_FW_DIR) not in sys.path:
    sys.path.insert(0, str(_FW_DIR))

# Imports from the framework
from core.backend_manager import BackendConfig  # noqa: E402
from core.hamiltonian_loader import QubitHamiltonian  # noqa: E402
from algorithms import get_algorithm  # noqa: E402

logger = logging.getLogger(__name__)

# Project-standard dataset folder (per project README + user guidance)
DATASETS2_DIR: Path = _FW_DIR / "datasets2"


# ──────────────────────────────────────────────────────────────────────────
# Hamiltonian loading
# ──────────────────────────────────────────────────────────────────────────

@dataclass
class LoadedHamiltonian:
    """Container returned by :func:`load_small_hamiltonian`."""

    hamiltonian: QubitHamiltonian
    molecule: str
    basis: str
    n_qubits_active: int
    n_qubits_final: int
    cs_reduced: bool
    cs_metadata: Optional[Dict[str, Any]] = None
    casci_energy: Optional[float] = None
    hf_energy: Optional[float] = None


def load_small_hamiltonian(
    molecule: str = "water",
    basis: str = "sto-3g",
    cs_target_qubits: Optional[int] = 4,
    cs_dfs_cutoff: float = 30.0,
) -> LoadedHamiltonian:
    """Load a tiny qubit Hamiltonian for a molecule in ``datasets2/``.

    The pipeline is:

    1. Read geometry from ``datasets2/<molecule>/<molecule>.h5``.
    2. Run ``active_space_truncation.run_pipeline`` (HF -> MP2 -> CASCI ->
       Jordan-Wigner qubit Hamiltonian) at the requested ``basis``.
    3. If ``cs_target_qubits`` is provided and the active-space Hamiltonian
       has more qubits than that, apply a contextual-subspace reduction.

    Parameters
    ----------
    molecule:
        Folder name inside ``datasets2/`` (e.g. ``"water"``, ``"gly"``,
        ``"hydrogen"``).
    basis:
        PySCF basis set passed to the pipeline.  ``"sto-3g"`` is recommended
        for the experiments because it keeps the active-space step fast and
        the resulting Hamiltonian small.
    cs_target_qubits:
        Desired qubit count after CS reduction.  Pass ``None`` to skip the
        CS step entirely.
    cs_dfs_cutoff:
        Time budget (seconds) for the greedy DFS inside the CS reduction.
    """
    h5_path = DATASETS2_DIR / molecule / f"{molecule}.h5"
    if not h5_path.is_file():
        available = sorted(p.name for p in DATASETS2_DIR.iterdir() if p.is_dir())
        raise FileNotFoundError(
            f"H5 dataset not found: {h5_path}\nAvailable molecules in "
            f"datasets2/: {available}"
        )

    # 1+2. Active-space pipeline (quiet=True suppresses heavy print output)
    from active_space_truncation.run_pipeline import run_pipeline as _asp

    pipeline = _asp(
        molecule=molecule,
        basis=basis,
        h5_path=str(h5_path),
        quiet=True,
    )
    ham: QubitHamiltonian = pipeline["hamiltonian"].qubit_hamiltonian
    n_active = ham.n_qubits
    casci_energy = pipeline["active_space"].casci_energy
    hf_energy = pipeline["diagnostics"].hf_energy

    # 3. Optional CS reduction
    cs_meta: Optional[Dict[str, Any]] = None
    cs_reduced = False
    if cs_target_qubits is not None and n_active > cs_target_qubits:
        from contextual_subspace.cs_reduction import (
            apply_contextual_subspace_reduction,
        )

        ham, cs_meta = apply_contextual_subspace_reduction(
            ham,
            target_qubits=cs_target_qubits,
            dfs_cutoff_seconds=cs_dfs_cutoff,
        )
        cs_reduced = bool(cs_meta.get("reduced", False))

    logger.info(
        "Loaded %s/%s: active-space %d qubits -> final %d qubits (cs=%s, %d terms)",
        molecule, basis, n_active, ham.n_qubits, cs_reduced, ham.n_terms,
    )

    return LoadedHamiltonian(
        hamiltonian=ham,
        molecule=molecule,
        basis=basis,
        n_qubits_active=n_active,
        n_qubits_final=ham.n_qubits,
        cs_reduced=cs_reduced,
        cs_metadata=cs_meta,
        casci_energy=casci_energy,
        hf_energy=hf_energy,
    )


# ──────────────────────────────────────────────────────────────────────────
# VQE construction
# ──────────────────────────────────────────────────────────────────────────

# Algorithms safe to construct via :func:`build_vqe`.
#
# The first two also satisfy two stronger properties used by the noise and
# barren-plateau experiments:
#   * ``build_ansatz()`` is sufficient to set ``n_parameters`` *before*
#     ``optimize`` is called (so we can sample initial params externally).
#   * ``cost_fn`` is a PennyLane QNode of (params,) -> float so we can
#     differentiate it directly with parameter-shift.
#
# ``adapt_vqe`` does not satisfy those (it grows the ansatz iteratively
# inside its own ``run`` method), but it is safe to *construct* via
# ``build_vqe`` so the trainability / accuracy-vs-parameters experiments
# can use it.
SAFE_ALGORITHMS = ("hardware_efficient_vqe", "pennylane_vqe", "adapt_vqe")


def make_backend_config(
    n_qubits: int,
    noise_model: Optional[str] = None,
    noise_strength: float = 0.0,
) -> BackendConfig:
    """Build a noiseless or noisy ``BackendConfig`` for the given qubit count.

    A noise model of ``None`` (or ``"none"``) yields a noiseless statevector
    config that uses ``default.qubit``; otherwise we fall back to the
    density-matrix simulator ``default.mixed`` and the framework's per-layer
    noise inserter applies the requested PennyLane channel after each layer.
    """
    if noise_model in (None, "none", "") or noise_strength == 0.0:
        return BackendConfig.statevector(n_qubits=n_qubits)
    return BackendConfig.noisy(
        n_qubits=n_qubits,
        noise_model=noise_model,
        noise_strength=noise_strength,
    )


def build_vqe(
    hamiltonian: QubitHamiltonian,
    algorithm: str = "hardware_efficient_vqe",
    n_layers: int = 2,
    backend_config: Optional[BackendConfig] = None,
    optimizer: str = "COBYLA",
    max_iterations: int = 100,
    convergence_threshold: float = 1e-10,
    random_seed: int = 42,
    **algo_kwargs,
):
    """Instantiate a VQE algorithm and call ``build_ansatz`` once.

    After this call ``vqe.cost_fn`` is a usable PennyLane QNode,
    ``vqe.n_parameters`` is set, and ``vqe.optimize(initial_parameters)`` can
    be called directly without going through ``vqe.run()`` (which would
    re-build the ansatz and re-randomise initial parameters).

    ``convergence_threshold`` defaults to ``1e-10`` (much tighter than the
    framework default of ``1e-6``).  COBYLA passes this value through as
    ``rhoend`` and would otherwise terminate after only ~10 iterations on
    these small problems, masking real differences between runs.
    """
    if algorithm not in SAFE_ALGORITHMS:
        logger.warning(
            "Algorithm %r is not in SAFE_ALGORITHMS=%s; experiments may "
            "behave unexpectedly.", algorithm, SAFE_ALGORITHMS,
        )

    AlgorithmCls = get_algorithm(algorithm)
    if backend_config is None:
        backend_config = make_backend_config(hamiltonian.n_qubits)

    vqe = AlgorithmCls(
        hamiltonian,
        n_layers=n_layers,
        optimizer=optimizer,
        max_iterations=max_iterations,
        convergence_threshold=convergence_threshold,
        random_seed=random_seed,
        backend_config=backend_config,
        **algo_kwargs,
    )
    # Build once so ``n_parameters`` and ``cost_fn`` are populated.
    vqe.build_ansatz()
    return vqe


# ──────────────────────────────────────────────────────────────────────────
# Initialisation strategies
# ──────────────────────────────────────────────────────────────────────────

INIT_STRATEGIES = ("random_uniform", "small_random", "near_identity", "zeros")


def sample_initial_parameters(
    n_parameters: int,
    strategy: str = "small_random",
    rng: Optional[np.random.Generator] = None,
) -> np.ndarray:
    """Sample one initial parameter vector for a chosen strategy.

    Strategies
    ----------
    ``random_uniform``
        Uniform on ``[0, 2*pi]`` -- the textbook "naive" hardware-efficient
        initialisation that suffers from barren plateaus at modest depth
        (McClean et al., 2018).
    ``small_random``
        Uniform on ``[-pi/8, pi/8]`` -- moderate Grant et al. (2019) style
        initialisation.
    ``near_identity``
        Uniform on ``[-0.05, 0.05]`` -- "drop into the solution region".
        Because the ansatz prepares the HF reference first, near-zero
        rotations leave the state close to HF, where the cost landscape has
        large gradients.
    ``zeros``
        Identically zero -- exact HF reference for ansatze whose gates are
        identity at theta=0 (HVA, UCC).  For general HW-efficient ansatze
        it is still a fixed deterministic point, useful as a control.
    """
    rng = rng if rng is not None else np.random.default_rng()

    if strategy == "random_uniform":
        return rng.uniform(0.0, 2.0 * np.pi, size=n_parameters)
    if strategy == "small_random":
        return rng.uniform(-np.pi / 8.0, np.pi / 8.0, size=n_parameters)
    if strategy == "near_identity":
        return rng.uniform(-0.05, 0.05, size=n_parameters)
    if strategy == "zeros":
        return np.zeros(n_parameters)
    raise ValueError(
        f"Unknown init strategy {strategy!r}.  Supported: {INIT_STRATEGIES}"
    )


# ──────────────────────────────────────────────────────────────────────────
# Gradients
# ──────────────────────────────────────────────────────────────────────────

def parameter_shift_gradient(
    cost_fn,
    params: np.ndarray,
    indices: Optional[List[int]] = None,
    shift: float = np.pi / 2,
) -> np.ndarray:
    """Evaluate the parameter-shift-rule gradient component-wise.

    For ansatze built from single-rotation gates (``RX``, ``RY``, ``RZ``,
    ``SingleExcitation``, ``DoubleExcitation``, ...), the parameter-shift
    rule with ``shift = pi/2`` gives the *exact* analytic gradient with two
    circuit evaluations per parameter:

    .. math::

        \\partial_k E(\\theta) =
            \\tfrac{1}{2}\\,\\bigl( E(\\theta + \\tfrac{\\pi}{2}\\,e_k)
                                  - E(\\theta - \\tfrac{\\pi}{2}\\,e_k) \\bigr)

    Parameters
    ----------
    cost_fn:
        A PennyLane QNode (or any callable ``(params,) -> float``).
    params:
        Point at which to evaluate the gradient.
    indices:
        Subset of parameter indices to differentiate.  ``None`` means all.
    shift:
        Shift angle (default ``pi/2``).
    """
    params = np.asarray(params, dtype=float)
    if indices is None:
        indices = list(range(len(params)))

    grad = np.zeros(len(indices), dtype=float)
    for j, k in enumerate(indices):
        e_k = np.zeros_like(params)
        e_k[k] = shift
        e_plus = float(cost_fn(params + e_k))
        e_minus = float(cost_fn(params - e_k))
        grad[j] = 0.5 * (e_plus - e_minus)
    return grad


# ──────────────────────────────────────────────────────────────────────────
# Output helpers
# ──────────────────────────────────────────────────────────────────────────

def make_run_dir(
    experiment_name: str,
    molecule: str,
    algorithm: str,
    base_dir: Optional[Path] = None,
) -> Path:
    """Create a timestamped per-run output directory.

    Layout:
        framework/experiments/results/<experiment_name>/<timestamp>_<molecule>_<algorithm>/

    All experiments in this package use this helper so every CSV ends up
    under a date-stamped, experiment-name-stamped folder for easy
    archiving and diffing across runs.
    """
    from datetime import datetime

    base_dir = base_dir or (_FW_DIR / "experiments" / "results" / experiment_name)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    out = base_dir / f"{timestamp}_{molecule}_{algorithm}"
    out.mkdir(parents=True, exist_ok=True)
    return out


# ──────────────────────────────────────────────────────────────────────────
# Cost-evaluation counter
# ──────────────────────────────────────────────────────────────────────────

class CostEvalCounter:
    """Wrap a VQE algorithm so every energy evaluation requested through
    its ``cost_function`` is counted (and its energy recorded).

    Usage::

        counter = CostEvalCounter()
        counter.attach(vqe)
        ...   # run vqe
        n_evals = counter.count
        counter.detach(vqe)

    Why we need this
    ----------------
    Different algorithms report ``n_iterations`` differently.  ADAPT-VQE
    counts *operator additions* (~tens), while hardware-efficient counts
    COBYLA outer iterations (~hundreds).  Each "ADAPT iteration" actually
    issues ``|pool|*2 + ~100`` cost evaluations under the hood, so any
    fair "trainability" comparison must count the actual cost-function
    calls.

    What is counted
    ---------------
    ``count`` is the number of energy evaluations *requested by the
    optimiser / gradient screening* -- the quantity a real device would
    pay for.  ``BaseVQE.callback`` re-evaluates the energy purely for
    bookkeeping after every optimiser step; those calls are tallied
    separately in ``callback_evals`` and are NOT part of ``count``.

    Every counted evaluation's energy is stored in ``energies`` so the
    cost needed to first reach a given accuracy can be read off exactly
    with :meth:`evals_to_reach` (no interpolation).

    Optional early stop
    -------------------
    ``stop_below`` (an absolute energy in Ha) makes the counter raise
    :class:`AccuracyReached` as soon as an evaluation falls at or below
    it, so a budget-limited run can end the moment it hits the target
    accuracy.  Callers catch the exception.
    """

    def __init__(self, stop_below: Optional[float] = None) -> None:
        self.count: int = 0
        self.callback_evals: int = 0
        self.energies: List[float] = []
        self.stop_below = stop_below
        self._orig_cost_function = None
        self._orig_callback = None
        self._in_callback = False
        self._vqe = None

    def attach(self, vqe) -> None:
        if self._vqe is not None:
            raise RuntimeError("CostEvalCounter is already attached")
        self._vqe = vqe
        self.count = 0
        self.callback_evals = 0
        self.energies = []

        self._orig_cost_function = vqe.cost_function
        self._orig_callback = getattr(vqe, "callback", None)

        def counted_cost_function(parameters):
            energy = self._orig_cost_function(parameters)
            if self._in_callback:
                self.callback_evals += 1
                return energy
            self.count += 1
            self.energies.append(float(energy))
            if self.stop_below is not None and energy <= self.stop_below:
                raise AccuracyReached(self.count, float(energy))
            return energy

        vqe.cost_function = counted_cost_function  # type: ignore[assignment]

        if self._orig_callback is not None:
            orig_cb = self._orig_callback

            def flagged_callback(parameters):
                self._in_callback = True
                try:
                    return orig_cb(parameters)
                finally:
                    self._in_callback = False

            vqe.callback = flagged_callback  # type: ignore[assignment]

    def detach(self, vqe) -> None:
        if self._orig_cost_function is not None:
            vqe.cost_function = self._orig_cost_function  # type: ignore[assignment]
        if self._orig_callback is not None:
            vqe.callback = self._orig_callback  # type: ignore[assignment]
        self._vqe = None
        self._orig_cost_function = None
        self._orig_callback = None
        self._in_callback = False

    def running_min(self) -> np.ndarray:
        """Best energy seen after each counted evaluation."""
        if not self.energies:
            return np.array([], dtype=float)
        return np.minimum.accumulate(np.asarray(self.energies, dtype=float))

    def evals_to_reach(self, energy_threshold: float) -> Optional[int]:
        """1-based index of the first evaluation with energy <= threshold,
        or ``None`` if never reached."""
        rm = self.running_min()
        hit = np.flatnonzero(rm <= energy_threshold)
        return int(hit[0]) + 1 if hit.size else None


class AccuracyReached(Exception):
    """Raised by :class:`CostEvalCounter` when ``stop_below`` is met."""

    def __init__(self, n_evals: int, energy: float) -> None:
        super().__init__(f"accuracy target reached after {n_evals} evaluations "
                         f"(E = {energy:.8f} Ha)")
        self.n_evals = n_evals
        self.energy = energy


# ──────────────────────────────────────────────────────────────────────────
# Accuracy thresholds (used by the "cost to reach a fixed accuracy" metric)
# ──────────────────────────────────────────────────────────────────────────

CHEMICAL_ACCURACY_HA: float = 1.6e-3   # 1 kcal/mol ~ 1.6 mHa

# Error thresholds relative to E_CASCI (Ha), loosest -> tightest.
ACCURACY_LADDER_HA: Tuple[float, ...] = (1.0e-1, 1.0e-2, CHEMICAL_ACCURACY_HA)


def threshold_label(threshold_ha: float) -> str:
    """Human-readable label, e.g. ``0.0016 -> '1.6mHa'``."""
    return f"{threshold_ha * 1e3:g}mHa"


# ──────────────────────────────────────────────────────────────────────────
# Parameter-drift metrics (single definition shared by all noise code)
# ──────────────────────────────────────────────────────────────────────────

def wrap_angles(delta: np.ndarray) -> np.ndarray:
    """Map angle differences to (-pi, pi]."""
    return (np.asarray(delta, dtype=float) + np.pi) % (2.0 * np.pi) - np.pi


def parameter_drift_metrics(
    theta_noisy: np.ndarray,
    theta_clean: np.ndarray,
    theta_init: Optional[np.ndarray] = None,
) -> Dict[str, float]:
    """All drift quantities for one noisy run, computed in one place.

    Primary quantity (the one plotted in the paper)::

        param_drift_l2   = || theta_noisy* - theta_clean* ||_2

    i.e. the Euclidean distance between the *two optimised parameter
    vectors* (noisy optimum vs noiseless optimum, both started from the
    same ``theta_init``).  It is NOT a distance from the initial point.

    Companion quantities (so the number is interpretable):

    ``param_drift_rms``
        ``param_drift_l2 / sqrt(n_params)`` -- root-mean-square shift per
        parameter in radians.  Comparable across ansatz sizes.
    ``param_drift_l2_wrapped``
        Same as ``param_drift_l2`` after wrapping each coordinate's
        difference to (-pi, pi]; rotation angles that differ by 2*pi are
        the same gate, so this is the physically meaningful distance.
    ``clean_displacement_l2`` / ``noisy_displacement_l2``
        ``||theta* - theta_init||_2`` for each run: how far each
        optimiser travelled from the shared start.  These are the
        "initial vs final" distances.
    """
    theta_noisy = np.asarray(theta_noisy, dtype=float)
    theta_clean = np.asarray(theta_clean, dtype=float)
    n = max(1, theta_noisy.size)
    l2 = float(np.linalg.norm(theta_noisy - theta_clean))
    out = {
        "param_drift_l2": l2,
        "param_drift_rms": l2 / float(np.sqrt(n)),
        "param_drift_l2_wrapped": float(
            np.linalg.norm(wrap_angles(theta_noisy - theta_clean))
        ),
        "clean_displacement_l2": float("nan"),
        "noisy_displacement_l2": float("nan"),
    }
    if theta_init is not None:
        theta_init = np.asarray(theta_init, dtype=float)
        out["clean_displacement_l2"] = float(np.linalg.norm(theta_clean - theta_init))
        out["noisy_displacement_l2"] = float(np.linalg.norm(theta_noisy - theta_init))
    return out


# ──────────────────────────────────────────────────────────────────────────
# Parameter-count calibration for hardware-efficient VQE
# ──────────────────────────────────────────────────────────────────────────

# Per-qubit per-layer rotation counts as implemented by HardwareEfficientVQE.
_HE_ROT_PER_QUBIT = {"RY": 1, "RY_RZ": 2, "full": 3}


def he_layers_for_target_params(
    n_qubits: int,
    target_params: int,
    rotation_gates: str = "RY",
    minimum_layers: int = 0,
) -> Tuple[int, int]:
    """Pick the smallest ``n_layers`` so the hardware-efficient ansatz has
    at least ``target_params`` parameters; return ``(n_layers, actual_n_params)``.

    HardwareEfficientVQE counts parameters as
    ``n_qubits * rotations_per_qubit * (n_layers + 1)``.  This helper inverts
    that relation so we can match an algorithm-agnostic parameter target
    (e.g. for the trainability comparison with ADAPT-VQE).

    The ``minimum_layers`` argument lets callers force at least one
    entangling layer even for very small targets.
    """
    rpq = _HE_ROT_PER_QUBIT.get(rotation_gates)
    if rpq is None:
        raise ValueError(f"Unknown rotation_gates={rotation_gates!r}; "
                         f"supported: {list(_HE_ROT_PER_QUBIT)}")
    per_layer_block = n_qubits * rpq
    if per_layer_block <= 0:
        raise ValueError("n_qubits and rotation count must be positive")

    # n_params = per_layer_block * (n_layers + 1) >= target_params
    # => n_layers >= ceil(target_params / per_layer_block) - 1
    import math
    n_layers = max(
        minimum_layers,
        max(0, math.ceil(target_params / per_layer_block) - 1),
    )
    return n_layers, per_layer_block * (n_layers + 1)
