"""
Base VQE Class

Abstract base class for all VQE algorithm implementations.
Provides common functionality for running VQE simulations.
"""
import numpy as np
from abc import ABC, abstractmethod
from typing import Dict, List, Optional, Callable, Tuple, Any
from dataclasses import dataclass, field
import time
import logging
from tqdm import tqdm

from .hamiltonian_loader import QubitHamiltonian
from .backend_manager import BackendConfig, create_device, get_noise_inserter
from .hf_verification import compute_hf_energy, verify_hf_energy

logger = logging.getLogger(__name__)


@dataclass
class VQEResult:
    """Data class for VQE results"""
    molecule_abbrev: str
    molecule_name: str
    algorithm_name: str
    calculated_energy: float
    reference_energy: float
    error: float
    relative_error: float
    n_iterations: int
    n_qubits: int
    n_parameters: int
    runtime_seconds: float
    convergence_history: List[float] = field(default_factory=list)
    optimal_parameters: Optional[np.ndarray] = None
    final_gradient_norm: Optional[float] = None
    converged: bool = False
    metadata: Dict[str, Any] = field(default_factory=dict)
    # Backend & noise fields
    backend_type: str = "statevector"
    noise_model: Optional[str] = None
    noise_strength: float = 0.0
    # Hartree-Fock verification
    hf_energy: Optional[float] = None
    
    def to_dict(self) -> Dict:
        """Convert to dictionary for JSON serialization"""
        # Helper function to ensure JSON serializable types
        def ensure_json_serializable(obj):
            import numpy as np
            if isinstance(obj, np.integer):
                return int(obj)
            elif isinstance(obj, np.floating):
                return float(obj)
            elif isinstance(obj, np.ndarray):
                return obj.tolist()
            elif isinstance(obj, list):
                return [ensure_json_serializable(item) for item in obj]
            elif isinstance(obj, dict):
                return {key: ensure_json_serializable(value) for key, value in obj.items()}
            return obj
        
        return {
            "molecule_abbrev": self.molecule_abbrev,
            "molecule_name": self.molecule_name,
            "algorithm_name": self.algorithm_name,
            "calculated_energy": float(self.calculated_energy),
            "reference_energy": float(self.reference_energy),
            "error": float(self.error),
            "relative_error": float(self.relative_error),
            "n_iterations": self.n_iterations,
            "n_qubits": self.n_qubits,
            "n_parameters": self.n_parameters,
            "runtime_seconds": float(self.runtime_seconds),
            "convergence_history": [float(e) for e in self.convergence_history],
            "optimal_parameters": self.optimal_parameters.tolist() if self.optimal_parameters is not None else None,
            "final_gradient_norm": float(self.final_gradient_norm) if self.final_gradient_norm else None,
            "converged": self.converged,
            "metadata": ensure_json_serializable(self.metadata),
            "backend_type": self.backend_type,
            "noise_model": self.noise_model,
            "noise_strength": float(self.noise_strength),
            "hf_energy": float(self.hf_energy) if self.hf_energy is not None else None,
        }


class BaseVQE(ABC):
    """
    Abstract base class for VQE implementations.
    
    All VQE algorithms should inherit from this class and implement
    the required abstract methods.
    """
    
    def __init__(self,
                 hamiltonian: QubitHamiltonian,
                 optimizer: str = "COBYLA",
                 max_iterations: int = 100,
                 convergence_threshold: float = 1e-6,
                 n_shots: int = 0,
                 random_seed: Optional[int] = None,
                 backend_config: Optional[BackendConfig] = None,
                 **kwargs):
        """
        Initialize the VQE solver.
        
        Args:
            hamiltonian: QubitHamiltonian object
            optimizer: Optimization method name
            max_iterations: Maximum optimization iterations
            convergence_threshold: Convergence threshold for energy
            n_shots: Number of measurement shots (0 for exact simulation)
            random_seed: Random seed for reproducibility
            backend_config: BackendConfig for device creation. If None,
                            defaults to statevector (default.qubit).
            **kwargs: Additional algorithm-specific parameters
        """
        self.hamiltonian = hamiltonian
        self.optimizer_name = optimizer
        self.max_iterations = max_iterations
        self.convergence_threshold = convergence_threshold
        self.n_shots = n_shots
        self.random_seed = random_seed
        self.kwargs = kwargs
        
        # Backend configuration
        if backend_config is not None:
            self.backend_config = backend_config
        else:
            self.backend_config = BackendConfig.statevector(
                n_qubits=hamiltonian.n_qubits
            )
        # Ensure n_qubits is consistent
        self.backend_config.n_qubits = hamiltonian.n_qubits
        self.noise_inserter = get_noise_inserter(self.backend_config)
        
        # Set random seed
        if random_seed is not None:
            np.random.seed(random_seed)
        
        # Algorithm info
        self.name: str = "base_vqe"
        self.description: str = "Base VQE implementation"
        
        # Results tracking
        self.convergence_history: List[float] = []
        self.iteration_count: int = 0
        self.optimal_parameters: Optional[np.ndarray] = None
        self.optimal_energy: Optional[float] = None
        self.progress_bar: Optional[tqdm] = None
        
        # Hartree-Fock energy (computed in run())
        self.hf_energy: Optional[float] = None
        
        # Build components
        self.n_qubits = hamiltonian.n_qubits
        self.n_parameters: int = 0
        
    def _prepare_initial_state(self):
        """Prepare the initial state inside a QNode circuit.

        If the Hamiltonian carries a CS-rotated initial state (from contextual
        subspace reduction), use ``qml.StatePrep``.  Otherwise fall back to
        the standard Hartree-Fock preparation (flip first *n_electrons* qubits).
        """
        import pennylane as qml

        cs_state = getattr(self.hamiltonian, "cs_initial_state", None)
        if cs_state is not None and np.linalg.norm(cs_state) > 1e-10:
            qml.StatePrep(cs_state, wires=range(self.n_qubits))
        else:
            if cs_state is not None:
                import logging
                logging.getLogger(__name__).warning(
                    "CS initial state has zero norm; falling back to HF initial state"
                )
            n_electrons = self.hamiltonian.molecule.n_electrons or self.n_qubits // 2
            for i in range(min(n_electrons, self.n_qubits)):
                qml.PauliX(wires=i)

    @abstractmethod
    def build_ansatz(self) -> Any:
        """
        Build the variational ansatz circuit.
        
        Returns:
            Ansatz circuit (format depends on backend)
        """
        pass
    
    @abstractmethod
    def cost_function(self, parameters: np.ndarray) -> float:
        """
        Evaluate the cost function (expectation value of Hamiltonian).
        
        Args:
            parameters: Variational parameters
            
        Returns:
            Energy expectation value
        """
        pass
    
    def get_initial_parameters(self) -> np.ndarray:
        """
        Get initial parameter values.
        Can be overridden for different initialization strategies.
        
        Returns:
            Initial parameter array
        """
        # Default: small random values near zero
        return np.random.uniform(-0.1, 0.1, self.n_parameters)
    
    def callback(self, parameters: np.ndarray):
        """
        Callback function called at each optimization step.
        
        Args:
            parameters: Current parameters
        """
        energy = self.cost_function(parameters)
        self.convergence_history.append(energy)
        self.iteration_count += 1
        
        # Update progress bar
        if self.progress_bar:
            self.progress_bar.set_postfix({'Energy': f'{energy:.6f}'})
            self.progress_bar.update(1)
        
        if self.iteration_count % 10 == 0:
            logger.debug(f"Iteration {self.iteration_count}: Energy = {energy:.8f}")
    
    def _perform_hf_verification(self) -> None:
        """
        Perform Hartree-Fock energy verification.
        
        This method computes the HF energy and logs the results.
        Sets self.hf_energy attribute.
        """
        try:
            self.hf_energy = compute_hf_energy(self.hamiltonian)
            logger.info(
                f"HF energy ⟨HF|H|HF⟩ = {self.hf_energy:.8f} Ha  "
                f"(reference = {self.hamiltonian.molecule.reference_energy:.8f} Ha)"
            )
        except Exception as exc:
            logger.warning(f"Could not compute HF energy: {exc}")
            self.hf_energy = None
    
    def optimize(self, initial_parameters: Optional[np.ndarray] = None) -> Tuple[np.ndarray, float]:
        """
        Run the optimization.
        
        Args:
            initial_parameters: Optional initial parameters
            
        Returns:
            Tuple of (optimal_parameters, optimal_energy)
        """
        from scipy.optimize import minimize
        from tqdm import tqdm

        if initial_parameters is None:
            initial_parameters = self.get_initial_parameters()

        # Reset tracking
        self.convergence_history = []
        self.iteration_count = 0

        optimizer_key = str(self.optimizer_name).strip().upper()
        if optimizer_key in {
            "BAYES",
            "BAYESIAN",
            "BAYESIAN_OPT",
            "BAYESIAN_OPTIMIZATION",
            "GP",
            "GP_MINIMIZE",
        }:
            return self._optimize_bayesian(initial_parameters)

        # Map optimizer names
        scipy_optimizers = {
            "COBYLA": "COBYLA",
            "L-BFGS-B": "L-BFGS-B",
            "SLSQP": "SLSQP",
            "NelderMead": "Nelder-Mead",
            "Powell": "Powell",
        }

        method = scipy_optimizers.get(self.optimizer_name, self.optimizer_name)

        # Wrap callback and cost function for tqdm progress bar
        pbar = tqdm(total=self.max_iterations, desc="VQE Optimization", unit="iter")
        self._last_iter = 0
        def tqdm_callback(params):
            self.callback(params)
            pbar.update(self.iteration_count - self._last_iter)
            self._last_iter = self.iteration_count

        result = minimize(
            self.cost_function,
            initial_parameters,
            method=method,
            callback=tqdm_callback,
            options={
                "maxiter": self.max_iterations,
                "disp": False,
            },
            tol=self.convergence_threshold,
        )
        pbar.close()

        self.optimal_parameters = result.x
        self.optimal_energy = result.fun

        return result.x, result.fun

    def _optimize_bayesian(self, initial_parameters: np.ndarray) -> Tuple[np.ndarray, float]:
        """Run Bayesian optimization using a Gaussian-process surrogate.

        This path requires ``scikit-optimize``. The search space defaults to
        ``[-pi, pi]`` for each parameter and can be overridden by passing
        ``bayes_bounds=(lower, upper)`` via algorithm kwargs.

        For high-dimensional ansatze, a Gaussian-process surrogate can become
        prohibitively expensive as observations accumulate. We therefore support
        ``bayes_backend`` with options ``gp``, ``forest``, ``gbrt``, and
        ``auto`` (default). ``auto`` picks a scalable surrogate based on
        dimensionality.
        """
        from tqdm import tqdm
        import warnings

        try:
            from skopt import gp_minimize, forest_minimize, gbrt_minimize
            from skopt.space import Real
        except ImportError as exc:
            raise ImportError(
                "Bayesian optimization requires scikit-optimize. "
                "Install it with: pip install scikit-optimize"
            ) from exc

        if self.n_parameters <= 0:
            raise ValueError("Cannot run Bayesian optimization with n_parameters <= 0")

        bounds = self.kwargs.get("bayes_bounds", (-np.pi, np.pi))
        if not isinstance(bounds, (tuple, list)) or len(bounds) != 2:
            raise ValueError(
                "bayes_bounds must be a 2-tuple/list like (lower, upper)"
            )
        low, high = float(bounds[0]), float(bounds[1])
        if low >= high:
            raise ValueError("bayes_bounds must satisfy lower < upper")

        theta0 = np.asarray(initial_parameters, dtype=float).reshape(-1)
        if theta0.size != self.n_parameters:
            raise ValueError(
                "Initial parameter size does not match n_parameters: "
                f"{theta0.size} != {self.n_parameters}"
            )

        # If the selected init strategy provides parameters outside the default
        # Bayesian bounds (e.g. random_uniform on [0, 2*pi]), expand bounds so
        # gp_minimize can legally evaluate x0.
        tmin = float(np.min(theta0))
        tmax = float(np.max(theta0))
        if tmin < low or tmax > high:
            new_low = min(low, tmin)
            new_high = max(high, tmax)
            logger.info(
                "Expanding bayes_bounds from [%.4f, %.4f] to [%.4f, %.4f] "
                "to include initial parameters.",
                low,
                high,
                new_low,
                new_high,
            )
            low, high = new_low, new_high

        n_calls = max(5, int(self.max_iterations))
        n_initial_points = min(
            max(10, int(np.sqrt(max(self.n_parameters, 1)) * 4)),
            max(1, n_calls - 1),
        )
        dimensions = [Real(low, high, name=f"theta_{i}") for i in range(self.n_parameters)]

        backend_raw = str(self.kwargs.get("bayes_backend", "auto")).strip().lower()
        if backend_raw in ("auto", ""):
            backend = "forest" if self.n_parameters >= 24 else "gp"
        elif backend_raw in {"gp", "forest", "gbrt"}:
            backend = backend_raw
        else:
            raise ValueError(
                f"Unknown bayes_backend={backend_raw!r}. "
                "Supported: auto, gp, forest, gbrt"
            )

        minimize_fn = {
            "gp": gp_minimize,
            "forest": forest_minimize,
            "gbrt": gbrt_minimize,
        }[backend]

        logger.info(
            "Bayesian optimizer backend=%s, n_params=%d, n_calls=%d, n_initial_points=%d",
            backend,
            self.n_parameters,
            n_calls,
            n_initial_points,
        )

        pbar = tqdm(total=n_calls, desc="VQE Bayesian Optimization", unit="eval")

        def objective(x: List[float]) -> float:
            theta = np.asarray(x, dtype=float)
            energy = float(self.cost_function(theta))
            self.convergence_history.append(energy)
            self.iteration_count += 1

            if self.progress_bar:
                self.progress_bar.set_postfix({'Energy': f'{energy:.6f}'})
                self.progress_bar.update(1)

            pbar.update(1)
            if self.iteration_count % 10 == 0:
                logger.debug(
                    "Bayes eval %d: Energy = %.8f",
                    self.iteration_count,
                    energy,
                )
            return energy

        minimize_kwargs = {
            "dimensions": dimensions,
            "n_calls": n_calls,
            "n_initial_points": n_initial_points,
            "random_state": self.random_seed,
            "x0": theta0.tolist(),
        }
        if backend == "gp":
            minimize_kwargs.update(
                {
                    "acq_func": str(self.kwargs.get("bayes_acq_func", "EI")),
                    "acq_optimizer": str(
                        self.kwargs.get(
                            "bayes_acq_optimizer",
                            "sampling" if self.n_parameters >= 24 else "lbfgs",
                        )
                    ),
                    "n_restarts_optimizer": int(
                        self.kwargs.get("bayes_n_restarts_optimizer", 0)
                    ),
                }
            )

        def _run_minimize(fn):
            # PennyLane 0.38 on Python 3.13 emits this warning once per circuit
            # call; suppress it here to avoid log flooding during optimization.
            with warnings.catch_warnings():
                warnings.filterwarnings(
                    "ignore",
                    category=FutureWarning,
                    message=r"functools\.partial will be a method descriptor in future Python versions.*",
                )
                return fn(objective, **minimize_kwargs)

        try:
            try:
                result = _run_minimize(minimize_fn)
            except Exception as exc:
                if backend == "gbrt":
                    logger.warning(
                        "Bayesian backend=gbrt failed (%s); falling back to forest backend.",
                        exc,
                    )
                    backend = "forest"
                    result = _run_minimize(forest_minimize)
                else:
                    raise
        finally:
            pbar.close()

        self.optimal_parameters = np.asarray(result.x, dtype=float)
        self.optimal_energy = float(result.fun)
        return self.optimal_parameters, self.optimal_energy
    
    def run(self) -> VQEResult:
        """
        Run the full VQE algorithm.
        
        Returns:
            VQEResult with all results and metadata
        """
        logger.info(f"Running {self.name} on {self.hamiltonian.molecule.name}")
        logger.info(f"Backend: {self.backend_config.label}")
        
        # ── Hartree-Fock energy verification ──────────────────────────
        self._perform_hf_verification()
        
        # Initialize progress bar
        self.progress_bar = tqdm(total=self.max_iterations, 
                               desc=f"{self.name} on {self.hamiltonian.molecule.abbreviation}",
                               unit="iter")
        
        # Build ansatz
        start_time = time.time()
        self.build_ansatz()
        
        # Run optimization
        optimal_params, optimal_energy = self.optimize()
        
        runtime = time.time() - start_time
        
        # Calculate errors (always relative to full system reference energy)
        ref_energy = self.hamiltonian.molecule.reference_energy
        error = optimal_energy - ref_energy
        relative_error = abs(error / ref_energy) if ref_energy != 0 else 0.0
        
        # Check convergence
        converged = len(self.convergence_history) > 1 and \
                   abs(self.convergence_history[-1] - self.convergence_history[-2]) < self.convergence_threshold
        
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
            },
            backend_type=self.backend_config.backend_type,
            noise_model=self.backend_config.noise_model,
            noise_strength=self.backend_config.noise_strength,
            hf_energy=self.hf_energy,
        )
        
        logger.info(f"Completed {self.name}: Energy = {optimal_energy:.8f}, "
                   f"Error = {error:.8f}, Runtime = {runtime:.2f}s")
        
        # Close progress bar
        if self.progress_bar:
            self.progress_bar.close()
        
        # Log reference energies if available
        if self.hamiltonian.molecule.truncated_ground_state_energy is not None:
            logger.info(f"Full system reference energy: {ref_energy:.8f} Hartree")
            logger.info(f"Truncated system ground state: {self.hamiltonian.molecule.truncated_ground_state_energy:.8f} Hartree")
            logger.info(f"VQE result vs full system reference: {error:.8f} Hartree")
        
        return result
    
    def __repr__(self) -> str:
        return f"{self.__class__.__name__}(name={self.name}, n_qubits={self.n_qubits})"
