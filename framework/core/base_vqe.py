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

from .hamiltonian_loader import QubitHamiltonian

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
    
    def to_dict(self) -> Dict:
        """Convert to dictionary for JSON serialization"""
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
            "metadata": self.metadata,
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
            **kwargs: Additional algorithm-specific parameters
        """
        self.hamiltonian = hamiltonian
        self.optimizer_name = optimizer
        self.max_iterations = max_iterations
        self.convergence_threshold = convergence_threshold
        self.n_shots = n_shots
        self.random_seed = random_seed
        self.kwargs = kwargs
        
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
        
        # Build components
        self.n_qubits = hamiltonian.n_qubits
        self.n_parameters: int = 0
        
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
        
        if self.iteration_count % 10 == 0:
            logger.debug(f"Iteration {self.iteration_count}: Energy = {energy:.8f}")
    
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
    
    def run(self) -> VQEResult:
        """
        Run the full VQE algorithm.
        
        Returns:
            VQEResult with all results and metadata
        """
        logger.info(f"Running {self.name} on {self.hamiltonian.molecule.name}")
        
        # Build ansatz
        start_time = time.time()
        self.build_ansatz()
        
        # Run optimization
        optimal_params, optimal_energy = self.optimize()
        
        runtime = time.time() - start_time
        
        # Calculate errors
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
            }
        )
        
        logger.info(f"Completed {self.name}: Energy = {optimal_energy:.8f}, "
                   f"Error = {error:.8f}, Runtime = {runtime:.2f}s")
        
        return result
    
    def __repr__(self) -> str:
        return f"{self.__class__.__name__}(name={self.name}, n_qubits={self.n_qubits})"
