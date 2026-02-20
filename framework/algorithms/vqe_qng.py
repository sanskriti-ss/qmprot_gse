"""
QNG VQE Implementation
VQE with Quantum Natural Gradient optimization using PennyLane.
"""
import numpy as np
from typing import Optional, Any
import logging
import sys
sys.path.append('..')
from core.base_vqe import BaseVQE, VQEResult
from core.hamiltonian_loader import QubitHamiltonian

logger = logging.getLogger(__name__)

class QNGVQE(BaseVQE):
    """
    VQE with Quantum Natural Gradient optimization.
    
    Uses the Fubini-Study metric tensor to perform natural gradient descent,
    which can converge faster than standard gradient descent by accounting
    for the geometry of quantum state space.
    """
    
    def __init__(self,
                 hamiltonian: QubitHamiltonian,
                 n_layers: int = 2,
                 stepsize: float = 0.01,
                 approx: str = "block-diag",
                 **kwargs):
        """
        Initialize QNG VQE.
        
        Args:
            hamiltonian: QubitHamiltonian object
            n_layers: Number of variational layers
            stepsize: Learning rate for QNG optimizer
            approx: Approximation method for metric tensor
                   - "block-diag": Block diagonal approximation (faster)
                   - "diag": Diagonal approximation (fastest, less accurate)
                   - None: Full metric tensor (slowest, most accurate)
            **kwargs: Additional arguments passed to BaseVQE
        """
        super().__init__(hamiltonian, **kwargs)
        
        self.name = "qng_vqe"
        self.description = "VQE with Quantum Natural Gradient optimization"
        self.n_layers = n_layers
        self.stepsize = stepsize
        self.approx = approx
        
        # Will be set in build_ansatz
        self.device = None
        self.cost_fn = None
        self.optimizer = None
        
    def build_ansatz(self) -> Any:
        """Build the UCCSD-inspired ansatz circuit (same as vanilla VQE)"""
        import pennylane as qml
        from core.backend_manager import create_device
        
        n_qubits = self.n_qubits
        n_layers = self.n_layers
        
        # Parameters: rotations on each qubit + entangling parameters
        # 3 rotation angles per qubit per layer + CNOT structure
        self.n_parameters = n_qubits * 3 * n_layers
        
        # Create device via backend manager
        self.device = create_device(self.backend_config)
        
        # Get Hamiltonian in PennyLane format
        H = self.hamiltonian.to_pennylane()
        
        # Noise insertion callback (no-op for statevector)
        insert_noise = self.noise_inserter
        
        @qml.qnode(self.device)
        def circuit(params):
            params = params.reshape(n_layers, n_qubits, 3)
            
            # Initial state preparation (Hartree-Fock like)
            n_electrons = self.hamiltonian.molecule.n_electrons or n_qubits // 2
            for i in range(min(n_electrons, n_qubits)):
                qml.PauliX(wires=i)
            
            # Variational layers
            for layer in range(n_layers):
                # Single-qubit rotations
                for qubit in range(n_qubits):
                    qml.RX(params[layer, qubit, 0], wires=qubit)
                    qml.RY(params[layer, qubit, 1], wires=qubit)
                    qml.RZ(params[layer, qubit, 2], wires=qubit)
                
                # Entangling layer (nearest-neighbor CNOTs)
                for i in range(0, n_qubits - 1, 2):
                    qml.CNOT(wires=[i, i + 1])
                for i in range(1, n_qubits - 1, 2):
                    qml.CNOT(wires=[i, i + 1])
                
                # Apply noise after each layer (no-op for statevector)
                insert_noise()
            
            return qml.expval(H)
        
        self.cost_fn = circuit
        
        # Initialize QNG optimizer
        import pennylane as qml
        self.optimizer = qml.QNGOptimizer(
            stepsize=self.stepsize,
            approx=self.approx,
            lam=0.001  # Regularization parameter for metric tensor
        )
        
        logger.info(f"Built QNG ansatz with {self.n_parameters} parameters, "
                   f"{n_layers} layers, stepsize={self.stepsize}, "
                   f"approx={self.approx}, backend={self.backend_config.label}")
        
        return circuit
    
    def cost_function(self, parameters: np.ndarray) -> float:
        """Evaluate the cost function"""
        if self.cost_fn is None:
            raise RuntimeError("Must call build_ansatz() first")
        return float(self.cost_fn(parameters))
    
    def get_initial_parameters(self) -> np.ndarray:
        """Get initial parameters - small random values"""
        return np.random.uniform(-0.1, 0.1, self.n_parameters)
    
    def optimize(self, 
                 initial_params: Optional[np.ndarray] = None,
                 max_iterations: Optional[int] = None) -> VQEResult:
        """
        Run QNG optimization.
        
        This overrides the base class optimize method to use QNG's step_and_cost.
        """
        if self.cost_fn is None:
            self.build_ansatz()
        
        if initial_params is None:
            params = self.get_initial_parameters()
        else:
            params = initial_params.copy()
        
        max_iter = max_iterations or self.max_iterations
        
        energy_history = []
        param_history = []
        
        logger.info(f"Starting QNG optimization (max_iterations={max_iter})")
        
        for iteration in range(max_iter):
            # QNG step - this computes metric tensor internally!
            params, energy = self.optimizer.step_and_cost(self.cost_fn, params)
            
            energy_history.append(float(energy))
            param_history.append(params.copy())
            
            if iteration % 10 == 0:
                logger.info(f"Iteration {iteration}: Energy = {energy:.8f}")
            
            # Check convergence
            if iteration > 0:
                energy_change = abs(energy_history[-1] - energy_history[-2])
                if energy_change < self.convergence_threshold:
                    logger.info(f"Converged at iteration {iteration} "
                              f"(ΔE = {energy_change:.2e})")
                    break
        
        final_energy = energy_history[-1]
        
        # Create result object
        result = VQEResult(
            energy=final_energy,
            parameters=params,
            n_iterations=len(energy_history),
            converged=(len(energy_history) < max_iter),
            energy_history=energy_history,
            algorithm=self.name
        )
        
        logger.info(f"Optimization complete: Final energy = {final_energy:.8f}")
        
        return result
