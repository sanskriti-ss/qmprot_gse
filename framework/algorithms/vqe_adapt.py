"""
ADAPT-VQE Implementation

Adaptive Derivative-Assembled Pseudo-Trotter VQE.
Grows the ansatz adaptively based on gradient information.
"""
import numpy as np
from typing import Optional, Any, List, Tuple
import logging

import sys
sys.path.append('..')
from core.base_vqe import BaseVQE, VQEResult
from core.hamiltonian_loader import QubitHamiltonian

logger = logging.getLogger(__name__)


class AdaptVQE(BaseVQE):
    """
    ADAPT-VQE implementation.
    
    Adaptively grows the ansatz by selecting operators from a pool
    based on their gradient magnitudes.
    """
    
    def __init__(self,
                 hamiltonian: QubitHamiltonian,
                 max_operators: int = 20,
                 gradient_threshold: float = 1e-4,
                 **kwargs):
        """
        Initialize ADAPT-VQE.
        
        Args:
            hamiltonian: QubitHamiltonian object
            max_operators: Maximum number of operators to add
            gradient_threshold: Minimum gradient to add an operator
            **kwargs: Additional arguments passed to BaseVQE
        """
        super().__init__(hamiltonian, **kwargs)
        
        self.name = "adapt_vqe"
        self.description = "Adaptive Derivative-Assembled Pseudo-Trotter VQE"
        self.max_operators = max_operators
        self.gradient_threshold = gradient_threshold
        
        # Operator pool and selected operators
        self.operator_pool: List[Tuple] = []
        self.selected_operators: List[int] = []
        self.parameters: np.ndarray = np.array([])
        
        # Will be set in build_ansatz
        self.device = None
        
    def build_ansatz(self) -> Any:
        """Build the operator pool for ADAPT-VQE"""
        import pennylane as qml
        
        n_qubits = self.n_qubits
        
        # Create device
        self.device = qml.device("lightning.qubit", wires=n_qubits) # switch to default if needed
        
        # Build operator pool (single and double excitations)
        self.operator_pool = []
        
        # Single excitation operators
        for i in range(n_qubits):
            for j in range(i + 1, n_qubits):
                self.operator_pool.append(('single', i, j))
        
        # Double excitation-like operators (simplified)
        for i in range(n_qubits - 1):
            for j in range(i + 1, n_qubits):
                self.operator_pool.append(('double', i, j, (i+1) % n_qubits, (j+1) % n_qubits))
        
        logger.info(f"Built operator pool with {len(self.operator_pool)} operators")
        
        # Initialize with empty ansatz
        self.selected_operators = []
        self.parameters = np.array([])
        self.n_parameters = 0
        
        return self.operator_pool
    
    def _build_circuit(self, params: np.ndarray):
        """Build circuit with currently selected operators"""
        import pennylane as qml
        
        n_qubits = self.n_qubits
        
        @qml.qnode(self.device)
        def circuit(params):
            # Initial state (Hartree-Fock like)
            n_electrons = self.hamiltonian.molecule.n_electrons or n_qubits // 2
            for i in range(min(n_electrons, n_qubits)):
                qml.PauliX(wires=i)
            
            # Apply selected operators
            for idx, op_idx in enumerate(self.selected_operators):
                op = self.operator_pool[op_idx]
                theta = params[idx] if idx < len(params) else 0.0
                
                if op[0] == 'single':
                    # Single excitation gate approximation
                    i, j = op[1], op[2]
                    qml.RY(theta, wires=i)
                    qml.CNOT(wires=[i, j])
                    qml.RY(-theta/2, wires=j)
                    qml.CNOT(wires=[i, j])
                    
                elif op[0] == 'double':
                    # Double excitation gate approximation
                    i, j, k, l = op[1], op[2], op[3], op[4]
                    qml.CNOT(wires=[i, j])
                    qml.RY(theta/2, wires=j)
                    qml.CNOT(wires=[k, l % n_qubits])
                    qml.RY(-theta/2, wires=l % n_qubits)
            
            H = self.hamiltonian.to_pennylane()
            return qml.expval(H)
        
        return circuit
    
    def cost_function(self, parameters: np.ndarray) -> float:
        """Evaluate the cost function"""
        circuit = self._build_circuit(parameters)
        return float(circuit(parameters))
    
    def _compute_gradients(self) -> np.ndarray:
        """Compute gradients for all operators in the pool"""
        gradients = []
        delta = 1e-5
        
        current_energy = self.cost_function(self.parameters)
        
        for op_idx in range(len(self.operator_pool)):
            if op_idx in self.selected_operators:
                gradients.append(0.0)
                continue
            
            # Temporarily add operator
            self.selected_operators.append(op_idx)
            test_params = np.append(self.parameters, delta)
            
            # Compute gradient via finite difference
            energy_plus = self.cost_function(test_params)
            test_params[-1] = -delta
            energy_minus = self.cost_function(test_params)
            
            gradient = (energy_plus - energy_minus) / (2 * delta)
            gradients.append(abs(gradient))
            
            # Remove operator
            self.selected_operators.pop()
        
        return np.array(gradients)
    
    def run(self) -> VQEResult:
        """Run ADAPT-VQE with iterative operator selection"""
        import time
        from scipy.optimize import minimize
        
        logger.info(f"Running {self.name} on {self.hamiltonian.molecule.name}")
        start_time = time.time()
        
        # Build operator pool
        self.build_ansatz()
        
        self.convergence_history = []
        
        # ADAPT loop
        for iteration in range(self.max_operators):
            # Compute gradients
            gradients = self._compute_gradients()
            max_grad_idx = np.argmax(gradients)
            max_grad = gradients[max_grad_idx]
            
            logger.debug(f"ADAPT iteration {iteration}: max gradient = {max_grad:.6f}")
            
            # Check convergence
            if max_grad < self.gradient_threshold:
                logger.info(f"Converged at iteration {iteration} (gradient below threshold)")
                break
            
            # Add best operator
            self.selected_operators.append(max_grad_idx)
            self.parameters = np.append(self.parameters, 0.0)
            self.n_parameters = len(self.parameters)
            
            # Optimize all parameters
            if len(self.parameters) > 0:
                result = minimize(
                    self.cost_function,
                    self.parameters,
                    method=self.optimizer_name,
                    options={"maxiter": 100}
                )
                self.parameters = result.x
                current_energy = result.fun
            else:
                current_energy = self.cost_function(self.parameters)
            
            self.convergence_history.append(current_energy)
        
        runtime = time.time() - start_time
        
        # Final results
        optimal_energy = self.cost_function(self.parameters)
        ref_energy = self.hamiltonian.molecule.reference_energy
        error = optimal_energy - ref_energy
        relative_error = abs(error / ref_energy) if ref_energy != 0 else 0.0
        
        result = VQEResult(
            molecule_abbrev=self.hamiltonian.molecule.abbreviation,
            molecule_name=self.hamiltonian.molecule.name,
            algorithm_name=self.name,
            calculated_energy=optimal_energy,
            reference_energy=ref_energy,
            error=error,
            relative_error=relative_error,
            n_iterations=len(self.selected_operators),
            n_qubits=self.n_qubits,
            n_parameters=self.n_parameters,
            runtime_seconds=runtime,
            convergence_history=self.convergence_history,
            optimal_parameters=self.parameters,
            converged=True,
            metadata={
                "optimizer": self.optimizer_name,
                "n_operators_selected": len(self.selected_operators),
                "selected_operators": self.selected_operators,
            }
        )
        
        logger.info(f"Completed {self.name}: Energy = {optimal_energy:.8f}, "
                   f"Error = {error:.8f}, Operators = {len(self.selected_operators)}")
        
        return result
