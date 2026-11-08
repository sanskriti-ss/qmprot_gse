# ABOUTME: Runs the registered adaptive RY/CNOT excitation approximations on a molecular objective.
# ABOUTME: Screens its circuit extensions and optimizes the selected angles.
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
from core.base_vqe import (
    BaseVQE,
    EvaluationBudgetExhausted,
    VQEResult,
    incumbent_history,
    inner_solve_iteration_cap,
    pool_probe_energy,
    record_operator_fit,
)
from core.hamiltonian_loader import QubitHamiltonian
from core.pauli_measurement import adaptive_state_measurement

logger = logging.getLogger(__name__)


class AdaptVQE(BaseVQE):
    """
    ADAPT-VQE implementation.
    
    Adaptively grows the ansatz by selecting operators from a pool
    based on their gradient magnitudes.
    """
    
    parameter_period = 4 * np.pi

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
        self.description = "Adaptive VQE with approximate RY/CNOT excitations (particle number can change)"
        self.excitation_implementation = "approximate RY/CNOT"
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
        from core.backend_manager import analytic_energy, create_device
        
        n_qubits = self.n_qubits
        
        # Create device via backend manager
        self.device = create_device(self.backend_config)
        self._energy_observable = None
        if self.backend_config.backend_type == "statevector" and self.backend_config.n_shots == 0:
            self._energy_observable = self.hamiltonian.to_pennylane()
        
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
        self._measure_energy = None
        if self._energy_observable is not None and type(self)._build_circuit is AdaptVQE._build_circuit:
            self.cost_fn = self._build_circuit(self.parameters)
            self._measure_energy = analytic_energy(self.cost_fn, self._energy_observable,
                                                   measure_state=adaptive_state_measurement(n_qubits, self._energy_observable))
        
        return self.operator_pool
    
    def _build_circuit(self, params: np.ndarray):
        """Build circuit with currently selected operators"""
        import pennylane as qml
        
        n_qubits = self.n_qubits
        insert_noise = self.noise_inserter
        
        @qml.qnode(self.device)
        def circuit(params):
            # Initial state preparation
            self._prepare_initial_state()
            
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
                
                # Apply noise after each operator (no-op for statevector)
                insert_noise()
            
            H = self._energy_observable
            if H is None:
                H = self.hamiltonian.to_pennylane()
            return qml.expval(H)
        
        return circuit
    
    def cost_function(self, parameters: np.ndarray) -> float:
        """Evaluate the cost function"""
        if self._measure_energy is not None:
            return float(self._measure_energy(parameters))
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
            test_params = np.append(self.parameters, delta)
            self.selected_operators.append(op_idx)
            try:
                # Compute gradient via finite difference at theta=0
                energy_plus = pool_probe_energy(self, test_params)
                test_params[-1] = -delta
                energy_minus = pool_probe_energy(self, test_params)
            finally:
                # Remove operator
                self.selected_operators.pop()

            gradient = (energy_plus - energy_minus) / (2 * delta)
            gradients.append(abs(gradient))
        
        return np.array(gradients)
    
    def run(self) -> VQEResult:
        """Run ADAPT-VQE with iterative operator selection"""
        import time

        logger.info(f"Running {self.name} on {self.hamiltonian.molecule.name}")
        logger.info(f"Backend: {self.backend_config.label}")
        start_time = time.time()
        
        # ── Hartree-Fock energy verification ──────────────────────────
        self._perform_hf_verification()
        
        # Build operator pool
        self.build_ansatz()
        
        self.convergence_history = []
        
        # ADAPT loop.  The inner solve goes through BaseVQE.optimize so
        # ``max_iterations``, the convergence tolerance, and the Bayesian
        # path all apply.  ``update_history=False`` keeps this list as one
        # energy per accepted operator.
        grad_converged = False
        budget_limited = None
        for iteration in range(self.max_operators):
            try:
                gradients = self._compute_gradients()
            except EvaluationBudgetExhausted:
                budget_limited = "gradient screening"
                logger.info("Evaluation budget exhausted during gradient screening")
                break
            max_grad = gradients.max(initial=0.0)
            
            logger.debug(f"ADAPT iteration {iteration}: max gradient = {max_grad:.6f}")
            
            if max_grad == 0 or max_grad < self.gradient_threshold:
                logger.info(f"Converged at iteration {iteration} (maximum pool gradient {max_grad:.2e})")
                grad_converged = True
                break
            max_grad_idx = np.argmax(gradients)

            operators_left = self.max_operators - len(self.selected_operators)
            step_cap = inner_solve_iteration_cap(
                self, operators_left, self.max_iterations,
            )
            if step_cap < 1:
                budget_limited = "re-optimization"
                logger.info(
                    "Evaluation budget cannot give each of the %d remaining "
                    "operators an iteration",
                    operators_left,
                )
                break
            
            self.selected_operators.append(max_grad_idx)
            self.parameters = np.append(self.parameters, 0.0)
            self.n_parameters = len(self.parameters)

            saved_max = self.max_iterations
            self.max_iterations = step_cap
            try:
                self.parameters, current_energy = self.optimize(
                    initial_parameters=self.parameters,
                    update_history=False,
                )
            except EvaluationBudgetExhausted:
                self.selected_operators.pop()
                self.parameters = np.asarray(self.parameters[:-1], dtype=float)
                self.n_parameters = len(self.parameters)
                budget_limited = "re-optimization"
                logger.info("Evaluation budget exhausted during re-optimization")
                break
            finally:
                self.max_iterations = saved_max
            record_operator_fit(self, current_energy)
        
        runtime = time.time() - start_time
        
        if self.convergence_history:
            optimal_energy = float(self.convergence_history[-1])
        else:
            try:
                optimal_energy = float(self.cost_function(self.parameters))
            except EvaluationBudgetExhausted:
                optimal_energy = float("nan")
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
            best_so_far_history=incumbent_history(self.convergence_history),
            optimal_parameters=self.parameters,
            converged=grad_converged,
            metadata={
                "optimizer": self.optimizer_name,
                "excitation_implementation": self.excitation_implementation,
                "n_operators_selected": len(self.selected_operators),
                "selected_operators": self.selected_operators,
                "evaluation_budget": budget_limited,
            },
            backend_type=self.backend_config.backend_type,
            noise_model=self.backend_config.noise_model,
            noise_strength=self.backend_config.noise_strength,
            hf_energy=self.hf_energy,
        )
        
        logger.info(f"Completed {self.name}: Energy = {optimal_energy:.8f}, "
                   f"Error = {error:.8f}, Operators = {len(self.selected_operators)}")
        
        return result
