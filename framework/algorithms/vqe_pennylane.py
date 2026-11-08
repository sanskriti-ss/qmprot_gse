# ABOUTME: Builds the generic layered PennyLane molecular VQE circuit.
# ABOUTME: Applies three local rotations per wire and fixed nearest-neighbor CNOTs.
"""
Pennylane VQE Implementation

Layered RX, RY and RZ rotations with nearest-neighbor CNOTs.
"""
import numpy as np
from typing import Optional, Any
import logging

import sys
sys.path.append('..')
from core.base_vqe import BaseVQE, VQEResult
from core.hamiltonian_loader import QubitHamiltonian

logger = logging.getLogger(__name__)


class PennylaneVQE(BaseVQE):
    """
    VQE with local rotations and staggered nearest-neighbor entanglers.
    
    Uses PennyLane for quantum circuit simulation.
    """
    
    parameter_period = 2 * np.pi

    def __init__(self,
                 hamiltonian: QubitHamiltonian,
                 n_layers: int = 2,
                 **kwargs):
        """
        Initialize Pennylane VQE.
        
        Args:
            hamiltonian: QubitHamiltonian object
            n_layers: Number of variational layers
            **kwargs: Additional arguments passed to BaseVQE
        """
        super().__init__(hamiltonian, **kwargs)
        
        self.name = "pennylane_vqe"
        self.description = "Layered rotation and CNOT VQE"
        self.n_layers = n_layers
        
        # Will be set in build_ansatz
        self.device = None
        self.cost_fn = None
        
    def build_ansatz(self) -> Any:
        """Build the rotation and CNOT circuit"""
        import pennylane as qml
        from core.backend_manager import create_device, fixed_circuit_energy
        
        n_qubits = self.n_qubits
        n_layers = self.n_layers
        
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
            
            # Initial state preparation
            self._prepare_initial_state()
            
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
        
        circuit.func.rotation_layout = self._rotation_layout
        self.cost_fn = circuit
        self._energy = fixed_circuit_energy(circuit, H)
        logger.info(f"Built ansatz with {self.n_parameters} parameters, {n_layers} layers, "
                   f"backend={self.backend_config.label}")
        
        return circuit
    
    def cost_function(self, parameters: np.ndarray) -> float:
        """Evaluate the cost function"""
        if self.cost_fn is None:
            raise RuntimeError("Must call build_ansatz() first")
        return float(self._energy(parameters))
    
    def get_initial_parameters(self) -> np.ndarray:
        """Get initial parameters - small random values"""
        return np.random.uniform(-0.1, 0.1, self.n_parameters)
