"""
Vanilla VQE Implementation

Standard VQE with UCCSD-like ansatz using PennyLane.
"""
import numpy as np
from typing import Optional, Any
import logging

import sys
sys.path.append('..')
from core.base_vqe import BaseVQE, VQEResult
from core.hamiltonian_loader import QubitHamiltonian

logger = logging.getLogger(__name__)


class VanillaVQE(BaseVQE):
    """
    Standard VQE implementation with UCCSD-inspired ansatz.
    
    Uses PennyLane for quantum circuit simulation.
    """
    
    def __init__(self,
                 hamiltonian: QubitHamiltonian,
                 n_layers: int = 2,
                 **kwargs):
        """
        Initialize Vanilla VQE.
        
        Args:
            hamiltonian: QubitHamiltonian object
            n_layers: Number of variational layers
            **kwargs: Additional arguments passed to BaseVQE
        """
        super().__init__(hamiltonian, **kwargs)
        
        self.name = "vanilla_vqe"
        self.description = "Standard VQE with UCCSD-inspired ansatz"
        self.n_layers = n_layers
        
        # Will be set in build_ansatz
        self.device = None
        self.cost_fn = None
        
    def build_ansatz(self) -> Any:
        """Build the UCCSD-inspired ansatz circuit"""
        import pennylane as qml
        
        n_qubits = self.n_qubits
        n_layers = self.n_layers
        
        # Parameters: rotations on each qubit + entangling parameters
        # 3 rotation angles per qubit per layer + CNOT structure
        self.n_parameters = n_qubits * 3 * n_layers
        
        # Create device
        self.device = qml.device("lightning.qubit", wires=n_qubits) # switch to default if needed
        
        # Get Hamiltonian in PennyLane format
        H = self.hamiltonian.to_pennylane()
        
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
            
            return qml.expval(H)
        
        self.cost_fn = circuit
        logger.info(f"Built ansatz with {self.n_parameters} parameters, {n_layers} layers")
        
        return circuit
    
    def cost_function(self, parameters: np.ndarray) -> float:
        """Evaluate the cost function"""
        if self.cost_fn is None:
            raise RuntimeError("Must call build_ansatz() first")
        return float(self.cost_fn(parameters))
    
    def get_initial_parameters(self) -> np.ndarray:
        """Get initial parameters - small random values"""
        return np.random.uniform(-0.1, 0.1, self.n_parameters)
