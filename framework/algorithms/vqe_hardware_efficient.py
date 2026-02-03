"""
Hardware-Efficient VQE Implementation

VQE with hardware-efficient ansatz optimized for near-term quantum devices.
"""
import numpy as np
from typing import Optional, Any
import logging

import sys
sys.path.append('..')
from core.base_vqe import BaseVQE, VQEResult
from core.hamiltonian_loader import QubitHamiltonian

logger = logging.getLogger(__name__)


class HardwareEfficientVQE(BaseVQE):
    """
    Hardware-Efficient VQE implementation.
    
    Uses a hardware-efficient ansatz with alternating layers of
    single-qubit rotations and entangling gates.
    """
    
    def __init__(self,
                 hamiltonian: QubitHamiltonian,
                 n_layers: int = 4,
                 entangling_gate: str = "CNOT",
                 rotation_gates: str = "RY_RZ",
                 **kwargs):
        """
        Initialize Hardware-Efficient VQE.
        
        Args:
            hamiltonian: QubitHamiltonian object
            n_layers: Number of variational layers
            entangling_gate: Type of entangling gate (CNOT, CZ, etc.)
            rotation_gates: Type of rotation gates (RY, RY_RZ, full)
            **kwargs: Additional arguments passed to BaseVQE
        """
        super().__init__(hamiltonian, **kwargs)
        
        self.name = "hardware_efficient_vqe"
        self.description = "VQE with hardware-efficient ansatz"
        self.n_layers = n_layers
        self.entangling_gate = entangling_gate
        self.rotation_gates = rotation_gates
        
        # Will be set in build_ansatz
        self.device = None
        self.cost_fn = None
        
    def build_ansatz(self) -> Any:
        """Build the hardware-efficient ansatz circuit"""
        import pennylane as qml
        
        n_qubits = self.n_qubits
        n_layers = self.n_layers
        
        # Calculate number of parameters based on rotation type
        if self.rotation_gates == "RY":
            rotations_per_qubit = 1
        elif self.rotation_gates == "RY_RZ":
            rotations_per_qubit = 2
        else:  # full
            rotations_per_qubit = 3
        
        self.n_parameters = n_qubits * rotations_per_qubit * (n_layers + 1)
        
        # Create device
        self.device = qml.device("lightning.qubit", wires=n_qubits) # switch to default if needed
        
        # Get Hamiltonian
        H = self.hamiltonian.to_pennylane()
        
        @qml.qnode(self.device)
        def circuit(params):
            param_idx = 0
            
            # Initial rotation layer
            for qubit in range(n_qubits):
                if self.rotation_gates == "RY":
                    qml.RY(params[param_idx], wires=qubit)
                    param_idx += 1
                elif self.rotation_gates == "RY_RZ":
                    qml.RY(params[param_idx], wires=qubit)
                    qml.RZ(params[param_idx + 1], wires=qubit)
                    param_idx += 2
                else:
                    qml.RX(params[param_idx], wires=qubit)
                    qml.RY(params[param_idx + 1], wires=qubit)
                    qml.RZ(params[param_idx + 2], wires=qubit)
                    param_idx += 3
            
            # Variational layers
            for layer in range(n_layers):
                # Entangling layer
                if self.entangling_gate == "CNOT":
                    for i in range(n_qubits - 1):
                        qml.CNOT(wires=[i, i + 1])
                    # Circular connectivity
                    if n_qubits > 2:
                        qml.CNOT(wires=[n_qubits - 1, 0])
                elif self.entangling_gate == "CZ":
                    for i in range(n_qubits - 1):
                        qml.CZ(wires=[i, i + 1])
                    if n_qubits > 2:
                        qml.CZ(wires=[n_qubits - 1, 0])
                
                # Rotation layer
                for qubit in range(n_qubits):
                    if self.rotation_gates == "RY":
                        qml.RY(params[param_idx], wires=qubit)
                        param_idx += 1
                    elif self.rotation_gates == "RY_RZ":
                        qml.RY(params[param_idx], wires=qubit)
                        qml.RZ(params[param_idx + 1], wires=qubit)
                        param_idx += 2
                    else:
                        qml.RX(params[param_idx], wires=qubit)
                        qml.RY(params[param_idx + 1], wires=qubit)
                        qml.RZ(params[param_idx + 2], wires=qubit)
                        param_idx += 3
            
            return qml.expval(H)
        
        self.cost_fn = circuit
        logger.info(f"Built HW-efficient ansatz with {self.n_parameters} parameters, "
                   f"{n_layers} layers, {self.entangling_gate} entangling")
        
        return circuit
    
    def cost_function(self, parameters: np.ndarray) -> float:
        """Evaluate the cost function"""
        if self.cost_fn is None:
            raise RuntimeError("Must call build_ansatz() first")
        return float(self.cost_fn(parameters))
    
    def get_initial_parameters(self) -> np.ndarray:
        """Get initial parameters - uniform random on [0, 2π]"""
        return np.random.uniform(0, 2 * np.pi, self.n_parameters)
