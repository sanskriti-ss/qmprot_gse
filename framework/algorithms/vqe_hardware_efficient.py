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
                 max_free_parameters: Optional[int] = None,
                 prepare_hf: bool = False,
                 **kwargs):
        """
        Initialize Hardware-Efficient VQE.
        
        Args:
            hamiltonian: QubitHamiltonian object
            n_layers: Number of variational layers
            entangling_gate: Type of entangling gate (CNOT, CZ, etc.)
            rotation_gates: Type of rotation gates (RY, RY_RZ, full)
            max_free_parameters: If set, only the first K rotation slots are
                variational; the remaining rotation gates are omitted (equal
                to fixing them at 0).  Lets the ansatz hit an exact parameter
                count (e.g. 10 / 50 / 100) instead of the next multiple of
                ``n_qubits * rotations_per_qubit``.  ``None`` keeps every slot.
            prepare_hf: If True, start from the Hartree-Fock (or CS-rotated)
                reference state instead of |0...0>.  Default False keeps the
                original behaviour.
            **kwargs: Additional arguments passed to BaseVQE
        """
        super().__init__(hamiltonian, **kwargs)
        
        self.name = "hardware_efficient_vqe"
        self.description = "VQE with hardware-efficient ansatz"
        self.n_layers = n_layers
        self.entangling_gate = entangling_gate
        self.rotation_gates = rotation_gates
        self.max_free_parameters = max_free_parameters
        self.prepare_hf = prepare_hf
        
        # Will be set in build_ansatz
        self.device = None
        self.cost_fn = None
        
    def build_ansatz(self) -> Any:
        """Build the hardware-efficient ansatz circuit"""
        import pennylane as qml
        from core.backend_manager import create_device
        
        n_qubits = self.n_qubits
        n_layers = self.n_layers
        
        # Calculate number of parameters based on rotation type
        if self.rotation_gates == "RY":
            rotations_per_qubit = 1
        elif self.rotation_gates == "RY_RZ":
            rotations_per_qubit = 2
        else:  # full
            rotations_per_qubit = 3
        
        n_slots = n_qubits * rotations_per_qubit * (n_layers + 1)
        if self.max_free_parameters is None:
            n_free = n_slots
        else:
            n_free = int(self.max_free_parameters)
            if not 1 <= n_free <= n_slots:
                raise ValueError(
                    f"max_free_parameters={n_free} must be in [1, {n_slots}] "
                    f"for {n_qubits} qubits, {n_layers} layers, "
                    f"rotation_gates={self.rotation_gates}"
                )
        self.n_parameters = n_free
        gate_seq = {"RY": (qml.RY,), "RY_RZ": (qml.RY, qml.RZ)}.get(
            self.rotation_gates, (qml.RX, qml.RY, qml.RZ)
        )
        
        # Create device via backend manager
        self.device = create_device(self.backend_config)
        
        # Get Hamiltonian
        H = self.hamiltonian.to_pennylane()
        
        # Noise insertion callback (no-op for statevector)
        insert_noise = self.noise_inserter
        
        def rotate(params, qubit, slot):
            # Slots >= n_free are frozen at 0 (identity), i.e. omitted.
            for k, gate in enumerate(gate_seq):
                if slot + k < n_free:
                    gate(params[slot + k], wires=qubit)
            return slot + len(gate_seq)
        
        @qml.qnode(self.device)
        def circuit(params):
            param_idx = 0
            
            if self.prepare_hf:
                self._prepare_initial_state()
            
            # Initial rotation layer
            for qubit in range(n_qubits):
                param_idx = rotate(params, qubit, param_idx)
            
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
                    param_idx = rotate(params, qubit, param_idx)
                
                # Apply noise after each layer (no-op for statevector)
                insert_noise()
            
            return qml.expval(H)
        
        self.cost_fn = circuit
        logger.info(f"Built HW-efficient ansatz with {self.n_parameters} parameters "
                   f"({n_slots} rotation slots), "
                   f"{n_layers} layers, {self.entangling_gate} entangling, "
                   f"prepare_hf={self.prepare_hf}, "
                   f"backend={self.backend_config.label}")
        
        return circuit
    
    def cost_function(self, parameters: np.ndarray) -> float:
        """Evaluate the cost function"""
        if self.cost_fn is None:
            raise RuntimeError("Must call build_ansatz() first")
        return float(self.cost_fn(parameters))
    
    def get_initial_parameters(self) -> np.ndarray:
        """Get initial parameters - uniform random on [0, 2π]"""
        return np.random.uniform(0, 2 * np.pi, self.n_parameters)
