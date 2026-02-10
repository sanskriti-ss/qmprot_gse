'''
Iterative Qubit Coupled Cluster (iQCC) Implementation

Iteratively transforms Hamiltonian and aims to minimize circuit depth

'''

import numpy as np
from typing import Optional, Any
import logging

import sys
sys.path.append('..')
from core.base_vqe import BaseVQE, VQEResult
from core.hamiltonian_loader import QubitHamiltonian
from core.iqcc_helpers import IQCC_Operator, PauliOperatorPool

logger = logging.getLogger(__name__)

class iQCC_VQE(BaseVQE):
    def __init__(self, 
                 hamiltonian: QubitHamiltonian, 
                 max_operators: int=20, 
                 gradient_threshold: float= 1e-4,
                 **kwargs):
        super().__init__(hamiltonian, kwargs)
        self.name = 'iqcc_vqe'
        self.description = 'Placeholder description'
        # define pauli operator pool
        self.max_operators = max_operators
        self.gradient_threshold = gradient_threshold

        self.selected_operators = []
        self.parameters = []

        self.device = None
        self.cost_fn = None
    def build_ansatz(self):
        import pennylane as qml
        from core.backend_manager import create_device

        n_qubits = self.n_qubits
        device = create_device(self.backend_config)

        def circuit(params, observable):
            for i in range(n_qubits // 2):
                qml.PauliX(wires=i)
            
            for theta, op in zip(params, self.selected_operators):
                qml.PauliRot(2 * theta * op.coefficient, op.pauli_word, wires=list(range(n_qubits)))
            
            return qml.expval(observable)
        
        # TODO: add logger + cost
        return circuit
    
    def compute_gradients(self, pool, circuit):
        gradients = {}
        for pauli_word in pool:
            A = IQCC_Operator(pauli_word)

            commutator = self.hamiltonian.commutator(pauli_word)
            if commutator.is_zero():
                continue
            obs = commutator.to_pennylane()
            grad = circuit(self.parameters, obs)
            gradients[A] = abs(float(grad))
        return gradients

    

        