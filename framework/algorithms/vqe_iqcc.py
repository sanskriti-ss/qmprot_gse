'''
Iterative Qubit Coupled Cluster (iQCC) Implementation

Iteratively transforms Hamiltonian and aims to minimize circuit depth

'''

import numpy as np
from typing import Optional, Any
import logging

import sys
import time
sys.path.append('..')
from core.base_vqe import BaseVQE, VQEResult
from core.hamiltonian_loader import QubitHamiltonian
from core.iqcc_helpers import IQCC_Operator, PauliOperatorPool
from core.algebraic_operators import of_commutator

logger = logging.getLogger(__name__)

class iQCC_VQE(BaseVQE):
    def __init__(self, 
                 hamiltonian: QubitHamiltonian, 
                 max_operators: int=20, 
                 gradient_threshold: float= 1e-4,
                 **kwargs):
        super().__init__(hamiltonian, kwargs)
        self.name = 'iqcc_vqe'
        self.description = 'Iterative Qubit Coupled Cluster'
        
        self.max_operators = max_operators
        self.gradient_threshold = gradient_threshold

        # create operator_pool
        self.operator_pool = PauliOperatorPool(2).generate(self.n_qubits)

        self.selected_operators = []
        self.parameters = []

        self.device = None
        self.cost_fn = None

    def build_ansatz(self):
        import pennylane as qml
        from core.backend_manager import create_device

        n_qubits = self.n_qubits
        self.device = create_device(self.backend_config)

        H = self.hamiltonian.to_pennylane()
        # Maybe experiment with insert_noise = self.noise_inserter

        def circuit(params, observable):
            for i in range(n_qubits // 2):
                qml.PauliX(wires=i)
            
            for theta, op in zip(params, self.selected_operators):
                qml.PauliRot(2 * theta * op.coefficient, op.pauli_word, wires=list(range(n_qubits)))
                #qml.PauliRot(2*theta, op, wires=range(self.n_qubits))

            # insert_noise() <-- could be interesting
            return qml.expval(observable)
        self.cost_fn = circuit
        # TODO: add logger
        return circuit
    
    def cost_function(self, parameters: np.ndarray) -> float:
        """Evaluate the cost function"""
        if self.cost_fn is None:
            raise RuntimeError("Must call build_ansatz() first")
        return float(self.cost_fn(parameters))
    
    ''' Legacy compute_gradients function
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
        return gradients'''

    def run(self) -> VQEResult:
        from openfermion import QubitOperator

        logger.info(f"Running {self.name} on {self.hamiltonian.molecule.name}")
        self._perform_hf_verification()

        start_time = time.time()

        # Initial empty ansatz
        self.selected_operators = []
        parameters = np.array([])

        for k in range(self.max_operators):

            logger.info(f"iQCC step {k+1}/{self.max_operators}")

            # Compute gradients over pool
            gradients = {}
            H_of = self.hamiltonian.to_openfermion()
            for op in self.operator_pool:
                op_of = QubitOperator(op)
                comm = of_commutator(H_of, op)
                if comm.is_zero():
                    continue

                grad = self._compute_expectation(comm, parameters)
                gradients[op] = abs(grad)

            if not gradients:
                break

            best_op, best_grad = max(gradients.items(), key=lambda x: x[1])

            logger.info(f"Selected operator with gradient {best_grad:.3e}")

            if best_grad < self.gradient_threshold:
                break

            # Append operator
            self.selected_operators.append(best_op)

            # Rebuild ansatz
            self.build_ansatz()

            # Expand parameter vector
            parameters = np.append(parameters, 0.0)

            # Optimize full parameter vector
            parameters, energy = self.optimize(parameters)

        runtime = time.time() - start_time

        # Final bookkeeping
        ref_energy = self.hamiltonian.molecule.reference_energy
        error = energy - ref_energy

        return VQEResult(
            molecule_abbrev=self.hamiltonian.molecule.abbreviation,
            molecule_name=self.hamiltonian.molecule.name,
            algorithm_name=self.name,
            calculated_energy=energy,
            reference_energy=ref_energy,
            error=error,
            relative_error=abs(error / ref_energy),
            n_iterations=self.iteration_count,
            n_qubits=self.n_qubits,
            n_parameters=len(parameters),
            runtime_seconds=runtime,
            convergence_history=self.convergence_history,
            optimal_parameters=parameters,
            converged=True,
            metadata={
                "n_selected_operators": len(self.selected_operators),
                "max_operators": self.max_operators,
            },
            backend_type=self.backend_config.backend_type,
            noise_model=self.backend_config.noise_model,
            noise_strength=self.backend_config.noise_strength,
            hf_energy=self.hf_energy,
        )

        

            