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
from core.iqcc_helpers import IQCC_Operator, PauliOperatorPool, of_to_pennylane
from core.algebraic_operators import of_commutator

import pennylane as qml
from core.backend_manager import create_device

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
        self.parameters: np.ndarray = np.array([])

        self.device = None
        self.cost_fn = None

    def build_ansatz(self):
        n_qubits = self.n_qubits
        self.device = create_device(self.backend_config)

        H = self.hamiltonian.to_pennylane()
        hf_bitstring = np.zeros(self.n_qubits, dtype=int)
        hf_bitstring[:self.hamiltonian.molecule.n_electrons] = 1

        @qml.qnode(self.device)
        def circuit(params):

            qml.BasisState(hf_bitstring, wires=range(self.n_qubits))
            for theta, op in zip(params, self.selected_operators):
                for term, coeff in op.terms.items():
                    pauli_word = ["I"] * n_qubits

                    for qubit_index, pauli_char in term:
                        pauli_word[qubit_index] = pauli_char
                    
                    pauli_string = "".join(pauli_word)
                    qml.PauliRot(2 * theta * coeff.real, pauli_string, wires=range(n_qubits))
                #qml.PauliRot(2*theta, op, wires=range(self.n_qubits))

            # insert_noise()
            
            return qml.expval(H)
        self.cost_fn = circuit
        #TODO: Add logger
        return circuit
    
    def compute_expectation(self, observable, params):
        n_qubits = self.n_qubits
        hf_bitstring = np.zeros(n_qubits, dtype=int)
        hf_bitstring[:self.hamiltonian.molecule.n_electrons] = 1

        @qml.qnode(self.device)
        def circuit(p):

            qml.BasisState(hf_bitstring, wires=range(n_qubits))

            for theta, op in zip(params, self.selected_operators):
                for term, coeff in op.terms.items():
                    pauli_word = ["I"] * n_qubits

                    for qubit_index, pauli_char in term:
                        pauli_word[qubit_index] = pauli_char
                    
                    pauli_string = "".join(pauli_word)
                    qml.PauliRot(2 * theta * coeff.real, pauli_string, wires=range(n_qubits))

            return qml.expval(observable)

        return float(circuit(params))

    def cost_function(self, parameters: np.ndarray) -> float:
        """Evaluate the cost function"""
        if self.cost_fn is None:
            raise RuntimeError("Must call build_ansatz() first")
        return float(self.cost_fn(self.parameters))
    

    def run(self) -> VQEResult:
        from openfermion import QubitOperator
        from scipy.optimize import minimize

        logger.info(f"Running {self.name} on {self.hamiltonian.molecule.name}")
        self._perform_hf_verification()

        start_time = time.time()

        # Initial empty ansatz
        self.selected_operators = []
        self.parameters = np.array([])
        self.build_ansatz()
        energy = self.cost_function(self.parameters)

        for k in range(self.max_operators):

            logger.info(f"iQCC step {k+1}/{self.max_operators}")

            # Compute gradients over pool
            gradients = {}
            H_of = self.hamiltonian.to_openfermion()
            for op in self.operator_pool:
                op_of = QubitOperator(op)
                comm = 1j * of_commutator(H_of, op_of)
                if not comm.terms:
                    continue
                comm_pl = of_to_pennylane(comm)
                grad = self.compute_expectation(comm_pl, self.parameters)
                #print(grad)
                gradients[op] = abs(float(grad))
            print("Number of gradients", len(gradients))
            print("max gradient:", max(gradients.values() if gradients else None))
            if not gradients:
                break

            best_op, best_grad = max(gradients.items(), key=lambda x: x[1])
            
            logger.info(f"Selected operator with gradient {best_grad:.3e}")

            if best_grad < self.gradient_threshold:
                break

            # Append operator
            self.selected_operators.append(QubitOperator(best_op))

            self.parameters = np.append(self.parameters, 0.0)

            self.build_ansatz()

            cost_fn = self.cost_fn

            # Optimize all parameters
            if len(self.parameters) > 0:
                print(type(self.optimizer_name))
                result = minimize(
                    cost_fn,
                    self.parameters,
                    method="COBYLA",
                    options={"maxiter": 100}
                )
                self.parameters = result.x
                energy = result.fun
            else:
                energy = self.cost_function(self.parameters)
            
            self.convergence_history.append(energy)

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
            n_parameters=len(self.parameters),
            runtime_seconds=runtime,
            convergence_history=self.convergence_history,
            optimal_parameters=self.parameters,
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

        

            