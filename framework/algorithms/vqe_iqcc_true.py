'''
Iterative Qubit Coupled Cluster (iQCC)-Inspired Adaptive VQE Implementation

Uses constant-depth, shallow quantum circuits which are iteratively updated through canonical transformations of the 
Hamiltonian on a classical computer.

WARNING: This can be extremely computationally expensive on classical computers. It is recommended to use the iqcc_inspired_vqe instead.

'''

import numpy as np
from typing import Optional, Any
import logging

import sys
import time
sys.path.append('..')
from core.base_vqe import BaseVQE, VQEResult
from core.hamiltonian_loader import QubitHamiltonian
from core.iqcc_helpers import IQCC_Operator, PauliOperatorPool, of_to_pennylane, prune_hamiltonian
from core.algebraic_operators import of_commutator

import pennylane as qml
from core.backend_manager import create_device

logger = logging.getLogger(__name__)

class iQCC_true_VQE(BaseVQE):
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
        self.max_terms = 10000

        n_el_raw = self.hamiltonian.molecule.n_electrons or self.n_qubits // 2
        if n_el_raw >= self.n_qubits:
            self._eff_n_electrons = max(1, self.n_qubits // 2)
        else:
            self._eff_n_electrons = n_el_raw

        # create operator_pool
        self.operator_pool = PauliOperatorPool(2).generate(self.n_qubits)

        self.selected_operators = []
        self.parameters: np.ndarray = np.array([])

        self.device = None
        self.cost_fn = None

    def _perform_hf_verification(self) -> None:
        """Compute HF energy using the effective active-space electron count."""
        from core.hf_verification import compute_hf_energy
        try:
            self.hf_energy = compute_hf_energy(
                self.hamiltonian, n_electrons=self._eff_n_electrons
            )
            logger.info(
                f"HF energy (truncated, n_el={self._eff_n_electrons}) = "
                f"{self.hf_energy:.8f} Ha"
            )
        except Exception as exc:
            logger.warning(f"Could not compute HF energy: {exc}")
            self.hf_energy = None
    
    def build_ansatz(self):
        n_qubits = self.n_qubits
        self.device = create_device(self.backend_config)

        H = self.hamiltonian.to_pennylane()
        @qml.qnode(self.device)
        def circuit(params):
            self._prepare_initial_state()
            return qml.expval(H)
        self.cost_fn = circuit
        #TODO: Add logger
        return circuit
    
    def compute_hf_energy(self, observable):
        @qml.qnode(self.device)
        def circuit():
            self._prepare_initial_state()
            return qml.expval(observable)

        return float(circuit())

    def cost_function(self, parameters: np.ndarray) -> float:
        """Evaluate the cost function"""
        if self.cost_fn is None:
            raise RuntimeError("Must call build_ansatz() first")
        return float(self.cost_fn(self.parameters))

    def run(self) -> VQEResult:
        from openfermion import QubitOperator
        from scipy.optimize import minimize_scalar

        prune_threshold = 1e-8

        logger.info(f"Running {self.name} on {self.hamiltonian.molecule.name}")
        self._perform_hf_verification()

        start_time = time.time()

        # Initial empty ansatz
        self.selected_operators = []
        self.parameters = np.array([])
        self.build_ansatz()
        energy = self.cost_function(self.parameters)

        H_current = self.hamiltonian.to_openfermion()

        for k in range(self.max_operators):

            logger.info(f"iQCC step {k+1}/{self.max_operators}")

            # Compute gradients over pool
            gradients = {}
            H_of = H_current
            for op in self.operator_pool:
                op_of = QubitOperator(op)
                comm = 1j * of_commutator(H_of, op_of)
                if not comm.terms:
                    continue
                comm_pl = of_to_pennylane(comm)
                grad = self.compute_hf_energy(comm_pl)
                gradients[op] = abs(float(grad))
            if not gradients:
                break

            best_op, best_grad = max(gradients.items(), key=lambda x: x[1])
            
            logger.info(f"Selected operator with gradient {best_grad:.3e}")

            if best_grad < self.gradient_threshold:
                break
            best_op_of = QubitOperator(best_op)

            def energy_tau(tau):
                comm1 = of_commutator(H_current, best_op_of)
                comm2 = of_commutator(comm1, best_op_of)

                H_trial = (H_current + tau * comm1 + 0.5 * tau**2 * comm2)

                H_trial = prune_hamiltonian(H_trial, threshold=prune_threshold, max_terms=self.max_terms)
                H_trial.compress()
                H_trial_pl = of_to_pennylane(H_trial)
                return self.compute_hf_energy(H_trial_pl)

            res = minimize_scalar(energy_tau, bounds=(-0.005,0.005), method='bounded')
            tau_opt = res.x

            comm1 = of_commutator(H_current, best_op_of)
            comm2 = of_commutator(comm1, best_op_of)

            H_current = (H_current + tau_opt * comm1 + 0.5 * tau_opt**2 * comm2)
            H_current.compress()
            
            energy = self.compute_hf_energy(of_to_pennylane(H_current))

            print("Term count:", len(H_current.terms))
            print("Gradient max:", best_grad)
            print("Tau:", tau_opt)
            print("Energy:", energy)
            
            self.convergence_history.append(energy)

        runtime = time.time() - start_time

        # Final bookkeeping
        ref_energy = (
            self.hf_energy
            if self.hf_energy is not None
            else self.hamiltonian.molecule.reference_energy
        )
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

        

            