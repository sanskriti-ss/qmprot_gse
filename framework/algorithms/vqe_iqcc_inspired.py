# ABOUTME: Grows an iQCC-inspired circuit using a Pauli generator pool.
# ABOUTME: Reoptimizes the circuit angles against the original molecular Hamiltonian.
'''
Iterative Qubit Coupled Cluster (iQCC)-Inspired Adaptive VQE Implementation

Draws from the adaptive operator selection of iQCC to achieve results which largely reflect those achieved with iQCC with far greater 
computational efficiency.

'''

from functools import cache

import numpy as np
from typing import Optional, Any
import logging

import sys
import time
sys.path.append('..')
from core.base_vqe import BaseVQE, VQEResult
from core.hamiltonian_loader import QubitHamiltonian
from core.iqcc_helpers import IQCC_Operator, PauliOperatorPool, basis_commutator_support, of_to_pennylane
from core.algebraic_operators import of_commutator

import pennylane as qml
from core.backend_manager import create_device

logger = logging.getLogger(__name__)

class iQCC_inspired_VQE(BaseVQE):
    def __init__(self, 
                 hamiltonian: QubitHamiltonian, 
                 max_operators: int=20, 
                 gradient_threshold: float= 1e-4,
                 **kwargs):
        super().__init__(hamiltonian, **kwargs)
        if self.backend_config.is_noisy and self.backend_config.noise_strength != 0:
            raise ValueError("iQCC-inspired VQE does not implement noise channels during screening")
        self.name = 'iqcc_inspired_vqe'
        self.description = 'Iterative Qubit Coupled Cluster-inspired adaptive VQE.'
        self.max_operators = max_operators
        self.gradient_threshold = gradient_threshold

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
        #hf_bitstring = np.zeros(n_qubits, dtype=int)
        #hf_bitstring[:self.hamiltonian.molecule.n_electrons] = 1
        @qml.qnode(self.device)
        def circuit(p):
            self._prepare_initial_state()

            for theta, op in zip(p, self.selected_operators):
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
        if np.shape(parameters) != (len(self.selected_operators),):
            raise ValueError("Expected exactly one angle per selected Pauli generator")
        return float(self.cost_fn(parameters))
    

    def run(self) -> VQEResult:
        from openfermion import QubitOperator
        from scipy.optimize import minimize

        logger.info(f"Running {self.name} on {self.hamiltonian.molecule.name}")
        self._perform_hf_verification()

        start_time = time.time()

        # Initial empty ansatz
        self.selected_operators = []
        self.parameters = np.array([])
        self.convergence_history = []
        self.build_ansatz()
        energy = self.cost_function(self.parameters)
        print("Initial energy:", energy)
        print("HF reference:", self.hf_energy)

        selected_generators = []
        n_energy_evals = 1
        n_screen_sweeps = 0
        n_commutator_builds = 0
        n_materialized_commutators = 0
        n_screen_expectations = 0
        n_screen_support_zeros = 0
        stop_reason = "operator limit"

        def screen_operator(op):
            nonlocal n_commutator_builds
            n_commutator_builds += 1
            generator = QubitOperator(op)
            def observable():
                nonlocal n_materialized_commutators
                n_materialized_commutators += 1
                comm = 1j * of_commutator(H_of, generator)
                return of_to_pennylane(comm) if comm.terms else None
            return next(iter(generator.terms)), cache(observable) if cache_observables else observable

        if self.max_operators > 0:
            H_of = self.hamiltonian.to_openfermion()
            support = basis_commutator_support(self, H_of, coefficient_bound=4)
            active_wires = 0
            # Each commutator has at most one Pauli word per Hamiltonian term.
            pauli_factor_bound = len(self.operator_pool) * max(1, len(H_of.terms)) * self.n_qubits
            cache_observables = self.max_operators > 1 and pauli_factor_bound <= 8192
            if cache_observables:
                screen_operator = cache(screen_operator)
        for k in range(self.max_operators):
            n_screen_sweeps += 1

            logger.info(f"iQCC step {k+1}/{self.max_operators}")

            # Compute gradients over pool
            gradients = {}
            for op in self.operator_pool:
                term, observable = screen_operator(op)
                if support is not None:
                    classification = support.classify(term, active_wires)
                    if classification == "commutes":
                        continue
                    if classification == "disjoint":
                        gradients[op] = 0.
                        n_screen_expectations += 1
                        n_screen_support_zeros += 1
                        continue
                comm_pl = observable()
                if comm_pl is None:
                    continue
                grad = self.compute_expectation(comm_pl, self.parameters)
                n_screen_expectations += 1
                #print(grad)
                gradients[op] = abs(float(grad))
            print("Number of gradients", len(gradients))
            if not gradients:
                stop_reason = "gradient threshold"
                break
            print("max gradient:", max(gradients.values()))

            best_op, best_grad = max(gradients.items(), key=lambda x: x[1])
            
            logger.info(f"Selected operator with gradient {best_grad:.3e}")

            if best_grad < self.gradient_threshold:
                stop_reason = "gradient threshold"
                break

            # Append operator
            self.selected_operators.append(QubitOperator(best_op))
            selected_generators.append(best_op)
            if support is not None:
                # Gates on selected wires can populate either bit value on those wires.
                for wire, _ in next(iter(self.selected_operators[-1].terms)):
                    active_wires |= 1 << wire

            self.parameters = np.append(self.parameters, 0.0)

            self.build_ansatz()

            # Optimize all parameters
            result = minimize(
                self.cost_function,
                self.parameters,
                method="COBYLA",
                options={"maxiter": 100}
            )
            self.parameters = result.x
            energy = result.fun
            n_energy_evals += result.nfev
            self.convergence_history.append(energy)

        self.iteration_count = len(self.selected_operators)
        self.n_parameters = len(self.parameters)
        self.optimal_parameters = self.parameters.copy()
        self.optimal_energy = energy
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
            relative_error=abs(error / ref_energy) if ref_energy else 0.,
            n_iterations=self.iteration_count,
            n_qubits=self.n_qubits,
            n_parameters=len(self.parameters),
            runtime_seconds=runtime,
            convergence_history=self.convergence_history.copy(),
            optimal_parameters=self.optimal_parameters.copy(),
            converged=stop_reason == "gradient threshold",
            metadata={
                "n_selected_operators": len(self.selected_operators),
                "max_operators": self.max_operators,
                "selected_generators": selected_generators,
                "rotation_convention": "exp(-i theta P), first selected generator acts first",
                "stop_reason": stop_reason,
                "inner_optimizer": "COBYLA",
                "inner_max_iterations": 100,
                "n_molecular_energy_evals": int(n_energy_evals),
                "n_screen_sweeps": n_screen_sweeps,
                "n_screen_expectations": n_screen_expectations,
                "n_screen_support_zeros": n_screen_support_zeros,
                "n_commutator_builds": n_commutator_builds,
                "n_materialized_commutators": n_materialized_commutators,
            },
            backend_type=self.backend_config.backend_type,
            noise_model=self.backend_config.noise_model,
            noise_strength=self.backend_config.noise_strength,
            hf_energy=self.hf_energy,
        )

        

            