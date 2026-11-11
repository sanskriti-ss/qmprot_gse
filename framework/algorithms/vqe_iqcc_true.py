# ABOUTME: Dresses a molecular Hamiltonian by exact conjugation with selected Pauli rotations.
# ABOUTME: Rescores each accepted rotation product on the original Hamiltonian and reference state.
"""Iterative qubit coupled cluster with U(tau) = exp(-i tau P / 2)."""
import logging
import time

import numpy as np
from openfermion import QubitOperator
import pennylane as qml

from core.backend_manager import create_device
from core.base_vqe import BaseVQE, VQEResult
from core.hamiltonian_loader import QubitHamiltonian
from core.iqcc_helpers import PauliOperatorPool, basis_commutator_support, of_to_pennylane

logger = logging.getLogger(__name__)


def _pauli_hamiltonian(hamiltonian):
    result = QubitOperator()
    for word, coefficient in zip(hamiltonian.pauli_strings, hamiltonian.coefficients):
        if "(" in word:
            word = hamiltonian._openfermion_to_pauli_string(word, hamiltonian.n_qubits)
        if any(p not in "IXYZ" for p in word) or any(p != "I" for p in word[hamiltonian.n_qubits:]):
            raise ValueError(f"Invalid Pauli word {word!r}")
        if not np.isfinite(coefficient) or np.imag(coefficient) != 0:
            raise ValueError("True iQCC requires finite real coefficients for Hermitian Pauli words")
        term = tuple((wire, p) for wire, p in enumerate(word) if p != "I")
        result.terms[term] = result.terms.get(term, 0.) + float(np.real(coefficient))
    return result


def _rotation_terms(hamiltonian, generator):
    """Anticommuting part A of H and derivative B = i P A for one unit Pauli P."""
    anti = QubitOperator()
    derivative = QubitOperator()
    (generator_term, generator_coefficient), = generator.terms.items()
    for term, coefficient in hamiltonian.terms.items():
        product = QubitOperator(generator_term + term, generator_coefficient)
        (word, phase), = product.terms.items()
        if phase.imag:
            anti.terms[term] = coefficient
            derivative.terms[word] = -phase.imag * coefficient
    return anti, derivative


def _dress(hamiltonian, anti, derivative, angle):
    """U†HU leaves commuting terms fixed and rotates A into cos(tau) A + sin(tau) B."""
    dressed = QubitOperator()
    terms = hamiltonian.terms.copy()
    cosine, sine = np.cos(angle), np.sin(angle)
    for word, coefficient in anti.terms.items():
        terms[word] = cosine * coefficient
    for word, coefficient in derivative.terms.items():
        terms[word] = terms.get(word, 0.) + sine * coefficient
    dressed.terms = {word: coefficient for word, coefficient in terms.items() if coefficient != 0}
    return dressed


def _optimal_angle(anti_energy, slope, bound):
    """Minimize A(cos(tau)-1) + B sin(tau) on the closed, symmetric interval."""
    minimum = np.arctan2(-slope, -anti_energy)
    candidates = (0., -bound, bound, float(np.clip(minimum, -bound, bound)))
    return min(candidates, key=lambda tau: -2 * anti_energy * np.sin(tau / 2)**2 + slope * np.sin(tau))


class iQCC_true_VQE(BaseVQE):
    def __init__(self, 
                 hamiltonian: QubitHamiltonian, 
                 max_operators: int=20, 
                 gradient_threshold: float= 1e-4,
                 bound: float=0.005,
                 **kwargs):
        super().__init__(hamiltonian, **kwargs)
        if self.backend_config.is_noisy and self.backend_config.noise_strength != 0:
            raise ValueError("True iQCC does not implement noise channels during Hamiltonian dressing")
        self.name = 'iqcc_true_vqe'
        self.description = 'Iterative Qubit Coupled Cluster'
        self.max_operators = max_operators
        self.gradient_threshold = gradient_threshold
        if not np.isfinite(bound) or bound < 0:
            raise ValueError("The rotation bound must be finite and nonnegative")
        self.bound = bound

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
        self.device = create_device(self.backend_config)
        original = _pauli_hamiltonian(self.hamiltonian)
        H = of_to_pennylane(original) if original.terms else 0. * qml.Identity(0)

        @qml.qnode(self.device)
        def circuit(params):
            self._prepare_initial_state()
            # Successive dressings produce U1 U2 ... Uk; Uk acts on the reference first.
            for index in reversed(range(len(self.selected_operators))):
                term, = QubitOperator(self.selected_operators[index]).terms
                word = "".join(pauli for _, pauli in term)
                wires = [wire for wire, _ in term]
                qml.PauliRot(params[index], word, wires=wires)
            return qml.expval(H)

        self.cost_fn = circuit
        return circuit

    def compute_hf_energy(self, observable):
        @qml.qnode(self.device)
        def circuit():
            self._prepare_initial_state()
            return qml.expval(observable)

        return float(circuit())

    def cost_function(self, parameters: np.ndarray) -> float:
        """Score the supplied angles against the original molecular Hamiltonian."""
        if self.cost_fn is None:
            raise RuntimeError("Must call build_ansatz() first")
        if np.shape(parameters) != (len(self.selected_operators),):
            raise ValueError("Expected exactly one angle per selected Pauli generator")
        return float(self.cost_fn(parameters))

    def run(self) -> VQEResult:
        logger.info(f"Running {self.name} on {self.hamiltonian.molecule.name}")
        self._perform_hf_verification()
        start_time = time.time()
        self.selected_operators = []
        self.parameters = np.array([])
        self.convergence_history = []
        self.build_ansatz()
        energy = self.cost_function(self.parameters)
        dressed = _pauli_hamiltonian(self.hamiltonian)
        stop_reason = "operator limit"
        n_screen_sweeps = 0
        n_commutator_builds = 0
        n_materialized_commutators = 0
        n_screen_expectations = 0
        n_screen_basis_zeros = 0
        n_angle_expectations = 0

        for step in range(self.max_operators):
            logger.info(f"iQCC step {step+1}/{self.max_operators}")
            n_screen_sweeps += 1
            best = None
            best_gradient = 0.
            # Finite operator sums preserve exact structural zeros in Lightning's statevector contraction.
            support = basis_commutator_support(self, dressed, coefficient_bound=2)
            for word in self.operator_pool:
                generator = QubitOperator(word)
                n_commutator_builds += 1
                if support is not None:
                    classification = support.classify(next(iter(generator.terms)))
                    if classification == "commutes":
                        continue
                    if classification == "disjoint":
                        n_screen_expectations += 1
                        n_screen_basis_zeros += 1
                        continue
                anti, derivative = _rotation_terms(dressed, generator)
                n_materialized_commutators += 1
                if not derivative.terms:
                    continue
                n_screen_expectations += 1
                gradient = self.compute_hf_energy(of_to_pennylane(derivative))
                if abs(gradient) > abs(best_gradient):
                    best_gradient = gradient
                    best = word, anti, derivative
            if best is None or abs(best_gradient) < self.gradient_threshold:
                stop_reason = "gradient threshold"
                break

            word, anti, derivative = best
            anti_energy = self.compute_hf_energy(of_to_pennylane(anti))
            n_angle_expectations += 1
            angle = _optimal_angle(anti_energy, best_gradient, self.bound)
            if angle == 0:
                stop_reason = "no allowed decrease"
                break
            self.selected_operators.append(word)
            self.parameters = np.append(self.parameters, angle)
            dressed = _dress(dressed, anti, derivative, angle)
            energy = self.cost_function(self.parameters)
            self.convergence_history.append(energy)
            logger.info(f"Selected {word}, angle {angle:.8g}, original-H energy {energy:.12f} Ha")

        self.iteration_count = len(self.selected_operators)
        self.n_parameters = len(self.parameters)
        self.optimal_parameters = self.parameters.copy()
        self.optimal_energy = energy
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
            n_parameters=self.n_parameters,
            runtime_seconds=time.time() - start_time,
            convergence_history=self.convergence_history.copy(),
            optimal_parameters=self.optimal_parameters.copy(),
            converged=stop_reason == "gradient threshold",
            metadata={
                "n_selected_operators": self.n_parameters,
                "selected_generators": self.selected_operators.copy(),
                "rotation_convention": "exp(-i tau P/2), rightmost selected generator acts first",
                "max_operators": self.max_operators,
                "angle_bound": self.bound,
                "energy_measurement": "sampled" if self.backend_config.n_shots else "analytic",
                "stop_reason": stop_reason,
                "n_molecular_energy_evals": 1 + self.iteration_count,
                "n_screen_sweeps": n_screen_sweeps,
                "n_screen_expectations": n_screen_expectations,
                "n_screen_basis_zeros": n_screen_basis_zeros,
                "n_commutator_builds": n_commutator_builds,
                "n_materialized_commutators": n_materialized_commutators,
                "n_angle_expectations": n_angle_expectations,
            },
            backend_type=self.backend_config.backend_type,
            noise_model=self.backend_config.noise_model,
            noise_strength=self.backend_config.noise_strength,
            hf_energy=self.hf_energy,
        )
