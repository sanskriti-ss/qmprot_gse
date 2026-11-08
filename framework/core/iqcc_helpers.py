# ABOUTME: Defines Pauli operator pools and Hamiltonian transforms for iQCC.
# ABOUTME: Builds the observables consumed by the iQCC objectives.
from dataclasses import dataclass
from typing import Dict
import itertools
import math

import numpy as np

from core.hamiltonian_loader import pauli_product

@dataclass(frozen=True)
class IQCC_Operator:
    '''
    Represents a generator A = i * P, where P is a Pauli word.
    '''
    pauli_word: str
    coeff: float = 1.0
    def __str__(self):
        return f"{self.coeff} * {self.pauli_word}"
    
class PauliOperatorPool:
    def __init__(self, max_weight=2):
        self.max_weight = max_weight

    def generate(self, n_qubits):
        paulis = ["X", "Y", "Z"]
        pool = []

        for weight in range(1, self.max_weight + 1):
            for qubits in itertools.combinations(range(n_qubits), weight):
                for ops in itertools.product(paulis, repeat=weight):
                    word = " ".join(f"{op}{q}" for op, q in zip(ops, qubits))
                    pool.append(word)
        return pool


class PauliCommutatorSupport:
    """Pauli support of [H, P] on wires that can vary from a basis reference."""
    def __init__(self, hamiltonian):
        self.z_by_flip = {}
        for term in hamiltonian.terms:
            x, z = self.masks(term)
            self.z_by_flip.setdefault(x, []).append(z)

    @staticmethod
    def masks(term):
        flip = sum(1 << wire for wire, pauli in term if pauli in "XY")
        phase = sum(1 << wire for wire, pauli in term if pauli in "YZ")
        return flip, phase

    def classify(self, generator, active_wires=0):
        x, z = self.masks(generator)
        classification = "commutes"
        for hx, phases in self.z_by_flip.items():
            # A product flipping a fixed wire connects orthogonal basis sectors.
            can_overlap = ((hx ^ x) & ~active_wires) == 0
            if can_overlap or classification == "commutes":
                if any(((x & hz) ^ (z & hx)).bit_count() % 2 for hz in phases):
                    if can_overlap:
                        return "overlap"
                    classification = "disjoint"
        return classification


def basis_commutator_support(solver, hamiltonian, coefficient_bound):
    """Restrict exact-zero screening to finite analytic Lightning contractions on a basis ket."""
    reference = solver.hamiltonian.cs_initial_state
    if (solver.backend_config.backend_type != "statevector" or solver.device.name != "lightning.qubit"
            or solver.device.shots.total_shots is not None or reference is not None and np.count_nonzero(reference) != 1):
        return None
    values = hamiltonian.terms.values()
    if all(c.imag == 0 for c in values):
        limit = np.finfo(solver.device.c_dtype).max / coefficient_bound
        if sum(abs(float(c.real)) for c in values) < limit:
            return PauliCommutatorSupport(hamiltonian)
    return None

def of_to_pennylane(of_operator):
    """Convert a Hermitian Pauli sum into a measurable PennyLane observable."""
    import pennylane as qml
    coeffs = []
    ops = []

    for term, coeff in of_operator.terms.items():
        if coeff.imag != 0 or not math.isfinite(coeff.real):
            raise ValueError(f"Expected a finite real Pauli coefficient for {term}, got {coeff}")
        coeffs.append(float(coeff.real))
        if not term:
            ops.append(qml.Identity(0))
            continue

        pauli_ops = []
        for wire, pauli in term:
            if pauli == 'X':
                pauli_ops.append(qml.PauliX(wire))
            elif pauli == 'Y':
                pauli_ops.append(qml.PauliY(wire))
            elif pauli == 'Z':
                pauli_ops.append(qml.PauliZ(wire))

        ops.append(pauli_product(pauli_ops))

    return qml.Hamiltonian(coeffs, ops)

def prune_hamiltonian(H, threshold=1e-8, max_terms=None):
    """
    Remove small-coefficient terms and optionally cap total term count.
    """
    new_terms = {}

    for term, coeff in H.terms.items():
        if abs(coeff) > threshold:
            new_terms[term] = coeff

    # Optional hard cap
    if max_terms is not None and len(new_terms) > max_terms:
        # Keep largest coefficients only
        sorted_terms = sorted(
            new_terms.items(),
            key=lambda x: abs(x[1]),
            reverse=True
        )[:max_terms]
        new_terms = dict(sorted_terms)

    from openfermion import QubitOperator
    H_pruned = QubitOperator()
    H_pruned.terms = new_terms
    return H_pruned
