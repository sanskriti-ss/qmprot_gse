from dataclasses import dataclass
from typing import Dict
import itertools

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

def of_to_pennylane(of_operator):
    import pennylane as qml
    coeffs = []
    ops = []

    for term, coeff in of_operator.terms.items():
        if not term:
            coeffs.append(coeff)
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

        op = pauli_ops[0]
        for next_op in pauli_ops[1:]:
            op = op @ next_op

        coeffs.append(coeff)
        ops.append(op)

    return qml.Hamiltonian(coeffs, ops)
