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