# ABOUTME: Verifies ordered Hamiltonian entries and reference energies against explicit Kronecker products.
# ABOUTME: Covers occupied determinants, cutoff boundaries, Pauli phases, duplicates and alternate labels.
from functools import reduce
from itertools import product
import unittest

import numpy as np

from core.hamiltonian_loader import Molecule, QubitHamiltonian
from core.hf_verification import _build_hamiltonian_matrix, _parse_pauli_string, compute_hf_energy
from contextual_subspace.core import pauli_to_sparse


PAULIS = dict(I=np.eye(2, dtype=complex), X=np.array([[0, 1], [1, 0]], complex),
              Y=np.array([[0, -1j], [1j, 0]], complex), Z=np.diag([1., -1.]).astype(complex))


class DeterminantEnergy(unittest.TestCase):
    def test_ordered_entries_at_every_occupation(self):
        n = 4
        labels = ["".join(word) for word in product("IXYZ", repeat=n)]
        labels += ["Z(3)", "Identity", "Z(0) @ Z(3)", "IIII", "IIII", "IIII"]
        coefficients = np.random.default_rng(42).normal(size=len(labels))
        coefficients[-3:] = [2e-13, 1e-15, 1e-16]
        mol = Molecule("test", "test", n, len(labels), 0., "")
        ham = QubitHamiltonian(mol, coefficients, labels, n, len(labels))
        dense = np.zeros((1 << n, 1 << n), complex)
        for coeff, label in zip(coefficients, labels):
            if abs(coeff) >= 1e-15:
                word = _parse_pauli_string(label, n)
                dense += coeff * reduce(np.kron, [PAULIS[p] for p in word])
        self.assertEqual(_build_hamiltonian_matrix(ham).tobytes(), dense.tobytes())
        for row in range(1 << n):
            self.assertEqual(_build_hamiltonian_matrix(ham, row=row).tobytes(), dense[row].tobytes())
        for occupied in range(n+1):
            index = ((1 << occupied) - 1) << (n - occupied)
            self.assertEqual(compute_hf_energy(ham, occupied).hex(), dense[index, index].real.item().hex())
        ham.pauli_strings[0] = "XB"
        with self.assertLogs("core.hf_verification", level="ERROR") as log, self.assertRaises(KeyError):
            compute_hf_energy(ham, 1)
        self.assertIn("Error in compute_hf_energy", log.output[0])

    def test_rotation_matrix_preserves_complex_phase_bytes(self):
        for word in map("".join, product("IXYZ", repeat=4)):
            x = int("".join("0" if p in "IZ" else "1" for p in word), 2)
            z = int("".join("0" if p in "IX" else "1" for p in word), 2)
            rows = range(16)
            values = np.array([(-1.)**((r & z).bit_count()) * ((-1j)**word.count("Y")) for r in rows])
            matrix = pauli_to_sparse(word)
            self.assertEqual(matrix.data.tobytes(), values.tobytes())
            np.testing.assert_array_equal(matrix.indices, [r ^ x for r in rows])
