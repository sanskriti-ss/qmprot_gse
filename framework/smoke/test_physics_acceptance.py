# ABOUTME: Checks the small-matrix energy references used by the molecular solver audit.
# ABOUTME: Exercises complex phases, a linearly dependent trial basis and ordered Pauli wires.
import unittest

import numpy as np

from core.hamiltonian_loader import Molecule, QubitHamiltonian
from smoke.physics_acceptance import dense_hamiltonian, projections


class PhysicalReferences(unittest.TestCase):
    def test_complex_trials_and_rank_deficient_projection(self):
        hamiltonian = np.array([[0., 1.], [1., 0.]])
        determinant = np.array([1., 0.])
        phased = projections(np.array([0., 1.j]), hamiltonian, determinant)
        self.assertAlmostEqual(phased["complex"], -1.)
        self.assertAlmostEqual(phased["real"], 0.)
        duplicate = projections(determinant, hamiltonian, determinant)
        self.assertAlmostEqual(duplicate["complex"], 0.)
        self.assertAlmostEqual(duplicate["real"], 0.)

    def test_pauli_term_with_imaginary_matrix_elements(self):
        molecule = Molecule("two", "two", 2, 2, -1., "")
        hamiltonian = QubitHamiltonian.from_dict({"ZI": 2., "XY": 0.5}, molecule)
        matrix = dense_hamiltonian(hamiltonian)
        np.testing.assert_array_equal(matrix.diagonal(), [2., 2., -2., -2.])
        self.assertEqual(matrix[0, 3], -0.5j)
        self.assertEqual(matrix[1, 2], 0.5j)
        self.assertAlmostEqual(np.linalg.eigvalsh(matrix)[0], -np.sqrt(4.25))
