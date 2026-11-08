# ABOUTME: Checks algebraic screening of iQCC commutators against symbolic products and actual device measurements.
# ABOUTME: Covers the support of rotated reference states, commuting generators and sampled execution.
from contextlib import redirect_stdout
from io import StringIO
import unittest

import numpy as np
from openfermion import QubitOperator, commutator

from algorithms.vqe_iqcc_inspired import iQCC_inspired_VQE
from core.backend_manager import BackendConfig
from core.hamiltonian_loader import Molecule, QubitHamiltonian
from core.iqcc_helpers import PauliCommutatorSupport, of_to_pennylane


class PauliSupportTest(unittest.TestCase):
    def test_logical_screen_and_backend_work_on_zero_gradient_tie(self):
        h = QubitHamiltonian.from_dict({"Z": 1.}, Molecule("s", "s", 1, 1, -1., "", n_electrons=1))
        for device in ("lightning.qubit", "default.qubit"):
            config = BackendConfig.statevector(1, device_name=device)
            solver = iQCC_inspired_VQE(h, max_operators=1, gradient_threshold=0., backend_config=config)
            with redirect_stdout(StringIO()) as output:
                result = solver.run()
            self.assertIn("Number of gradients 2", output.getvalue())
            self.assertEqual(result.metadata["selected_generators"], ["X0"])
            self.assertEqual(result.metadata["n_commutator_builds"], 3)
            self.assertEqual(result.metadata["n_screen_expectations"], 2)
            self.assertEqual(result.metadata["n_screen_support_zeros"], 2 if device == "lightning.qubit" else 0)
            self.assertEqual(result.metadata["n_materialized_commutators"], 0 if device == "lightning.qubit" else 3)

    def test_symbolic_classification_and_rotated_expectations(self):
        ham = QubitHamiltonian.from_dict({"III": 2.1, "XZZ": .37, "XYZ": -.5, "YYX": .23,
                                          "ZZI": 1.25, "XYY": .91, "IXZ": -.63},
                                         Molecule("h", "h", 3, 7, -1., "", n_electrons=2))
        solver = iQCC_inspired_VQE(ham, backend_config=BackendConfig.statevector(3, device_name="lightning.qubit"))
        solver.build_ansatz()
        h_of = ham.to_openfermion()
        support = PauliCommutatorSupport(h_of)
        measured = 0
        for selected, active in (([], 0), (["X0"], 1), (["X0 Y1", "Y0 Z1"], 3)):
            angles = np.array([.29, -.1])[:len(selected)]
            solver.selected_operators = [QubitOperator(word) for word in selected]
            for word in solver.operator_pool:
                generator = QubitOperator(word)
                derivative = 1j * commutator(h_of, generator)
                result = support.classify(next(iter(generator.terms)), active)
                if not derivative.terms:
                    self.assertEqual(result, "commutes")
                elif any(all(pauli == "Z" or (1 << wire) & active for wire, pauli in term)
                         for term in derivative.terms):
                    self.assertEqual(result, "overlap")
                else:
                    self.assertEqual(result, "disjoint")
                    energy = solver.compute_expectation(of_to_pennylane(derivative), angles)
                    self.assertEqual(abs(energy).hex(), (0.).hex())
                    measured += 1
        self.assertGreater(measured, 20)
