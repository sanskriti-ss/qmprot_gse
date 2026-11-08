# ABOUTME: Checks fixed-circuit stopping reports against real optimizer termination conditions.
# ABOUTME: Exercises evaluation exhaustion, a successful solve, and a budget-only Bayesian study.
import contextlib
import io
import unittest

from algorithms import get_algorithm
from core.hamiltonian_loader import Molecule, QubitHamiltonian


class OptimizerTermination(unittest.TestCase):
    def run_constant(self, optimizer, budget):
        molecule = Molecule("pair", "pair", 2, 1, -1., "", n_electrons=1)
        hamiltonian = QubitHamiltonian.from_dict({"II": -1.}, molecule)
        solver = get_algorithm("pennylane_vqe")(hamiltonian, n_layers=1, optimizer=optimizer,
                                               max_iterations=budget, random_seed=42)
        with contextlib.redirect_stderr(io.StringIO()):
            return solver.run()

    def test_flat_energy_does_not_certify_a_budget_stop(self):
        result = self.run_constant("COBYLA", 12)
        self.assertFalse(result.converged)
        self.assertTrue(result.metadata["incumbent_stalled"])
        self.assertFalse(result.metadata["optimizer_success"])
        self.assertEqual(result.metadata["optimizer_evaluations"], 12)
        self.assertTrue(result.metadata["optimizer_message"])

    def test_scipy_success_is_recorded(self):
        result = self.run_constant("COBYLA", 256)
        self.assertTrue(result.converged)
        self.assertTrue(result.metadata["optimizer_success"])
        self.assertLess(result.metadata["optimizer_evaluations"], 256)

    def test_bayesian_budget_has_no_optimizer_convergence_flag(self):
        result = self.run_constant("BAYESIAN", 5)
        self.assertFalse(result.converged)
        self.assertIsNone(result.metadata["optimizer_success"])
        self.assertEqual(result.metadata["optimizer_evaluations"], 5)
