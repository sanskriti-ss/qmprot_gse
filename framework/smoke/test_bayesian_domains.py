# ABOUTME: Verifies molecular starting energies and the Bayesian domain for each fixed circuit family.
# ABOUTME: Checks gate periodicity and rejects invalid starts for weighted Hamiltonian evolution.
import contextlib
from io import StringIO
from pathlib import Path
import unittest

import numpy as np

from algorithms import get_algorithm
from experiments.provenance import load_problem


class BayesianDomains(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.hydrogen = load_problem(Path(__file__).parent / "reproductions/layered_vqe/hydrogen.npz").hamiltonian

    def test_frozen_energy_of_first_real_acquisition(self):
        for name, period in (("hva_vqe", None), ("ucc_vqe", 4*np.pi),
                             ("qaoa_inspired_vqe", 2*np.pi), ("hardware_efficient_vqe", 2*np.pi),
                             ("pennylane_vqe", 2*np.pi), ("nn_ae_vqe", 2*np.pi), ("cb_vqe", 2*np.pi)):
            with self.subTest(method=name):
                solver = get_algorithm(name)(self.hydrogen, n_layers=2, random_seed=42,
                                              optimizer="BAYESIAN", max_iterations=5)
                solver.build_ansatz()
                theta = np.random.default_rng(42).uniform(0, 2*np.pi, solver.n_parameters)
                energy = solver.cost_function(theta)
                with contextlib.redirect_stderr(StringIO()):
                    solver.optimize(theta)
                self.assertAlmostEqual(solver.convergence_history[0], energy, places=10)
                self.assertEqual(solver.parameter_period, period)
                if period is not None:
                    for index in range(theta.size):
                        shifted = theta.copy()
                        shifted[index] += period
                        self.assertAlmostEqual(solver.cost_function(shifted), energy, places=10)
                half_width = np.pi if period == 2*np.pi else 2*np.pi
                self.assertEqual(solver.optimizer_outcome["bayes_bounds"], [-half_width, half_width])

    def test_hamiltonian_evolution_uses_declared_literal_box(self):
        solver = get_algorithm("hva_vqe")(self.hydrogen, n_layers=2, optimizer="BAYESIAN",
                                          bayes_bounds=(-.1, .1), max_iterations=5)
        solver.build_ansatz()
        with self.assertRaisesRegex(ValueError, "outside the fixed Bayesian bounds"):
            solver.optimize(np.ones(solver.n_parameters))
