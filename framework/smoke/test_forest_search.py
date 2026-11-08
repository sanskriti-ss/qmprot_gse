# ABOUTME: Checks the energy forest's seed consumption and predictive moments against scikit-optimize.
# ABOUTME: Exercises native estimator fallbacks and the registered molecular optimization entry point.
import contextlib
from io import StringIO
from pathlib import Path
import unittest

import numpy as np
from skopt import Optimizer, forest_minimize
from skopt.space import Real

from algorithms import get_algorithm
from core.forest import EnergyForest, forest_search_options
from experiments.provenance import load_problem


class ForestSearch(unittest.TestCase):
    def test_native_seed_and_predictions(self):
        dimensions = [Real(-1., 1.) for _ in range(4)]
        for seed in (0, np.int64(43)):
            stock = Optimizer(dimensions, base_estimator="ET", n_jobs=1, random_state=seed)
            selected = Optimizer(dimensions, **forest_search_options(dimensions, seed))
            self.assertIsInstance(selected.base_estimator_, EnergyForest)
            self.assertEqual(stock.base_estimator_.get_params(), selected.base_estimator_.get_params())
            np.testing.assert_array_equal(stock.rng.randint(2**30, size=40), selected.rng.randint(2**30, size=40))
            rng = np.random.default_rng(37)
            x = rng.uniform(size=(32, 4))
            y = rng.normal(size=32)
            grid = rng.uniform(size=(100, 4))
            original = stock.base_estimator_.fit(x, y)
            candidate = selected.base_estimator_.fit(x, y)
            for points in (grid, grid.astype(np.float32), grid[:2].tolist()):
                for left, right in zip(original.predict(points, return_std=True),
                                       candidate.predict(points, return_std=True), strict=True):
                    self.assertEqual(left.tobytes(), right.tobytes())
        self.assertEqual(forest_search_options(dimensions, None), {})

    def test_molecular_gbrt_fallback_preserves_forest_search(self):
        h = load_problem(Path(__file__).parent / "reproductions/layered_vqe/hydrogen.npz").hamiltonian
        solver = get_algorithm("hardware_efficient_vqe")(h, n_layers=5, optimizer="BAYESIAN", bayes_backend="gbrt",
                                                         max_iterations=24, random_seed=43)
        solver.build_ansatz()
        start = np.random.default_rng(41).uniform(-.2, .2, solver.n_parameters)
        with self.assertLogs("core.base_vqe", level="WARNING") as warnings, contextlib.redirect_stderr(StringIO()):
            angles, energy = solver.optimize(start.copy())
        self.assertEqual(len(warnings.records), 1)
        self.assertIn("falling back to forest backend", warnings.output[0])
        self.assertEqual(solver.optimizer_outcome["optimizer_backend"], "forest")
        dimensions = [Real(*solver.bayesian_bounds(), name=f"theta_{i}") for i in range(solver.n_parameters)]
        stock = forest_minimize(lambda x: solver.cost_function(np.array(x)), dimensions, n_calls=24,
                                n_initial_points=19, random_state=43, x0=start.tolist())
        self.assertEqual(angles.tobytes(), np.asarray(stock.x).tobytes())
        self.assertEqual(float(energy).hex(), float(stock.fun).hex())
        self.assertEqual(np.asarray(solver.convergence_history).tobytes(), stock.func_vals.tobytes())
