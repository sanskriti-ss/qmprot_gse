# ABOUTME: Checks aggregate evaluation markers against real hydrogen trainability runs.
# ABOUTME: Covers every supported algorithm and an incompletely observed target grid.
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest

from matplotlib.figure import Figure
import numpy as np

from experiments.trainability import _load_trainability_rows, _plot_trainability_costs
from smoke.batch import FRAMEWORK


class TrainabilityFigures(unittest.TestCase):
    def test_three_algorithms_and_missing_target(self):
        order = ["qubit_adapt_vqe", "hardware_efficient_vqe", "adapt_vqe"]
        with tempfile.TemporaryDirectory() as tmp:
            process = subprocess.run(
                [sys.executable, "-m", "experiments.trainability", "--molecules", "hydrogen",
                 "--algorithms", *reversed(order), "--param-targets", "1", "4", "--max-iter", "15",
                 "--eval-budget", "44", "--output-root", tmp],
                cwd=FRAMEWORK, capture_output=True, text=True,
                env=dict(os.environ, QMPROT_GEOMETRIES=str(FRAMEWORK / "smoke/geometries")),
            )
            log = process.stdout + process.stderr
            self.assertEqual(process.returncode, 0, log)
            for token in ("[ERROR]", "[WARNING]", "Warning:", "Traceback"):
                self.assertNotIn(token, log)
            self.assertTrue((Path(tmp) / "all_molecules_cost_evals_vs_params.png").is_file())
            rows = _load_trainability_rows(Path(tmp) / "hydrogen")

        self.assertEqual(len(rows), 6)
        self.assertTrue(any(row.actual_n_params < row.target_n_params for row in rows))
        cohort = [r for r in rows if (r.algorithm, r.target_n_params) != ("adapt_vqe", 4)]
        ax = Figure().subplots()
        _plot_trainability_costs(ax, [("hydrogen", r) for r in cohort])
        self.assertEqual([line.get_label() for line in ax.lines], [f"{algo} median" for algo in order])
        for algo, line in zip(order, ax.lines):
            costs = {r.target_n_params: r.n_cost_evals for r in cohort if r.algorithm == algo}
            np.testing.assert_equal(line.get_ydata(), [costs.get(target, np.nan) for target in (1, 4)])
        points = [point for series in ax.collections for point in series.get_offsets()]
        self.assertCountEqual([(int(round(x)), int(y)) for x, y in points],
                              [((1, 4).index(r.target_n_params), r.n_cost_evals) for r in cohort])
        self.assertEqual(ax.get_xlabel(), "Requested parameter count")
        self.assertEqual(ax.get_ylabel(), "Cost-function evaluations spent")
