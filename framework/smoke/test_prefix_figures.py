# ABOUTME: Checks retained-term axes, QAOA exclusion and molecule cohorts for prefix-sweep figures.
# ABOUTME: Exercises the summary writer and inspects the resulting Matplotlib curves and SD bounds.
import csv
from dataclasses import replace
import json
from pathlib import Path
import tempfile
import unittest

import matplotlib
matplotlib.use("Agg")
from matplotlib import pyplot as plt
import numpy as np

from experiments.experiment5_term_count_sweep import TermCountRow, _prefix_figure, _write_summary_csv


class PrefixFigures(unittest.TestCase):
    def test_cohorts_and_full_prefix_range(self):
        base = TermCountRow(molecule="gly", algorithm="pennylane_vqe", n_parameters=26,
                            algorithm_n_parameters=8, n_qubits=4, n_terms_available=27, energy=-4.,
                            full_reference_energy=-5., prefix_reference_energy=-4., error_vs_full_reference=1.,
                            abs_error_vs_full_reference=1., error_vs_prefix_reference=0., abs_error_vs_prefix_reference=0.,
                            n_iterations=1, n_cost_evals=10, runtime_seconds=0.1, converged=False)
        rows = [base, replace(base, molecule="ala", abs_error_vs_full_reference=3.),
                replace(base, molecule="val", status="failed"), replace(base, molecule="his", abs_error_vs_full_reference=np.nan),
                replace(base, n_parameters=1, abs_error_vs_full_reference=0.),
                replace(base, algorithm="qaoa_inspired_vqe")]
        with tempfile.TemporaryDirectory() as directory:
            summary = Path(directory) / "summary.csv"
            _write_summary_csv(summary, rows)
            with summary.open() as f:
                data = list(csv.DictReader(f))
        high = next(row for row in data if row["algorithm"] == base.algorithm and row["n_parameters"] == "26")
        self.assertEqual(json.loads(high["molecules"]), ["ala", "gly"])
        self.assertEqual(int(high["n_samples"]), 2)
        self.assertEqual(float(high["mean_abs_error_vs_full_reference"]), 2.)
        self.assertEqual(float(high["std_abs_error_vs_full_reference"]), 1.)

        for filtered in (False, True):
            with self.subTest(filtered=filtered):
                fig = _prefix_figure(data, exclude_qaoa=filtered)
                try:
                    ax = fig.axes[0]
                    curves = {line.get_label(): line for line in ax.get_lines()}
                    self.assertEqual("qaoa_inspired_vqe" in curves, not filtered)
                    self.assertEqual("QAOA-inspired VQE excluded" in ax.get_title(), filtered)
                    np.testing.assert_array_equal(curves[base.algorithm].get_xdata(), [1, 26])
                    np.testing.assert_array_equal(curves[base.algorithm].get_ydata(), [0., 2.])
                    self.assertGreaterEqual(ax.get_xlim()[1], 26.)
                    self.assertLessEqual(ax.get_ylim()[0], 0.)
                    self.assertIn("population SD", ax.get_legend().get_texts()[-1].get_text())
                    band = ax.collections[0].get_paths()[0].vertices
                    np.testing.assert_array_equal(np.unique(band[band[:, 0] == 26, 1]), [1., 3.])
                finally:
                    plt.close(fig)
