# Figure 16: Trainability evaluation cost

Generate a PNG and PDF from the included sample:

```bash
python prof_figures/figure_16_trainability_evaluation_cost/generate.py
```

To use fresh results, pass `--input PATH` and optionally `--label "description of the run"` and `--output path/to/figure.png`. The script only reads saved results; it does not run a VQE.

Included sample: **Historical May 2026 data | target 10 only**. It is a plotting preview and is not a reproduction of the original paper figure. Inspect the run configuration and original data before drawing scientific conclusions.

Input: `all_molecules_results.csv` or a compatible results CSV. Each requested target gets its own panel; molecules share an evaluation axis. The sample has only the 10-parameter target. Hollow markers indicate that at least one run's attained parameter count differs from the requested target. A cost count is not a measure of the energy achieved; compare it with an energy or threshold-crossing figure.

With repeated runs, include a unique `seed`, `trial_seed`, or `random_seed` for each molecule, algorithm and target. The markers show the geometric mean and 95% Student-t intervals calculated in log space when two or more seeds are present. One-seed conditions have no interval. Rows with zero or missing counts cannot appear on the logarithmic axis and are counted in the subtitle.

The sample is from the May 14, 2026 `trainability_batch/20260514_153030_batch_comparison` output.
