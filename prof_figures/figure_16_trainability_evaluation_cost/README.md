# Figure 16: Trainability evaluation cost

Generate a PNG and PDF from the included sample:

```bash
python prof_figures/figure_16_trainability_evaluation_cost/generate.py
```

To use fresh results, pass `--input PATH` and optionally `--label "description of the run"` and `--output path/to/figure.png`. The script only reads saved results; it does not run a VQE.

Included sample: **Historical May 2026 data | target 10 only**. It is a plotting preview and is not a reproduction of the original paper figure. Inspect the run configuration and original data before drawing scientific conclusions.

Input: `all_molecules_results.csv` or a compatible results CSV. Rows must be unique for each molecule, algorithm and requested target; select a seed before plotting. The sample has only the 10-parameter target.

The sample is from the May 14, 2026 `trainability_batch/20260514_153030_batch_comparison` output.
