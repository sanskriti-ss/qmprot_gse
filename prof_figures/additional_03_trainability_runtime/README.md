# Trainability runtime

This is an **additional appendix view**, not a numbered figure in the current paper.

Generate the included preview from the repository root:

```bash
python prof_figures/additional_03_trainability_runtime/generate.py
```

Supply a fresh CSV with `--input /path/to/results.csv`, and optionally `--label "run description"` and `--output /path/to/figure.png`. The generator reads saved data and does not launch VQE calculations. PNG and PDF files are written together.

Included sample: **Historical May 2026 trainability batch | target 10 only**. These historical outputs predate the molecule-ID fixes and are illustrative, not validated fresh results.

Input: trainability `all_molecules_results.csv` or a compatible batch CSV. Runtime is wall-clock seconds recorded by the experiment and depends on hardware and implementation. The sample from the May 14 batch contains target 10 only.

## Uncertainty

The included preview has one run per condition, so it does not support a confidence interval. With two or more independently seeded runs per condition, combine their rows in the input CSV and retain a unique `seed`, `trial_seed`, or `random_seed` column. The figure shows a two-sided 95% Student-t interval across seeds at each condition. Intervals are absent where only one seed is available; a zero-width interval can also arise from identical values. Runtime uses the geometric mean and an interval calculated on log runtime. For noisy energy, each seed must have its own clean baseline repeated consistently across its noise conditions. These intervals describe variation across the sampled seeds, not uncertainty from shots, systematic error, or a population of molecules. Very small seed counts give unstable intervals.
