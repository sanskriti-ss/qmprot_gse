# Figure 13: COBYLA initialization convergence

Generate a PNG and PDF from the included sample:

```bash
python prof_figures/figure_13_cobyla_initialization_convergence/generate.py
```

To use fresh results, pass `--input PATH` and optionally `--label "description of the run"` and `--output path/to/figure.png`. The script only reads saved results; it does not run a VQE.

Included sample: **Historical May 2026 data | 16 molecules | COBYLA**. It is a plotting preview and is not a reproduction of the original paper figure. Inspect the run configuration and original data before drawing scientific conclusions.

Input: a CSV with molecule, strategy, trial, iteration and energy, a batch output with `convergence_traces_all.csv`, or per-molecule folders with `convergence_traces.csv` and `run_config.json`. Each figure must use one depth and one run per molecule.

The sample is from the May 14, 2026 `barren_plateau_batch/20260514_153420_batch_hardware_efficient_vqe` output. Historical molecule identifiers need independent verification.
