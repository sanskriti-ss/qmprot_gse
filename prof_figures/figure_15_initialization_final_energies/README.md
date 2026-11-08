# Figure 15: Final energies by initialization

Generate a PNG and PDF from the included sample:

```bash
python prof_figures/figure_15_initialization_final_energies/generate.py
```

To use fresh results, pass `--input PATH` and optionally `--label "description of the run"` and `--output path/to/figure.png`. The script only reads saved results; it does not run a VQE.

Included sample: **Hydrogen pilot only | Bayesian | one trial per strategy**. It is a plotting preview and is not a reproduction of the original paper figure. Inspect the run configuration and original data before drawing scientific conclusions.

Input: a per-molecule output or parent directory containing `optimization_results.csv` and `run_config.json`. The sample is a one-trial H₂ pilot; its dots are not a distribution.
