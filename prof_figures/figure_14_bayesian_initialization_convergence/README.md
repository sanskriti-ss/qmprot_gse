# Figure 14: Bayesian initialization convergence

Generate a PNG and PDF from the included sample:

```bash
python prof_figures/figure_14_bayesian_initialization_convergence/generate.py
```

To use fresh results, pass `--input PATH` and optionally `--label "description of the run"` and `--output path/to/figure.png`. The script only reads saved results; it does not run a VQE.

Included sample: **Hydrogen pilot only | Bayesian | one trial per strategy**. It is a plotting preview and is not a reproduction of the original paper figure. Inspect the run configuration and original data before drawing scientific conclusions.

Input: a CSV with molecule, strategy, trial, iteration and energy, a batch output with `convergence_traces_all.csv`, or per-molecule folders with `convergence_traces.csv` and `run_config.json`. Each figure must use one depth and one run per molecule.
