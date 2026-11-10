# Hamiltonian prefix study

This additional view reads Experiment 5's `summary_by_algorithm_term_count.csv`.
It produces a PNG and PDF using the shared appendix styling, without importing
experiment solvers or running quantum calculations.

```bash
framework/.venv/bin/python prof_figures/additional_07_prefix_energy/generate.py \
  --input /path/to/run/summary_by_algorithm_term_count.csv \
  --output /path/to/run/prof_rerun_prefix_energy.png \
  --label "Fresh rerun"
```

Use `--exclude-qaoa` for the view without `qaoa_inspired_vqe`.
The left panel compares the stored prefix energy with full CASCI. Because these
energies belong to different Hamiltonians, it does not measure full-Hamiltonian
accuracy of the prepared state. The right panel compares the stored prefix energy
with the prefix ground energy. Shading is the population standard deviation across
molecule means, clipped at zero. It is not a confidence interval. Consult the summary's
`n_molecules`, `molecules` and `trial_seeds_by_molecule` columns for contributing cohorts;
the plot does not enforce a common molecule cohort across points.
