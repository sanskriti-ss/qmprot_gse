# ADAPT energy and evaluation cost

This proposed appendix figure puts (a) energy relative to molecular HF and (b) cumulative objective evaluations on the same attained-operator axis for one molecule. The horizontal zero in panel (a) is the molecular HF energy. The input must have both energy and cost for the same recorded checkpoints.

Generate the included glycine preview from the repository root:

```bash
python prof_figures/additional_06_adapt_energy_and_cost/generate.py
```

For another run, provide `--input /path/to/batch_all_results.csv --molecule gly`. Optional `--label` and `--output` arguments control the provenance label and destination. PNG and PDF files are written together. This generator reads existing data only.

The sample in the `additional_04_adapt_operator_cost` folder is a historical May 2026 batch with operator targets 1–3. It predates molecule-ID fixes and should not be treated as a verified new result. The figure reads the `attained_k` field when present; historical files use `k`.

## Uncertainty

The included preview has one run per condition, so it does not support a confidence interval. With two or more independently seeded runs per condition, combine their rows in the input CSV and retain a unique `seed`, `trial_seed`, or `random_seed` column. The figure shows a two-sided 95% Student-t interval across seeds at each condition. Intervals are absent where only one seed is available; a zero-width interval can also arise from identical values. Runtime uses the geometric mean and an interval calculated on log runtime. For noisy energy, each seed must have its own clean baseline repeated consistently across its noise conditions. These intervals describe variation across the sampled seeds, not uncertainty from shots, systematic error, or a population of molecules. Very small seed counts give unstable intervals.
