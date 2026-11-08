# Combined trainability energy and runtime

This proposed appendix view pairs (a) signed final energy relative to molecular Hartree–Fock and (b) recorded wall-clock runtime for the same molecules, algorithms, and requested parameter target. Negative energy differences indicate a result below HF. Runtime is on a logarithmic scale and depends on the computing environment.

Generate the included historical preview from the repository root:

```bash
python prof_figures/additional_05_trainability_combined/generate.py
```

To plot fresh results, supply `--input /path/to/all_molecules_results.csv --target 10`. Optionally pass `--label "run description"` and `--output /path/to/figure.png`. If a file has multiple parameter targets, `--target` is required. For repeated runs, include a `seed`, `trial_seed`, or `random_seed` column; each seed must be unique per molecule and algorithm. PNG and PDF files are written together. The script only reads saved results.

The included preview uses the historical May 2026 sample in the Figure 16 folder. It contains **target 10 only** and predates molecule-ID fixes, so check identities and run provenance before interpreting it. It is not a complete reproduction of the paper's trainability figure.

## Uncertainty

The included preview has one run per condition, so it does not support a confidence interval. With two or more independently seeded runs per condition, combine their rows in the input CSV and retain a unique `seed`, `trial_seed`, or `random_seed` column. The figure shows a two-sided 95% Student-t interval across seeds at each condition. Intervals are absent where only one seed is available; a zero-width interval can also arise from identical values. Runtime uses the geometric mean and an interval calculated on log runtime. For noisy energy, each seed must have its own clean baseline repeated consistently across its noise conditions. These intervals describe variation across the sampled seeds, not uncertainty from shots, systematic error, or a population of molecules. Very small seed counts give unstable intervals.
