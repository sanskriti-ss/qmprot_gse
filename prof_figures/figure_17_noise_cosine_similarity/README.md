# Figure 17: Noise cosine similarity

Generate a PNG and PDF from the included sample:

```bash
python prof_figures/figure_17_noise_cosine_similarity/generate.py
```

To use fresh results, pass `--input PATH` and optionally `--label "description of the run"` and `--output path/to/figure.png`. The script only reads saved results; it does not run a VQE.

Included sample: **Historical May 2026 data | input IDs require independent verification**. It is a plotting preview and is not a reproduction of the original paper figure. Inspect the run configuration and original data before drawing scientific conclusions.

Input: `batch_all_results.csv`. Rows must be unique per molecule, noise channel and strength. The historical sample predates the glutamine/glutamic-acid ID fix, so validate molecule identities before interpreting it.

The sample is from the May 9, 2026 `noise_resilience_batch/20260509_172216_nq6_hardware_efficient_vqe_batch` output.
