# Appendix figure scripts

These folders correspond to Figures 13–17 in the appendix of `QMProt.pdf`. Each includes a generator, sample input, PNG, PDF, and instructions. The plots share the purple, pink, and orange palette used by `paper_combined_figure.py`. They read saved CSVs and do not run quantum calculations.

| Paper figure | Folder | Included preview |
| --- | --- | --- |
| 13 | `figure_13_cobyla_initialization_convergence` | Historical COBYLA batch, 16 molecule IDs |
| 14 | `figure_14_bayesian_initialization_convergence` | Hydrogen pilot, one trial per strategy |
| 15 | `figure_15_initialization_final_energies` | Same hydrogen pilot, one trial per strategy |
| 16 | `figure_16_trainability_evaluation_cost` | Historical batch, target 10 only |
| 17 | `figure_17_noise_cosine_similarity` | Historical noise batch, 16 molecule IDs |

These previews show available data and are not certified reconstructions of the manuscript figures. In particular, Figures 14–16 need the full corresponding results to make the paper's multi-molecule, multi-trial or multi-target figures. Historical rows predate the molecule-ID fixes; confirm identities and provenance before interpreting them. The new scripts accept explicit input and output paths; see each folder's README.

The historical samples for Figures 13, 16, and 17 came from the local May 2026 `framework/experiments/results/` batches. Figures 14 and 15 use the hydrogen initialization result from the handoff's `study-campaign-1308d897ed02458b9aecfcbddbcbac57`. The sample CSVs are included so the previews can be regenerated without those local result folders. The included pilot configuration and older batch configuration record the original settings where available.

## Additional views

These proposed appendix views do not have figure numbers in the current manuscript. Each folder contains a generator, instructions, and PNG/PDF previews. The scripts use `additional_views.py` and `figure_style.py` and accept explicit paths to fresh results.

| Folder | View | Included sample |
| --- | --- | --- |
| `additional_01_noise_energy_comparison` | Noisy and clean-rescored energies relative to a fitted clean baseline | Historical glycine noise batch |
| `additional_02_trainability_energy` | Signed final energy relative to molecular HF | Historical trainability batch, target 10 only |
| `additional_03_trainability_runtime` | Runtime versus parameter target | Same trainability batch, target 10 only |
| `additional_04_adapt_operator_cost` | Cumulative evaluations versus attained ADAPT operators | Historical ADAPT batch, targets 1–3; four failed molecules omitted |
| `additional_05_trainability_combined` | Energy relative to HF and runtime for the same molecules and target | Historical trainability batch, target 10 only |
| `additional_06_adapt_energy_and_cost` | Paired energy relative to HF and cumulative evaluation cost versus operators | Historical glycine ADAPT trajectory, targets 1–3 |

Historical rows predate the molecule-ID corrections and are illustrative. Verify the molecule identities and other run provenance before drawing conclusions. The noise and trainability views reuse the sample CSVs in the Figure 16 and Figure 17 folders; the ADAPT views use the sample CSV in `additional_04_adapt_operator_cost`. The combined views group paired results for a compact publication layout.

For the additional views, repeated observations can be supplied in one CSV with a distinct `seed`, `trial_seed`, or `random_seed` per condition. Error bars are 95% Student-t intervals across independent seeds when at least two are present. Runtime uses a geometric mean and an interval in log space. The included previews have one run per condition and therefore show no interval; their captions explicitly say so. These intervals describe variation across runs, not finite-shot or systematic uncertainty.
