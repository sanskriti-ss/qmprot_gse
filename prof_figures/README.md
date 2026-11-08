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
