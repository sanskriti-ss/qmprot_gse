# Appendix figure scripts

## Parallel amino-acid benchmark plots

For the root `run_amino19_14q_parallel.sh` benchmark, use the saved-checkpoint
plotter below. It reads results only and can plot an incomplete run without
launching missing fits.

```bash
cd /mnt/c/qmprot_2026/qmprot_gse
framework/.venv/bin/python prof_figures/generate_amino_benchmark.py \
  framework/results/amino19_14q_parallel_20261110_081445
```

It writes `prof_rerun_benchmark_*.png` and matching PDFs directly in the supplied
run root: four comparison heatmaps and a convergence sheet for every molecule.
The manifest records input hashes and missing fits. Existing figures with these
names are replaced. Missing checkpoints remain explicitly marked; saved results
may have reached their optimization budget without converging. Convergence
history indices have algorithm-specific meanings and are not equal evaluation
budgets. Runtime plots show saved fit durations under concurrent execution,
excluding interrupted attempts. These are additional rerun views, not numbered
paper figures or the five-study plots below.

The November 10 benchmark's missing true-iQCC results were rerun separately after
correcting its returned algorithm name. Its `iqcc_repair_manifest.json` records
the supplemental source/configuration and hashes of both the 171 preserved fits
and 19 added fits. Retain that manifest and the `iqcc_true_repair_20261111/`
directory alongside the original run provenance.

These folders correspond to Figures 13–17 in the appendix of `QMProt.pdf`. Each includes a generator, sample input, PNG, PDF, and instructions. The plots share the purple, pink, and orange palette used by `paper_combined_figure.py`. They read saved CSVs and do not run quantum calculations.

## Generate figures after experiments 1–5

In WSL, run the Bash entry point with the **exact output directories**, in experiment order.
It uses `framework/.venv/bin/python` automatically. To use another environment, set
`QMPROT_PYTHON` to its Python executable. Relative inputs are resolved from your current directory.

```bash
cd /mnt/c/qmprot_2026/qmprot_gse
bash prof_figures/generate_rerun.sh \
  framework/experiments/results/noise_resilience_batch/20261109_100609_nq6_hardware_efficient_vqe_batch \
  framework/experiments/results/barren_plateau_batch/20261109_114819_batch_hardware_efficient_vqe \
  framework/experiments/results/trainability/20261109_140603_all_molecules_comparison \
  framework/experiments/results/accuracy_vs_params_batch/20261109_184553_batch \
  framework/experiments/results/experiment5_term_count_sweep/20261109_154511_batch
```

Replace these run directories when selecting newer results. Add `--dry-run` to validate
input selection and list planned outputs without writing. This does not validate every plotting
column or guarantee that individual generators will succeed. Add `--label "Fresh rerun | ..."`
to record verified geometry/orbital choices; the default makes no whole-molecule claim.

PNG and PDF files named `prof_rerun_*.png` and `prof_rerun_*.pdf` are written directly
into **each experiment's output root**, alongside its CSV files. Existing files with these
names are replaced on subsequent runs. Original experiment CSVs and native plots are preserved.
Each root also receives `prof_rerun_manifest.json` with commands, input hashes and generator
diagnostics. Failed figures are reported, remaining jobs are attempted, and any failure gives
a nonzero exit status. Check the manifest before using files left from an earlier invocation.

The script generates:

- Experiment 1: Figure 17 and an energy comparison for every saved molecule ID.
- Experiment 2: Figure 14 for Bayesian initialization (or Figure 13 for COBYLA, determined
  from saved configurations), plus Figure 15. It never launches the other optimizer.
- Experiment 3: Figure 16 and energy, runtime and combined views for every saved target.
- Experiment 4: operator cost and an energy/cost view for every nonfailed molecule ID.
- Experiment 5: full and QAOA-excluded prefix-study views, each comparing the stored
  prefix energies with both full CASCI and prefix ground energies. These are additional
  views, not numbered manuscript figures. Shading is population SD across molecule means,
  not confidence intervals. The full-reference comparison is not a full-Hamiltonian rescore.

The script does not combine historical runs, infer geometry provenance, or train VQEs.
Retain the original run configurations, failure reports and inputs alongside the figures.

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
