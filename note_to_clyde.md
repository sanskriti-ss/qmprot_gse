# Rerunning the QMProt experiments

Hi Clyde! These instructions describe how to run fresh experiments from the `experiments` branch. The 17 figures in the manuscript reuse results from several studies; this is not yet a single-command reconstruction of the paper. Some original ordered Hamiltonians, optimizer outputs, seeds, and per-run settings are missing. In particular, Figure 12 depends on the original Hamiltonian term order. See `framework/smoke/PAPER_REPLICATION_COST.md` and `framework/smoke/REPLICATION_SCOPE.md` for the historical gaps.

The merged branch has passed syntax and static checks, but the acceptance campaign has not yet been run against this exact integration. Please check it before committing to large runs. These commands are proposed settings for new results, not a claim that they reproduce the original figures.

## Setup

Install Python 3.12 and Git, then run:

```bash
git clone -b experiments https://github.com/sanskriti-ss/qmprot_gse.git
cd qmprot_gse/framework
python3.12 -m venv .venv
source .venv/bin/activate
python -m pip install -r smoke/requirements.txt

export OMP_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
export MKL_NUM_THREADS=1
export VECLIB_MAXIMUM_THREADS=1
export MPLBACKEND=Agg
export QMPROT_GEOMETRIES="$PWD/smoke/geometries"
```

Keep the `QMPROT_GEOMETRIES` export in the same shell as the study commands. The path above assumes you are in `framework`. Without the export, the experiments use `datasets2`, where `ser`, `trp`, and `tyr` are fragments. The pinned directory contains 19 whole amino acids and hydrogen. Asparagine is excluded because its published electron count and spin are inconsistent. See [Choosing geometry inputs](framework/experiments/README.md#choosing-geometry-inputs) for how to check the saved inputs.

## First check

The command below **does run calculations**. It uses a fresh output directory and exercises five small studies and their validation. It does not generate the 17 manuscript figures. The handoff validated a different revision, so this checkout's success is not yet confirmed.

```bash
python -m smoke.study_campaign /tmp/qmprot-acceptance
```

If the directory already exists, choose a different path. Check its `campaign.log` if a study fails. Large calculations should wait until the integration is verified.

## Fresh amino-acid studies

Run from `framework` with the environment variables above. These five commands produce new outputs; they do not recreate the historical runs exactly. The noise and unreduced Qubit-ADAPT batches can take substantial time.

```bash
molecules=(ala arg asp cys gln glu gly his ile leu lys met phe pro ser thr trp tyr val)

# 1. Noise resilience
python -m experiments.noise_resilience \
  --batch-molecules "${molecules[@]}" \
  --algorithm hardware_efficient_vqe \
  --cs-target-qubits 6 --n-layers 2 --rotation-gates RY \
  --max-iter 150 --seed 42 \
  --noise-models depolarizing phaseflip amplitude_damping \
  --noise-strengths 0 0.001 0.005 0.01 0.02 0.05

# 2. Initialization and gradient variance
python -m experiments.barren_plateau_batch \
  --molecules "${molecules[@]}" \
  --cs-target-qubits 6 --layers 1 2 4 6 8 10 \
  --n-samples 30 --opt-layers 6 --n-trials 5 \
  --optimizer BAYESIAN --max-iter 200 --seed 42

# 3. Trainability and cost to accuracy
python -m experiments.trainability \
  --molecules "${molecules[@]}" --cs-target-qubits 0 \
  --algorithms qubit_adapt_vqe hardware_efficient_vqe \
  --param-targets 10 50 100 --eval-budget 2000 \
  --max-iter 500 --seed 42

# 4. ADAPT accuracy versus ansatz size
python -m experiments.accuracy_vs_params \
  --molecules "${molecules[@]}" --cs-target-qubits 0 \
  --algorithm qubit_adapt_vqe --qubit-adapt-pool restricted \
  --k-list 1 2 3 --inner-max-iter 200 --seed 42

# 5. Hamiltonian prefix sweep; defaults include all registered algorithms
python -m experiments.experiment5_term_count_sweep \
  --molecules "${molecules[@]}" --cs-target-qubits 4 \
  --term-counts 1 2 3 5 10 --max-iter 200 --seed 42
```

Each runner writes dated output under `framework/experiments/results/`. Preserve the input snapshots, source archive, configuration, runtime metadata, and results together. For a completed trainability output, the intended replay command is:

```bash
python -m experiments.replay_trainability /path/to/trainability-output
```

Replay has not yet been checked on this merged revision. For a prefix rerun on the same ordered Hamiltonians, use `--prepared-inputs /path/to/prior/run/inputs`; changing the ordered terms changes the prefixes.

## Figures

The appendix figure scripts are organized under `prof_figures/` by paper figure number. They read existing results; they do not train VQEs. Supply the input files or directories documented by each script. Some original paper plotting scripts use hard-coded historical paths, so a new result may need a plotting script or explicit input path adapted before it reproduces the paper layout. The new plots should be labeled as reruns, not the original historical results.

The original work used both canonical and natural orbitals in different investigations. This checkout retains the natural-orbital fallback. Record the chosen orbital basis, contextual reduction, seed, pool, and molecule roster when comparing runs.
