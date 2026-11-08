# Deep-circuit and study efficiency

This pass starts at `116b39e`, after the [first exact-efficiency sweep](EXACT_EFFICIENCY.md). It covers the four VQE studies (trainability, adaptive accuracy versus size, initialization and noise) and the eleven-solver Hamiltonian-prefix batch. Glycine has 12 qubits and 1,819 terms before contextual reduction. The frozen eight-qubit glycine fixture has 361 terms.

## Complete local runs

| Native workload | Previous optimized revision | This pass | Speedup |
|---|---:|---:|---:|
| Unreduced glycine trainability, six HE/Qubit-ADAPT fits at targets 10/50/100 and cap 2,000 | 139.01 s | 46.43 s | **2.99×** |
| Contextual glycine prefix batch, 11 solvers, 66 fits, through 200 terms | 26.45 s | 21.61 s | 1.22× |
| Unreduced glycine initialization, gradients and gradient descent | 13.87 s | 6.13 s | 2.26× |
| Full H₂ production noise sweep, 16 independent clean/positive-strength fits | 12.99 s | 12.88 s | 1.01× |
| H₂ and full glycine Qubit-ADAPT accuracy versus size, one accepted addition per molecule | 13.02 s | 13.29 s | 0.98× |

Trainability uses medians across two comparisons run in opposite order. Baseline times were 137.81–140.20 s and candidate times 43.14–49.71 s. Prefix and noise times are medians of two sequential runs per revision in counterbalanced order. A separate noise pair timed during concurrent checks measured 14.15 versus 16.52 s. Initialization and adaptive size each use one pair. All these workloads wrote native study artifacts and figures. All 13,046 training events also match the 481.08 s baseline from before either sweep, after omitting only the timing harness's workload label. The same native input, endpoints and both trace files match byte for byte across both rounds. This gives a cumulative local improvement of about 10.4× on that six-fit workload. [First-pass evidence](reproductions/study_checks/exact_efficiency.json) · [Adaptive-size check](reproductions/study_checks/deep_accuracy_size.json)

## Same-worker Modal runs

| Native workload | Previous optimized revision | This pass | Speedup |
|---|---:|---:|---:|
| Full-budget glycine trainability, same six configurations as the local run | 363.28 s | 125.09 s | **2.90×** |
| H₂ prefix sweep, all 11 solvers, two seeds, full 15-term operator | 7.39 s | 6.93 s | 1.07× |
| H₂ initialization and noise, 42 method/optimizer configurations with independent replay | 94.46 s | 88.28 s | 1.07× |
| Full H₂ production noise sweep, 16 fits | 56.23 s | 54.49 s | 1.03× |
| Frozen H₂ and eight-qubit glycine noisy objectives, three channels, six inputs each, process startup included | 87.83 s | 72.43 s | 1.21× |

Each pair used a single Modal CPU worker with one native compute thread. All 12,943 trainability events, 22 prefix case records, 227 scientific files in the initialization/noise matrix, 2,505 full-grid noise events and 36 individual noisy energies matched within the worker. HE fits at 10, 50 and 100 requested angles took 18.63 → 2.76 s, 124.41 → 22.92 s and 148.01 → 28.74 s. Qubit-ADAPT still spent about 40 s on its first screen. Each of the three warmed eight-qubit noisy readouts gained 1.23× on Modal and 1.53–1.60× locally. This scalar timing covers individual energies. Full eight-qubit noisy optimization time depends on how much time the solver spends in those evaluations. [Same-host pairs, native hashes and local repeats](reproductions/study_checks/deep_efficiency.json)

## Which computations changed

**Ordered Hamiltonian expectation.** For an exactly real state, every Pauli word either sends an amplitude to one other basis coordinate with a real sign, or sends it to the imaginary component. The latter contributes zero to the real expectation. The implementation groups 16 adjacent basis rows that share the same coefficient pattern. Within each lane it retains every contributing term in the native order, including duplicate destination columns. It also retains the native order of the final inner product. On 12-qubit glycine, the measured expectation fell from 20.15 to 4.24 ms locally (4.75×) and from 24.68 to 3.40 ms on Modal (7.26×). Two different pinned sparse kernels produce the required separate rounding of the multiply and add. The fast path checks both binary fingerprints, the actual native thread count, coefficient bounds, term order and state. The first 12-qubit measurement took 15 ms locally and 31 ms on Modal. A single cached plan used 10.37 MiB locally and 6.82 MiB on Modal, with 5.94 and 2.97 MiB of temporary amplitude blocks respectively. Complex states and unverified builds use the native measurement. [Arithmetic controls, versions and measurements](reproductions/study_checks/ordered_energy_efficiency.json)

**Fixed circuit construction.** For the registered HE, layered PennyLane and CB circuits, the implementation verifies that each free angle occupies one original RX, RY or RZ slot on a stationary native tape. Subsequent calls bind their angles directly. A changed preparation or entangling gate triggers reconstruction. The public QNode still exposes the evaluated tape. Sampled execution, active tracking and unsupported transforms use the full path. With 100 angles the setup alone fell from 6.14 to 0.26 ms. Isolating setup reduced the complete 2,000-call HE fit from 61.07 to 49.30 s. With the ordered expectation as well, one warmed 100-angle objective fell from 26.37 to 6.39 ms locally. The parameter-shift initialization study also uses this forward evaluation and retains both calls per gradient component. [Tape, state and optimizer parity](reproductions/study_checks/rotation_efficiency.json)

**Classically boosted VQE.** The eight-qubit contextual preparation compiles to 254 CNOTs and 59 RY gates, even when its state has only one occupied amplitude. Caching its actual preparation matrix preserves those gates' rounding. The cache retains at most two small matrices and verifies the full reference-state bytes. When the classical reference occupies one determinant, the cross term uses exactly one ordered Hamiltonian row. The 12-qubit, 200-term row used 64 KiB versus 256 MiB for the full matrix. The native ansatz unitary, measured Hadamard overlaps and Ritz solve still run. Six plotted contextual glycine CB fits went from 8.60 to 4.07 s locally (2.12×) and 18.35 to 10.98 s on Modal (1.67×), with identical unitary matrices and fitted results. [CB measurements and binary checks](reproductions/study_checks/boost_efficiency.json)

**Analytic noisy readout.** The 361-term glycine measurement generated 290 observable groups and 2,016 basis-rotation gates per noisy energy. Groups starting with the same native rotations on the same prepared density matrix can reuse intermediate states. A bounded stack holds up to 32 MiB, the result array retains the original group order, and the last group remains last so the device exposes the same terminal state. Small batches, finite shots, readout error, tracking and distinct preparations keep the existing execution path. The H₂ production circuit has eight groups and uses that small-batch path. Tests compare each group's scalar and both device density matrices against PennyLane. [Readout regression](test_mixed_readouts.py)

**Bayesian initialization.** On frozen four-qubit glycine, a 16-layer, 68-angle trial made 200 distinct proposals. The native circuit energies took a combined 0.108 s. The optimizer spent 17.79 s predicting means and uncertainties, 8.39 s fitting trees and 8.40 s drawing and transforming candidate grids. The pinned serial ExtraTrees estimator now descends each fitted tree once, then reads the mean and variance from that leaf in the native accumulation order. This reduced prediction to 4.15 s (4.28×). The normal optimizer handles acquisition, refitting and termination. Passing the estimator preserves the exact original random draw for its seed. [Estimator and guards](../core/forest.py) · [Profile and paired timings](reproductions/study_checks/forest_efficiency.json)

The extra change at `44456b7` was timed against `4d94bde`. A complete molecular initialization trial took **34.83 → 21.87 s locally (1.59×)** across two pairs in opposite order and **121.58 → 80.19 s on Modal (1.52×)** in one pair. The timer includes optimization, endpoint scoring and recording the comparison on the archived input, and excludes new chemistry, process start and plots. Hashing the candidate grids, means and uncertainties took about 0.30 s per local trial. All 200 angle/energy pairs, 168 grids of 10,000 candidate vectors, their predicted means and standard deviations, and the selected endpoint matched byte for byte on each host. [Paired Bayesian trial verifier](forest_trial.py)

Across hosts, the baseline objective at proposal two differed by 5.68 × 10⁻¹⁴ Ha at identical angles. Although both hosts sampled identical candidate grids throughout, the first different selected angles appeared at proposal 37. Either revision finished at −278.74166030942996 Ha locally and −278.81864914333795 Ha on Modal. The archived four-qubit input lacks the parent transform needed to recover electron-count probabilities. Its endpoint expressly records that limitation. Other inputs with the parent transform exercise electron-sector reporting in the canaries. [Bayesian comparison records](reproductions/study_checks/forest_efficiency.json)

## Scope of the measurements

Each timing pair uses the same host, seed, chemistry settings and ordered Hamiltonian. The trainability command covers both HE and Qubit-ADAPT at 10, 50 and 100 requested angles with an independent cap of 2,000 logical evaluations per fit. It includes preparation, callbacks, physical endpoints, aggregate CSVs and figures. The local prefix batch includes every registered solver at 10, 50 and 200 terms across two seeds. The full H₂ noise campaign uses the checked-in 150-iteration COBYLA configuration, all three channels and all five positive strengths. It retains the zero-strength anchors as well as the clean fit. The local initialization timing covers depths one and three, two initialization strategies, five gradient probes and a two-step gradient-descent fit. Timing values depend on the workload, CPU, native wheels and batch order.

For the previously discussed schedule of 360 Bayesian trials, 360 trials with this exact Modal fit duration would use about 8.0 worker-hours of fitting, down from 12.2. Actual chemistry preparation, molecule-dependent fit costs, batch I/O, plotting and worker concurrency determine the full campaign time. [Trial configuration and measured costs](reproductions/study_checks/forest_efficiency.json)

The independent physical audit still flags the approximate ADAPT double at zero angle on H₂. States found outside the target electron sector remain in the results. The full-cohort speedup depends on the mix of solvers, active spaces, optimizers and depths. CB still constructs the full ansatz unitary. Complex-state Hamiltonian measurement and general density-matrix evolution still scale with problem size. [Scientific scope and physical checks](SCIENTIFIC_READINESS.md)

## Repeating the comparison

From the repository root, use the pinned Python environment and the existing Modal credentials described in the [smoke setup](README.md).

```sh
baseline_dir=$(mktemp -d)
git archive 116b39e framework run_barren_plateau_part_b_all_molecules.py | tar -x -C "$baseline_dir"
cd framework
export QMPROT_BASELINE_FRAMEWORK="$baseline_dir/framework"
.venv/bin/python -m modal run -m smoke.modal_efficiency --deep
```

The worker runs the checkout against the extracted baseline and retains native outputs, source snapshots and binary optimizer traces. The published first pair used candidate `4d94bde`. Its `comparison.json`, `matrix-comparison.json` and `deep-comparison.json` cover trainability, prefixes, the initialization/noise matrix, the full noise sweep and frozen noisy readouts. For a local trainability/prefix pair, follow the [local shell example](EXACT_EFFICIENCY.md) with baseline `116b39e`. Use `python -m smoke.deep_studies FRESH_DIRECTORY` for the frozen H₂ and eight-qubit glycine channel timing.

To reproduce the Bayesian trial with both revisions on one Modal worker, run from the final checkout at the repository root with the archived input present.

```sh
baseline_dir=$(mktemp -d)
git archive 4d94bde framework run_barren_plateau_part_b_all_molecules.py | tar -x -C "$baseline_dir"
cd framework
export QMPROT_BASELINE_FRAMEWORK="$baseline_dir/framework"
export QMPROT_FOREST_INPUT="$PWD/results/profiling/scientific-readiness/acceptance/gly-bayesian-depth16/inputs/gly.npz"
.venv/bin/python -m modal run -m smoke.modal_forest_trial
.venv/bin/python -m smoke.forest_trial results/smoke/forest-trial-03a0750a04ba4301bd02aa9a41112b0f
```

Use the run ID printed by Modal to check a fresh download. The last command above rechecks the archived pair. Each forest-trial Modal directory also contains the frozen input as `gly.npz`.

## Handoff and verification

The incremental handoff is `framework/results/profiling/scientific-readiness/qmprot-deep-efficiency-handoff.zip`. It contains the Git bundle after `116b39e`, the measured input and scientific source archives, native outputs, optimizer traces and their verifier. The preceding `qmprot-readiness-handoff.zip` supplies that base revision and its earlier evidence. To verify the increment from a checkout containing `116b39e`, run the following from the repository root.

```sh
handoff=framework/results/profiling/scientific-readiness/qmprot-deep-efficiency-handoff.zip
verify_dir=$(mktemp -d)
unzip -p "$handoff" handoff/verify.py > "$verify_dir/verify.py"
framework/.venv/bin/python "$verify_dir/verify.py" verify "$handoff" --python framework/.venv/bin/python
```

The final code reran the full local 2,000-budget glycine grid with all 13,046 trace events preserved. Each host validated 55 scientific snapshots at fingerprint `6f9800b7`. The final regression suite and 22 physical endpoints are recorded separately from the timed first pair. [Final acceptance, physics and optimizer records](reproductions/study_checks/deep_acceptance.json)
