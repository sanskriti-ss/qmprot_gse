# Exact efficiency of the QMProt benchmark

The [deep-circuit and study efficiency report](DEEP_EFFICIENCY.md) measures the subsequent pass against the optimized revision below, including the full 10/50/100 grid on Modal and the fitted Bayesian optimizer.

The earlier 1.03× estimate covered one improvement in two full-budget glycine fits. Four avoided sparse-matrix builds saved about 2.9 seconds out of 111. The measurements below compare the complete set of optimizations against revision `ab1954f6c33f7c9640ada79b1722107a9e573ff6`. The single-threaded candidate scientific source archive has SHA-256 `f218d0b1820c65c8c0a7905ad55540391874a4abc56a1bc9d64006c102aba0ad`.

## Measured runs

| Native local workload | Baseline | Candidate | Wall speedup |
|---|---:|---:|---:|
| Glycine trainability, 12 qubits, 1,819 terms, HE and Qubit-ADAPT, 10/50/100 requested angles, evaluation cap 2,000 | 481.08 s | 144.18 s | 3.34× |
| Glycine prefix sweep, eight qubits, 11 solvers, 10/50/200 terms, two seeds, 66 fits | 53.19 s | 29.10 s | 1.83× |
| H₂ + 12-qubit glycine adaptive size study, Qubit-ADAPT, one addition, both molecules with all figures | 90.07 s | 13.17 s | 6.84× |
| Glycine trainability, 12 qubits, HE + Qubit-ADAPT + approximate ADAPT, 10 requested angles, cap 2,000 | 247.29 s | 25.13 s | 9.84× |
| H₂ noise sweep, HE + COBYLA, three channels, strengths 0 and 0.01 | 3.19 s | 2.14 s | 1.49× |
| Glycine initialization, gradients and gradient descent | 7.74 s | 6.26 s | 1.24× |

Times include process startup, the workload's configured chemistry or frozen input, fitting and writing outputs. The three-target, all-method and adaptive-size batches include their plots. The noise and initialization timings isolate the fixed-circuit changes, use two counterbalanced repetitions per version and omit plots. Runs use one native compute thread on macOS arm64. The three-target timing compares one complete run per source. Four small H₂ adaptive-size workloads with both requested optimizers gave 0.97–1.00× across counterbalanced repetitions. The input preparation, batch output and figures can dominate small fits. [Paired native results](reproductions/study_checks/exact_efficiency.json) · [Fixed-method measurements](reproductions/study_checks/fixed_energy_speed.json) · [Adaptive measurements](reproductions/study_checks/adaptive_energy_sweep.json)

## Where the time went

| Work removed | Measurement and scope |
|---|---|
| Rebuilding or serializing the Hamiltonian for each scalar energy | Fixed analytic circuits retain the original device decomposition and ordered Lightning observable. Matching parameter vectors can reuse energies across callbacks. At 10 angles the isolated complete glycine HE fit improved 1.57×. Analytic density-matrix groups can share their circuit state and still execute their own readout rotations. |
| Measuring Pauli words with zero overlap | The adaptive measurement checks every nonzero amplitude. A Pauli word can contribute only if its flipped bits connect two occupied basis states. The verified single-thread kernel measures the retained words in original order. In the initial unreduced glycine screen, typical probes had three occupied basis states and only 79–101 of 1,819 words could contribute. Measurement alone improved about 18×. The standalone 2,000-cap Qubit-ADAPT fit improved 8.83×. Beyond 16 occupied basis states, the full observable is measured. Tested deeper states reached 4,043 of 4,096. |
| Repeating the first ADAPT screen at different target sizes | Two later evaluations of the 972-generator glycine screen each reused 1,944 displaced energies. Each logical request, including budget exhaustion, is still recorded. Warm target-50 and target-100 Qubit-ADAPT fits took 0.46 and 0.13 s. An independent target invocation pays for its own first screen. |
| Constructing provably zero iQCC commutators | At the full glycine first step 570 of 630 commutators could be omitted. True iQCC fell from 6.59 to 4.79 s (1.38×), inspired iQCC from 52.17 to 6.51 s (8.01×), isolating screening from the shared HF change. A native two-method eight-qubit prefix batch through two steps improved 2.23×. The original selection rules, dressing and objective calls remain visible in the paired records. |
| Building a dense Hamiltonian for a determinant energy | For 12 qubits and 40 terms the reference check fell from 2.71 s to 27 µs. When the full dense matrix was explicitly needed, row-wise Pauli construction fell from 2.70 s to 12.9 ms. The complete 10-angle approximate ADAPT fit on 1,819 terms fell from 135.63 to 1.44 s, with the documented approximate gate still in use. |
| Rebuilding prefix operators and contextual rotations for physical diagnostics | Sharing ordered prefix snapshots cut reference-plus-endpoint matrix work from 79.4 to 52.4 ms on 12 qubits. An eight-qubit state lifted to 12 spin orbitals for charge and spin took 16.9 ms before and 1.33 ms after. CSR indices, row pointers, values and physical outputs matched. |

In the 66-fit sweep, solver-only speedups ranged from 1.01× for CB to 3.54× for inspired iQCC. CB still pays for constructing its ansatz unitary and evaluating the boost. The two deepest HE fits took 114.55 of the remaining 144.18 seconds. In 64 warmed, distinct 100-angle calls, 1.300 s went into the native ordered Hamiltonian expectation, 0.455 s into circuit construction and transforms, and 0.029 s into state evolution. Each instrumented energy matched the registered solver. This makes native expectation and fixed-circuit preparation the largest measured targets for that workload. A Hamiltonian contraction with a different accumulation order needs a separate precision study. One proposed HVA gate expansion differed by up to 6.2 × 10⁻¹⁰ Ha and was excluded. [iQCC screening evidence](reproductions/study_checks/iqcc_screen_speed.json) · [Profile driver](reproductions/study_checks/deep_energy_profile.py)

## Modal timings

Both source trees ran sequentially on the same two-core, 4 GiB worker with one native compute thread. Each timing includes process startup, input preparation or loading, optimization and native study output. The H₂ matrix also includes its independent dense replays. Container startup, transfer and queue time sit outside these timers.

| Workload on the same worker | Baseline | Candidate | Wall speedup |
|---|---:|---:|---:|
| Unreduced glycine, HE and Qubit-ADAPT, target 10, cap 2,000 per fit | 359.39 s | 87.76 s | 4.10× |
| Full H₂ prefix, all eleven solvers, two seeds, 22 fits | 11.45 s | 8.83 s | 1.30× |
| H₂ initialization and noise matrix, 42 configurations, 78 initialization fits and 48 independent noise fits | 176.78 s | 107.28 s | 1.65× |

On this worker the Qubit-ADAPT fit itself fell from 288.09 to 47.03 s (6.13×), and the HE fit from 49.83 to 21.67 s (2.30×). The full 2,554-event energy/parameter stream and physical endpoints matched their same-worker baseline. All 22 prefix cases and all 227 scientific files in the H₂ matrix also matched. Optimizer trajectories can differ between machines, even with fixed starting seeds. [Native receipts and digests](reproductions/study_checks/exact_efficiency.json)

## Correctness and extrapolation

The local six-fit trial preserved all 13,046 objective and callback records byte for byte, including energy and parameter bits. Both evaluation traces, endpoints, charge distributions, CASCI references, and aggregate scientific tables also matched. The 66 prefix results matched in angles, energies, histories, stopping decisions, existing work counters and sector weights. The H₂ initialization/noise matrix preserved all 227 scientific files, including the fitted angles, gradients, noise endpoints and convergence traces. Timings and newly exposed counters are reported separately. Independent physical replay uses its stated tolerance for a different calculation. [Verifier](efficiency.py) · [Hydrogen matrix verifier](solver_efficiency.py)

The full five-study acceptance covers both requested optimizers where supported, all registered algorithms and both local and cloud execution. The separate 22-endpoint independent physics audit flags the registered approximate ADAPT double at zero angle on H₂. Wrong-electron-number endpoints stay in the benchmark. Vacuum metadata and open-shell preparation remain outside the checked molecular cohort. The underlying HF lookup treats zero electrons in unreduced molecule metadata as half filling. [Scientific scope](SCIENTIFIC_READINESS.md)

An all-molecule production speedup requires the original runtime shares by solver, active space, optimizer and study. For baseline wall-time shares `w_i` and measured speedups `s_i`, the serial estimate is `1 / sum(w_i / s_i)`. Include preparation and output as workloads, then account for the slowest worker in a parallel schedule. At the measured sizes a prefix-heavy campaign and a one-addition adaptive campaign have substantially different costs. More terms can also change the optimizer trajectory and the number of accepted generators. These measurements do not establish an all-molecule duration.

## Reproducing the comparisons

Start in the repository root with the pinned environment from the [smoke setup](README.md). This restores both the scientific framework and its top-level initialization driver for the pre-optimization baseline.

```sh
baseline_dir=$(mktemp -d)
git archive ab1954f framework run_barren_plateau_part_b_all_molecules.py | tar -x -C "$baseline_dir"
cd framework
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 VECLIB_MAXIMUM_THREADS=1
export QMPROT_BASELINE_FRAMEWORK="$baseline_dir/framework"
.venv/bin/python -m modal run -m smoke.modal_efficiency
```

The Modal command runs both source trees sequentially on one CPU worker and downloads both native result sets. The equivalent local production-size pair below uses the frozen eight-qubit input in the handoff. Run it from `framework` with the baseline and thread variables set above. Each version runs in its own interpreter with the same fixture root, seed, budgets and plotting configuration.

```sh
candidate=$PWD
python_bin="$candidate/.venv/bin/python"
paired=$(mktemp -d)
frozen="$candidate/results/profiling/semantic-acceptance/verified-gly-roster-A/inputs"
export MPLBACKEND=Agg QMPROT_GEOMETRIES="$candidate/smoke/geometries" QMPROT_WORKLOAD=full-three-targets
(
  cd "$paired" || exit 1
  set -e
  for version in baseline candidate; do
    source_root=$candidate
    if [ "$version" = baseline ]; then source_root=$QMPROT_BASELINE_FRAMEWORK; fi
    export PYTHONPATH="$source_root"
    /usr/bin/time -p "$python_bin" "$candidate/smoke/record.py" experiments.trainability \
      "$paired/$version-events.jsonl" --molecules gly --cs-target-qubits 0 \
      --algorithms qubit_adapt_vqe hardware_efficient_vqe --param-targets 10 50 100 \
      --eval-budget 2000 --max-iter 500 --seed 42 --output-root "$paired/$version-trainability" \
      > "$paired/$version-trainability.log" 2>&1
    /usr/bin/time -p "$python_bin" -m experiments.experiment5_term_count_sweep \
      --molecules gly --cs-target-qubits 8 --prepared-inputs "$frozen" \
      --term-counts 10 50 200 --seeds 42 43 --n-layers 2 --max-operators 2 --max-iter 32 \
      --output-dir "$paired/$version-prefix" > "$paired/$version-prefix.log" 2>&1
  done
)
"$python_bin" -m smoke.efficiency "$paired"
```

For stored pairs, the last command repeats the verification without rerunning the fits. For the adaptive-size comparison the raw native directories and counters are retained alongside the runner. The archived runtime configurations, ordered Hamiltonians and native commands specify the exact workloads. [Shared component driver](reproductions/study_checks/shared_energy_sweep.py) · [Adaptive-size measurements](reproductions/study_checks/accuracy_size_efficiency.json)
