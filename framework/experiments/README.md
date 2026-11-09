# `framework/experiments/`

Reproducible, self-contained studies built on top of the existing framework
modules:

- `core/` -- `BaseVQE`, `BackendConfig`, `QubitHamiltonian`, the device factory
  and the per-layer noise-inserter callback,
- `algorithms/` -- the VQE ansatze (we use `hardware_efficient_vqe`,
  `pennylane_vqe`, `qubit_adapt_vqe` and `adapt_vqe`),
- `active_space_truncation/` -- HF -> MP2 -> CASCI -> qubit Hamiltonian
  pipeline driven directly from `datasets2/<molecule>/<molecule>.h5`,
- `contextual_subspace/` -- optional CS reduction to drive the qubit count
  down to ~4 for fast experiments (used by experiments 1 and 2; **not**
  used by experiments 3 and 4 -- see those sections for the reason).

Every experiment writes its outputs to a date-and-experiment-stamped
folder of the form
`results/<experiment_name>/<YYYYmmdd_HHMMSS>_<molecule>_<algo>/` so runs
are trivially diff-able.

Trainability CSVs include `hf_energy` (molecular HF) and `casci_energy`
(active-space CASCI), including batch aggregates and resumed results. These
columns let the appendix energy/runtime plots consume fresh results directly.
The Part B HF chart uses the explicit column names `molecular_hf_energy` and
`active_space_casci_energy`; it requires a complete successful molecule cohort.
The accuracy poster labels the contributing cohort at each attained operator
count and only normalizes correlation recovery where HF minus CASCI exceeds
1.6 mHa. Recovery values outside 0–100% remain visible.

The cheap end (4 qubits, water/glycine/hydrogen) runs on a laptop in a
couple of minutes per experiment; experiments 3 and 4 default to the
12-qubit active-space water Hamiltonian and take a few minutes longer
because adaptive ansatze need a real correlation gap to grow into.

## Layout

```
experiments/
├── __init__.py
├── _common.py                  # shared helpers (loading, init, gradient, counter, calibrator, run-dir)
├── noise_resilience.py         # Experiment 1
├── barren_plateau.py           # Experiment 2
├── trainability.py             # Experiment 3
├── accuracy_vs_params.py       # Experiment 4
├── experiment5_term_count_sweep.py # Experiment 5
└── results/
    ├── noise_resilience/<timestamp>_<molecule>_<algo>/
    ├── barren_plateau/<timestamp>_<molecule>_<algo>/
    ├── trainability/<timestamp>_<molecule>_comparison/
    ├── accuracy_vs_params/<timestamp>_<molecule>_<algo>/
    └── experiment5_term_count_sweep/<timestamp>_batch/
```

`_common.py` provides:

- `load_small_hamiltonian(molecule, basis, cs_target_qubits)` -- runs the
  active-space pipeline on `datasets2/<molecule>/<molecule>.h5` and (if
  needed) folds in a CS reduction down to a target qubit count.
- `make_backend_config(...)`, `build_vqe(...)` -- convenience constructors
  that bypass `main.py` so we can re-use one initial parameter vector
  across many runs.
- `sample_initial_parameters(n, strategy, rng)` -- the four init strategies
  (`random_uniform`, `small_random`, `near_identity`, `zeros`).
- `parameter_shift_gradient(cost_fn, params, indices)` -- exact analytic
  gradient via the parameter-shift rule (`shift = pi/2`) for any ansatz
  built from single-rotation gates.
- `CostEvalCounter` -- attaches to a VQE instance and counts every
  underlying cost-function call, regardless of which method made it.
  Used by experiments 3 and 4 to compare algorithms with very different
  notions of "iteration".
- `he_layers_for_target_params(...)` -- inverts the
  `n_qubits * rpq * (n_layers + 1)` formula used by
  `HardwareEfficientVQE` so we can pick a depth that matches an
  algorithm-agnostic parameter target.
- `make_run_dir(experiment, molecule, algorithm)` -- creates the
  date-stamped output directory used by every experiment.

## Experiment 1 -- Noise resilience

**Question.** If you optimise the *same* ansatz on the *same* Hamiltonian
from the *same* initial parameter vector, how much do the optimum and the
recovered energy drift when each layer is followed by a fixed amount of
single-qubit noise?

**Method.** A noiseless baseline plus a sweep over
`noise_models x noise_strengths` using PennyLane's gate-level channels
(`DepolarizingChannel`, `PhaseFlip`, `AmplitudeDamping`, ...).  All runs
share the same theta_0.  We record:

- `E_noisy` -- what COBYLA converged to on the noisy backend,
- `E_clean_at_noisy_params` -- the noisy theta\* re-evaluated on the
  noiseless statevector simulator (the "true" performance of the noisy
  result),
- `param_drift_l2 = ||theta_noisy* - theta_clean*||_2`,
- `param_drift_cosine` between the two parameter vectors,
- the full convergence trace.

**Run it.**

```bash
cd framework
python -m experiments.noise_resilience \
    --molecule water --algorithm hardware_efficient_vqe \
    --n-layers 2 --max-iter 80 --cs-target-qubits 4
```

Defaults: water, hardware_efficient VQE (n_layers=2), CS-reduced to 4
qubits, sto-3g basis, COBYLA up to 80 iterations,
noise_models = `{depolarizing, phaseflip, amplitude_damping}`,
strengths = `{0, 0.001, 0.005, 0.01, 0.02, 0.05}`.

Useful flags: `--molecule gly|hydrogen|...`, `--n-layers`, `--init-strategy`,
`--noise-models`, `--noise-strengths`, `--no-plots`.

**Outputs** (in `results/noise_resilience/<timestamp>_<mol>_<algo>/`):

- `run_config.json`
- `results.csv`, `convergence_traces.csv`
- `energy_vs_noise.png` -- both `E_noisy` and `E_clean_at_noisy_params`
  versus `p`, plus baseline / CASCI lines.
- `param_drift_vs_noise.png` -- L2 drift and cosine similarity panels.
- `convergence.png` -- one subplot per noise model, all `p` values overlaid.

## Experiment 2 -- Barren plateaus and parameter initialisation

**Question.** How does the initialisation distribution of the variational
parameters affect (A) the *gradient variance at start* (the McClean et al.
2018 barren-plateau diagnostic) and (B) the *convergence quality* of the
subsequent optimisation?

**Hypothesis.** Sampling from a wide distribution like `[0, 2*pi]` lands in
the exponentially flat region of the cost landscape (a barren plateau);
sampling from a narrow distribution near zero leaves the state close to
Hartree-Fock, where the gradient is large and the optimiser converges to
a substantially better solution.

**Method.**

- **Part A (gradient variance vs depth).** For each ansatz depth in
  `--layers` and each strategy in `--bp-strategies`, draw `--n-samples`
  independent parameter vectors and compute one component of the
  parameter-shift gradient at each of them.  Report `Var[g_k]` and
  `E[|g_k|]`.
- **Part B (optimisation comparison).** Hold `n_layers = --opt-layers` and
  run `--n-trials` independent optimisations from each strategy in
  `--opt-strategies`.  Plot mean +- std convergence curves and final-energy
  boxplots.

**Run it.**

```bash
cd framework
python -m experiments.barren_plateau \
    --molecule water --algorithm hardware_efficient_vqe \
    --layers 1 2 4 6 --n-samples 20 \
    --opt-layers 4 --n-trials 5 --max-iter 100
```

Useful flags: `--skip-part-a` / `--skip-part-b` to save time when iterating,
`--gradient-index` to differentiate a different parameter, `--cs-target-qubits 0`
to skip CS, `--basis cc-pvdz` for a richer Hamiltonian.

**Outputs** (in `results/barren_plateau/<timestamp>_<mol>_<algo>/`):

- `run_config.json`
- `gradient_variance.csv`, `optimization_results.csv`,
  `convergence_traces.csv`
- `gradient_variance_vs_layers.png` -- log-scale `Var[g_k]` and `E[|g_k|]`
  vs depth, one curve per init strategy.
- `convergence_by_init.png` -- mean +- std VQE convergence per strategy,
  with HF / CASCI reference lines.
- `final_energy_by_init.png` -- per-strategy final-energy boxplot.

## Experiment 3 -- Trainability at matched parameter counts

**Question.** For a fixed VQE problem, how does the trainability of an
*adaptive* ansatz (`qubit_adapt_vqe`) compare to that of a *fixed*
hardware-efficient ansatz when both use the same number of variational
parameters?

**Method.** Pre-choose target parameter counts (default `10, 50, 100`)
and run, for each target:

- **Qubit-ADAPT-VQE** with `max_operators = target_n_params`.  Grows the
  ansatz one Pauli rotation at a time, picking the operator with the
  largest energy gradient from the qubit pool.
- **HardwareEfficientVQE** at the smallest `n_layers` that yields
  `>= target_n_params` parameters.  Defaults to RY-only rotations so the
  parameter count grows linearly with depth.

**Why qubit-ADAPT instead of `adapt_vqe`?**  The simpler `adapt_vqe`
class uses RY+CNOT gate approximations whose finite-difference gradient
at theta=0 is zero by parameterisation symmetry on these small
Hamiltonians, so it exits at k=0.  `qubit_adapt_vqe` uses proper
`PauliRot(2*theta, ...)` operators, which behave correctly and have the
extra benefit of pre-screening the pool against the Hamiltonian's
off-diagonal flip patterns.  Both algorithms are still selectable via
`--algorithms`.

**Why not CS-reduce?**  The contextual-subspace reduction places the
start state at the noncontextual ground state, which is by construction
a stationary point of the operator pool -- ADAPT then exits at k=0.
The default `--cs-target-qubits 0` (i.e. skip CS) keeps the start state
at HF, where ADAPT can grow toward CASCI.

**Comparison metric.**  "How long did it take to train?" is measured as
the **total number of cost-function evaluations**, not as
`n_iterations`.  ADAPT reports `n_iterations = number_of_operators_added`
(~tens), while HardwareEfficient reports `n_iterations =
COBYLA_outer_iters` (~hundreds), but each ADAPT outer step issues
`|operator_pool| * 2 + ~100` cost evaluations under the hood.  We
attach `CostEvalCounter` to each algorithm to capture the true number
of QNode evaluations, then also report the algorithm-specific outer
counter and wall time.

**Run it.**

```bash
cd framework
python -m experiments.trainability \
    --molecule water --param-targets 10 50 100 --max-iter 500
```

Batch across datasets2 amino acids:

```bash
cd framework
python -m experiments.trainability \
  --molecules ala arg asp cys gln glu gly his ile leu lys met phe pro ser thr trp tyr val \
  --param-targets 10 50 100 --max-iter 500
```

`trainability.py` does not have a `--batch-amino-acids` flag. Use `--molecules`
for a controlled amino-acid batch; `--all-molecules` also includes water and
small chemical fragments found under `datasets2/`. The current `asn.h5.gz`
file is a Hamiltonian file rather than the geometry file required by these
pipelines, so omit asparagine until its geometry dataset is repaired.

Useful flags: `--molecule hydrogen` (very fast, for plumbing checks),
`--algorithms qubit_adapt_vqe hardware_efficient_vqe`,
`--he-rotation-gates RY|RY_RZ|full`, `--adapt-gradient-threshold 1e-8`,
`--no-plots`.

**Outputs** (in `results/trainability/<timestamp>_<mol>_comparison/`):

- `run_config.json`
- `results.csv`              -- one row per (target, algorithm).
- `convergence_traces.csv`   -- long-format energy traces.
- `cost_evals_vs_params.png` -- log-y bars of cost-fn evals per algo.
- `final_energy_vs_params.png` -- final energy reached per algorithm
  with HF and CASCI reference lines.
- `runtime_vs_params.png`    -- wall-clock comparison.

## Experiment 4 -- ADAPT accuracy as a function of operator count

**Question.** For a fixed VQE problem, how does the achievable energy
improve as we allow the ansatz to use more variational parameters?

**Method.** Use `qubit_adapt_vqe` with an operator pool: at each step
it greedily picks the gradient-largest operator, so after `k` steps the
ansatz contains the (greedy) "top-k" operators for this Hamiltonian.

**Efficiency trick.** A single `qubit_adapt_vqe` run with
`max_operators = max(k_list)` records the energy after every operator
addition in `convergence_history`, so we recover energies at every
requested k value from one ADAPT run.  We also snapshot the cumulative
cost-fn count and timestamp at each addition (via a class-level property
that re-wraps `convergence_history` on every assignment so the hook
survives the `self.convergence_history = []` inside `run()`).

**Modular k-list.** The default sweep is `1, 2, 3, 5, 10`.  Pass
`--k-list` to evaluate any list -- including the `30 100 300` the user
asked about explicitly.  ADAPT runs once up to `max(k_list)` operators,
which can be slow for large values (each iteration runs a full pool
gradient screen + an inner COBYLA optimisation).

**Run it.**

```bash
cd framework
python -m experiments.accuracy_vs_params \
    --molecule water --k-list 1 2 3 5 10
```

Useful flags: `--algorithm adapt_vqe` (the simpler RY+CNOT version),
`--qubit-adapt-pool full|restricted` to select the Qubit-ADAPT pool,
`--gradient-threshold 1e-12` to keep adding operators even when
gradients are tiny, `--inner-max-iter 200`.

For Qubit-ADAPT, the full pool size and the pool size after the exact
Hamiltonian-commutator screen are different quantities. At 12 qubits the
generated full odd-Y pool contains 4,092 operators; only operators that
fail to commute with the Hamiltonian are gradient-screened. The run config
records both `n_pool_operators` and `n_screenable_operators`, so reports of
the screened candidate count can be compared without confusing it with the
generated pool size. If CS reduction is active, Qubit-ADAPT uses the full
pool even when `--qubit-adapt-pool restricted` was requested; the run config
records both the requested and effective pool modes.

**Outputs** (in `results/accuracy_vs_params/<timestamp>_<mol>_<algo>/`):

- `run_config.json`
- `results.csv`                    -- one row per requested k.
- `full_convergence.csv`           -- the entire ADAPT trace
  (one row per operator addition), so a finer "energy vs k" curve is
  always available even if `--k-list` is sparse.
- `energy_vs_n_params_linear.png`  -- linear energy axis with HF/CASCI lines.
- `energy_vs_n_params.png`         -- log-y |E - CASCI| with the
  1.6 mHa "chemical accuracy" line.
- `cost_evals_vs_k.png`            -- cumulative cost-fn calls to reach k.

## Experiment 5 -- Hamiltonian term-count sweep

This is separate from Experiment 4: it varies the number of Hamiltonian
terms retained in a prefix, not the VQE ansatz parameter count. Its plots
compare energies against the full CASCI reference, so their error includes
both Hamiltonian truncation and VQE optimization error; the per-run CSV also
stores error against the individual prefix Hamiltonian reference.

## Notes on initialisation strategies

| Strategy        | Distribution                      | Intent                                                         |
|-----------------|-----------------------------------|----------------------------------------------------------------|
| `random_uniform`| Uniform on `[0, 2*pi]`            | Textbook "naive" init, BP-prone (McClean 2018)                 |
| `small_random`  | Uniform on `[-pi/8, pi/8]`        | Grant et al. (2019) style, moderate-amplitude near zero        |
| `near_identity` | Uniform on `[-0.05, 0.05]`        | "Drop into the solution region" (close to HF for these ansatze) |
| `zeros`         | Identically zero                  | Deterministic HF reference for excitation-style ansatze         |

For `hardware_efficient_vqe`, the circuit prepares the HF reference state
*before* the variational layers, so near-zero rotations leave the state
close to HF and the cost landscape there has substantial gradient.

## Pre-flight smoke test

Cheap runs that hit the main code paths:

```bash
cd framework

# Experiment 1 (~1 min, 4 qubits via CS):
python -m experiments.noise_resilience \
    --molecule water --n-layers 2 --max-iter 100 \
    --noise-models depolarizing phaseflip \
    --noise-strengths 0 0.005 0.02

# Experiment 2 (~1-2 min, 4 qubits via CS):
python -m experiments.barren_plateau \
    --molecule water --layers 1 2 4 6 \
    --n-samples 15 --n-trials 3 --opt-layers 6 --max-iter 100

# Experiment 3 (~5 sec, 4 qubits, hydrogen):
python -m experiments.trainability \
    --molecule hydrogen --param-targets 4 10 --max-iter 200

# Experiment 4 (~80 sec, 12 qubits via active-space, no CS):
python -m experiments.accuracy_vs_params \
    --molecule water --k-list 1 2 3
```

For the full experiment 4 (`--k-list 1 2 3 5 10`) on water 12q expect
about 5 minutes; for `--k-list 1 2 3 5 10 30` expect ~20 minutes.

## Interpreting the results

**Noise resilience.** With water at 4 qubits we observe (typical run, sto-3g,
2 layers, 24 parameters):

- `||Δθ||₂` grows from 0 at `p=0` to ~2-3 by `p=0.02` for both noise
  channels.
- Cosine similarity stays above 0.85 -- the *direction* of the optimum
  parameter vector is mostly preserved, but the *magnitude* drifts.
- `phaseflip` typically causes more drift than `depolarizing` at the same
  strength.
- `E_clean_at_noisy_params` is sometimes *better* than the noiseless
  baseline -- noise can act as stochastic regularisation that lets COBYLA
  escape a noiseless local minimum.

**Barren plateaus and initialisation.** A strict McClean-style exponential
suppression of `Var[g_k]` requires more than 4 qubits; what we *do* see at
4 qubits + depths up to 10:

- `near_identity` keeps `Var[g_k] ~ 10⁻³`, an order of magnitude below the
  other strategies.  Small rotations + the HF-prep initial state means the
  cost function has small first-order sensitivity at start.
- `random_uniform` shows mild downward drift in `Var[g_k]` with depth
  (~10⁻¹ at depth 1 -> ~4·10⁻² at depth 6-10), the qualitative onset of
  a barren plateau.
- In the **optimisation comparison** at depth 6, `random_uniform` is
  consistently the *worst* in mean final energy and has the largest
  trial-to-trial variance, while `near_identity` is the most *consistent*
  (smallest std).  This is the practical consequence of "drop into the
  solution region": even when the gradient near HF is small, the landscape
  is bowl-shaped enough that COBYLA reliably descends to a good basin.

For a cleaner BP demonstration, run the BP experiment with `--cs-target-qubits
8` (or 0 to skip CS entirely) and `--layers 2 4 6 8 10 12`; the qubit-scaling
of the variance becomes much clearer at 8+ qubits, at the cost of longer
runtime.

**Trainability.** With hydrogen (4 qubits) and `--param-targets 4 10`
both algorithms reach CASCI in <1 s, but the cost-fn count tells the
real story:

- Qubit-ADAPT uses ~130 cost-fn evaluations to add 4 operators and hit
  CASCI to machine precision.
- Hardware-efficient with 8-12 RY parameters needs ~340 cost-fn
  evaluations and converges to within ~10⁻⁹ Ha.

That is, ADAPT is ~2-3x more efficient on hydrogen *per cost-fn call*,
even though its per-step work (pool gradient screen + inner COBYLA)
makes it look "slower" if you count operator additions.  When ADAPT
runs out of pool (~4 ops at 4 qubits), the `actual_n_params` column
records the truncated count and the `notes` column flags the early exit.

On water at 12 qubits the gap widens further: ADAPT closes most of the
HF -> CASCI gap (~60 mHa) within 10 operators, while hardware-efficient
needs many layers (and many more cost-fn evals) to even approach the
same energies.

**Accuracy vs n_parameters.** With water at 12 qubits and
`--k-list 1 2 3 5 10` (qubit-ADAPT, no CS) we observe:

- `|E - CASCI|` drops monotonically from ~40 mHa at k=1 to ~10 mHa at
  k=10 -- a clear, smooth "accuracy improves with parameter count" curve
  on the log-scale plot.
- The `cumulative_cost_evals` column shows the trade-off: each new
  operator costs ~250-300 additional cost-fn evaluations.
- Reaching the chemical-accuracy line (1.6 mHa) on this Hamiltonian
  requires at least k~30 operators, well below the 736-operator pool
  -- run with `--k-list 1 2 3 5 10 30 100` to see the curve flatten.
