# Campaign smoke tests

The [deep-circuit and study efficiency report](DEEP_EFFICIENCY.md) gives full-budget local and Modal timings, the fitted Bayesian comparison and commands for rechecking exact solver outputs. The [first efficiency sweep](EXACT_EFFICIENCY.md) records the earlier baseline.

From the repository root on a machine with Python 3.12:

```sh
cd framework
python3.12 -m venv .venv
source .venv/bin/activate
python -m pip install -r smoke/requirements.txt
python -m smoke.representative
```

For Modal, use the same directory and virtual environment:

```sh
python -m pip install modal==1.6.3
# Run 'modal setup' once if you need to authenticate this machine.
python -m modal run -m smoke.modal_representative
```

The five-study acceptance campaign runs hydrogen accuracy checkpoints, initializations with fitted
Bayesian surrogates, trainability and noise, plus a contextual-subspace glycine prefix sweep.
It also runs a separate registered-versus-fermionic ADAPT comparison on H₂ and glycine in the
same environment. Both use two electrons in two unprojected molecular orbitals for that control.
Run from `framework` with a fresh output directory.

```sh
python -m smoke.study_campaign results/smoke/five-studies
python -m modal run -m smoke.modal_study_campaign
python -m smoke.study_campaign --validate results/smoke/five-studies
```

The campaign also runs 30 initialization configurations and 12 noise configurations on hydrogen.
Their angles, optimizer traces, sector weights and independently replayed values are under
`solver-studies/`. Under `paired-studies/`, the native batch CLIs exercise 44 full-H₂ prefix fits
(11 solvers, two requested optimizers, two seeds), eight ADAPT accuracy checkpoints and six trainability
fits. Every scientific source snapshot is checked against the campaign. To replay trainability endpoints from any native single-molecule or batch
output, run `python -m experiments.replay_trainability /path/to/trainability-output` from
`framework`. The replay checks the frozen operator, reported energies, particle-number and spin
observables, accepted threshold witnesses and the expected case grid.

The local and Modal commands check each study's requested cases, restored Hamiltonian, reference
energies, plots and source archive. The initializer check recomputes gradients independently.
The prefix check diagonalizes each saved prefix and reconstructs the final true-iQCC state from
the saved generators and angles to measure its energy. A separate fixed 26-term glycine projection
must exercise repeated iQCC dressing. Its original-H energy is independently checked at every accepted step.
The fresh chemistry's contextual search can return a different sector and a different operator count.
The Modal command validates after downloading
the native files. Adaptive stopping can differ across CPU libraries after H₂ has reached CASCI.
The companion `fermionic-control.json` records solver settings, both geometry fingerprints, the
matched-pool comparisons, gate fidelities and particle-sector leakage. Its source and geometry
archive accompanies the five studies' framework snapshots.

Both commands run hydrogen trainability, the documented 19-molecule trainability batch and the hydrogen noise sweep. The small input tree preserves every dataset ID, filename and display name. It includes **published glycine** and 19 distinct H₂ geometries. Each produces a real Hamiltonian. The 18 amino-acid names attached to H₂ fixtures are for exercising input routing and reporting. Their energies describe H₂ at the fixture bond length.

The default shortens solver budgets, trainability targets and noise strengths. It keeps normal screening, stopping rules, chemistry, optimizers, plots and batch writes. It checks every molecule's saved CASCI against an independently computed reference within `1e-9` Ha. Hashes in each workload's `measurement.json` identify the actual input set. All three workloads together produced the same 226 relative file paths and 66 CSV headers as the archived full suite.

| Workload | Structural coverage |
|---|---|
| Hydrogen trainability, 4 qubits, targets 2 and 21, 44 evaluations each | Full ADAPT candidate scoring, operator addition and refit, exhaustion during screening and the inner-budget check. The 21-parameter HE circuit has five entangling layers. The optimizer triggers repeated callbacks. |
| All 19 amino-acid IDs, target 1, eight evaluations, two scored candidates | Gln and Glu have different Hamiltonians. Glycine freezes five core orbitals and retains a six-orbital, 12-qubit active space with 1,819 Pauli terms. Both algorithms run for every ID and write individual and aggregate reports. |
| Hydrogen noise, two layers, 24 iterations, strengths 0 and 0.01 | Clean optimization, all three analytic noise channels, clean rescoring of noisy optima, parameter drift and plots. |

With a reference or a paired run, the gate compares **every logical request** across implementations, including binary64 energies, parameter hashes, operator selections, evaluation order and callbacks. A strict comparison requires the same trajectory. Single-run validation also checks budgets, thresholds, noise metrics and artifacts. Structural coverage on these finite cases gives a fast regression signal. Reserve the larger workloads for later confirmation of exact optimizations and for changes in numerical arithmetic.

The same options select workloads locally and on Modal.

| Option | Batch inputs | Solver settings |
|---|---|---|
| default | All 19 IDs, with published glycine | Short |
| `--quick` | Tiny Gln and Glu | Short |
| `--real-inputs` | Published Gln and Glu, and published hydrogen for the single-molecule workloads | Same as `--quick` |
| `--tiny-full` | Same files as the default | Identical to `--full`, including targets 10/50/100, 2,000 evaluations per row and every noise strength |
| `--campaign` | Published Gly, Arg and Gln | Full |
| `--full` | All 19 published amino acids | Full |

Use the default for changes touching chemistry, solvers or batch I/O. The two-molecule `--quick` option shortens the feedback loop for localized changes. `--tiny-full` changes only the geometry root relative to `--full` and takes longer. Memory scaling, long-run convergence and full-run speedups require production-sized measurements. These campaigns use analytic simulation with contextual subspace reduction disabled. The optional 3D orbital renderer is outside their verified output set.

On the development machine the stock default took 18.45 s from launch through validation, and `--quick` took 8.67 s. On a two-core Modal worker the default took 93.5 s for stock and 79.5 s for the exact-reuse candidate. That one pair took 196.4 s from local launch through download and comparison. Both local and Modal comparisons matched all 721 objective/callback records and the scientific fields in 675 CSV rows per pair.

## Comparing implementations

For a strict optimization comparison, set `QMPROT_FRAMEWORK` to the absolute path of the candidate checkout's `framework` directory before running the command. The harness, fixtures, campaign settings and environment come from the checkout that launches the test. When the variable is unset, it tests the launching checkout's implementation. Each run needs a fresh output directory.

```sh
python -m smoke.representative --output results/smoke/reference
QMPROT_FRAMEWORK=/path/to/candidate/framework python -m smoke.representative \
  --output results/smoke/candidate --reference results/smoke/reference
python -m smoke.representative --check results/smoke/candidate
python -m smoke.comparison results/smoke/reference results/smoke/candidate
```

Modal accepts `--reference` pointing to a locally downloaded suite. `--repeat` runs two fresh suites on one worker and compares the downloaded files. Modal prints the local destination and Volume ID for each. Each workload saves its CLI log, wall time, source fingerprint, dependency versions, geometry hashes, native result files and an ordered event stream containing binary64 energies, parameter hashes, selected operator indices and callback flags. Timing includes the event recorder. The checker validates the requested grid, budgets, backend, traces, threshold crossings, optimized noise parameters and plots. It recognizes SciPy's specific minimum-iteration warning on the amino-acid grid, where the allocated inner COBYLA budget can fall below three evaluations.

Use `--pair` to time and compare two implementations on the **same Modal worker**. It runs the baseline followed by the candidate and retrieves both before reporting any numerical difference. Omitting `QMPROT_BASELINE_FRAMEWORK` selects the launching checkout for the baseline. Use checkouts that support the pinned smoke dependency versions.

```sh
QMPROT_BASELINE_FRAMEWORK=/path/to/baseline/framework \
QMPROT_FRAMEWORK=/path/to/candidate/framework \
  python -m modal run -m smoke.modal_representative --pair
```

The optional `exact_reuse.patch` reproduces the earlier controlled experiment at commit `b933595`. It reuses the ordered Qubit-ADAPT observable and previously evaluated HE callback energies on the deterministic statevector backend. For this comparison, select an unpatched checkout of the same revision as the baseline. Prepare the candidate from the repository root:

```sh
git worktree add --detach ../qmprot-exact b933595
git -C ../qmprot-exact apply --unidiff-zero "$PWD/framework/smoke/exact_reuse.patch"
```

A structurally valid run can follow a different optimization path. Use the reference comparison to detect differences hidden by ten-decimal CSV rounding, and compare on the same worker when measuring determinism. A strict comparison reports the first different optimizer input and the largest endpoint change it sees. Finite-precision or intentionally different optimization methods can fail that comparison while passing the structural checks. Larger active spaces, other optimizers, independent seeds and higher-budget ADAPT runs require separate workloads.

In the 100-parameter glycine HE case, the native and sparse energy calculations agreed within 1.71e-12 Ha on the first 101 inputs. COBYLA chose different parameters on evaluation 102. The endpoints differed by 0.1537 Ha at the full budget, and rerunning the original native objective reproduced its trajectory. Use the full-budget endpoint and threshold crossings when forecasting the effect of a kernel change on a sweep.

Run the regression tests from the same environment with `python -m unittest discover -s smoke -p 'test_*.py'`. The suite also exercises the production `campaigns/run.sh` dispatcher, its manifest and automatic output directories. The optimization harness invokes the same experiment modules through a recorder to capture per-request parameters and energies.

The separate trainability integration test runs both solvers on published glycine and arginine with eight screening candidates and 64 evaluations. Run it with `python -m smoke.batch` or `python -m modal run -m smoke.modal_batch`. Use `python -m smoke.batch --check results/smoke/<run-id>` to recheck its outputs.

## Geometry inputs

The HDF5 files under `geometries/` contain the QMProt authors' published atom symbols, Cartesian coordinates, charge, spin, atom count, name and formula for hydrogen and the 19 amino acids in the Experiment 3 batch command. The experiment builds each Hamiltonian from these coordinates at runtime. The input test verifies the fields read by the real loader against a digest computed from the pinned source fields. The published asparagine entry pairs 70 electrons with spin 1, so the documented grid omits it. [QMProt source geometries, Laia Coronas Sala and Parfait Atchade-Adelomou](https://github.com/pifparfait/qmprot_strategy/blob/71260c18bb341de12dd3dd23a86234254f3ed48c/QMProt/QMProt.json) · [QMProt collection](https://pennylane.ai/datasets/collection/qmprot) · [CC BY-SA 4.0](https://creativecommons.org/licenses/by-sa/4.0/)
