# Why some VQE runs have large errors

## The five-molecule March comparison

The archive contains 55 runs across alanine, arginine, aspartic acid, cysteine and glycine. The table below uses the unweighted mean of the five absolute differences between the stored final energy and that molecule's active-space CASCI energy. All five molecules had eight contextual qubits. The timestamped JSON and incremental CSV agree. The archived plotting CSV lost the algorithm and energy in every row. The live CLI now has a regression that compares its three exports on real H₂. [Source rows](../plots/20260318_200609/run_results.csv) · [Saved parameters and histories](../plots/20260318_200609/all_results.json) · [Export regression](test_main_results.py)

| Archived circuit | Mean absolute error (Ha) | What the audit established |
|---|---:|---|
| Hardware-efficient | 1.340363 | The archived circuit at revision `db605e9` started at the zero bitstring and sampled uniform angles on [0, 2π]. Its separately reported HF energy did not verify the state entering this circuit. |
| Vanilla | 1.291904 | This circuit prepared the reference before applying its rotations. Even at zero angles its fixed CNOTs can permute the occupied bitstring. Its current generic PennyLane counterpart uses RX, RY, RZ and staggered nearest-neighbor CNOTs. |
| Classically boosted | 1.235390 | All five boost calls raised a PennyLane gate-construction error. The saved results contain the unboosted fits with zero improvement. |
| QAOA-inspired | 0.212520 | This two-layer ansatz initializes a product state and uses a fixed Ising phase operator. We reconstructed all five archived normalized states from their fitted angles. |
| Remaining seven methods | 0.003281 each, rounded | Their endpoint errors were at or very near the reference determinant's error. Several accepted zero operators. Similar energies alone establish little about their relative ability to recover correlation. |

For the failed boost, the logged reference determinants averaged 3.281 mHa above CASCI. A correctly solved Ritz problem containing that reference cannot have a higher minimum than its reference member. This bounds the **mean signed error** for those five cases. The actual corrected energies and absolute errors need the original projected operators and kets. On a separately frozen 26-term glycine input, the present CB solver matched an independent complex Ritz solve and improved its fitted energy by 0.120726 Ha. [Five failure locations and control](reproductions/study_checks/high_error_boost.json) · [Current CB regression](test_classical_boost.py)

## Three outliers on the same glycine operator

This distinct, reproducible case uses the first 200 terms of a 361-term STO-3G glycine Hamiltonian, four active spatial orbitals and eight qubits. The comparison holds the term ordering, coefficients, seed for each algorithm, two-layer circuit and COBYLA settings fixed. The exact prefix minimum is −279.150286556 Ha. Full molecular active-space CASCI is −279.134683412 Ha. Each entry below is an energy **above the exact prefix minimum**, in Hartree.

| Circuit | Original start, 64 evaluations | Same start, 800 evaluations | Explicit HF start¹ |
|---|---:|---:|---:|
| Hardware-efficient (`RY`, 24 angles) | 0.623794 | 0.038508 | 0.038755 |
| Generic PennyLane (48 angles) | 0.351010 | 0.244503 | 0.038772 |
| QAOA-inspired (12 angles) | 0.978809 | 0.119920 | 0.038772 |

¹ HE and generic use 64 evaluations. The QAOA number is the attainable energy from the explicit angles before optimization. A seeded 200-call fit from those angles stayed there within 10⁻¹² Ha. The three original-start results reproduce the archived 64-call rows to their printed precision. QAOA's two-layer circuit has a separate, provable representability limit on this operator. Its bound is 0.0000684 Ha, whereas the observed 64-call error was 0.978809 Ha. [QAOA circuit, bounds and seed controls](qaoa_audit.md) · [Archived 200-term rows](../results/profiling/paper-batches/beyond-10-gly/eight-all-200/batch_all_results.csv)

The default HE and generic states here had HF fidelities of 0.001295 and 3.43 × 10⁻⁸. Both can reach HF exactly by choosing first-layer Y rotations that compensate the fixed CNOT permutation. Independent NumPy state evolution and OpenFermion scoring reproduce the layered fits' reported energies within 4 × 10⁻¹³ Ha over 24 controls spanning this prefix, full four-qubit H₂ and a separate eight-qubit contextual glycine operator. Successful termination can still give a poor scientific result. On the generic circuit, COBYLA terminated successfully at the HF-initialized 800-call allowance, with 0.038772 Ha error. The original 64-call generic run carried a convergence flag even though it exhausted its budget. The current output retains optimizer status, budget and callback plateau separately. [Frozen operators, angles, outcomes](reproductions/layered_vqe/fits.json) · [State and endpoint checks](test_layered_controls.py) · [Stopping regressions](test_optimizer_termination.py)

For a replay from the framework directory, the hardware-efficient prefix command is `python -m smoke.layered_controls smoke/reproductions/layered_vqe/four_orbital_gly.npz --terms 200 --rotation-gates RY --algorithms hardware_efficient_vqe --seeds 20293 --budgets 64 800 --output /tmp/gly-he.json`. Select `pennylane_vqe` and seed `20242` for its paired control. Without prefix settings, the two full input controls use `hydrogen.npz` and `contextual_gly.npz` from the same directory and the default seed of 42. The explicit reference angles are diagnostic starts. The standard algorithm settings are retained for benchmarking.

## Scope for the manuscript

The March comparison and the 200-term control have different bases, active spaces and reference operators. Neither supplies the original per-molecule inputs for October Figure 12. That figure averages distances to full CASCI for changing prefix Hamiltonians and filters out QAOA-inspired VQE. A high plotted point can include a Hamiltonian-truncation effect even when the prefix eigenproblem is solved exactly. On fresh campaigns we save both errors, the ordered operators and the states. [Figure and reference audit](SCIENTIFIC_READINESS.md) · [Manuscript, Figure 12](../../QMProt.pdf)

Figure 4 on methionine plots mean Bayesian trial acquisitions at each evaluation. The low iteration count at which those curves begin oscillating does not identify convergence of the best fit. Independent November methionine data reproduce the printed molecular HF and CASCI references. Under a different 12-qubit setup, HE with HF preparation still ended 0.700–0.940 Ha above CASCI, while Qubit-ADAPT ended 0.58–1.33 mHa above it. The original individual fits and projected inputs needed to assign Figure 4's error to a specific cause are missing. [Figure 4 and independent methionine evidence](reproductions/study_checks/methionine_figure.md)
