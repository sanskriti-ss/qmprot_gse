# QAOA-inspired molecular energies

The March archive contains five two-layer, eight-qubit QAOA-inspired solves. Each used 12 angles, analytic energy measurements, COBYLA, seed 42 and a 100-evaluation budget. Replaying their saved angles from the recorded gate equations gave state norms squared within 2.3 × 10⁻¹⁵ of one. Their original reduced Hamiltonians are unavailable, which prevents an independent energy rescore. The molecular HF values below come from the PySCF log. [Saved endpoints and angles](../plots/20260318_200609/all_results.json) · [Chemistry log](../run_all_cs.log)

| March molecule | Above active CASCI (mHa) | Above molecular HF (mHa) |
|---|---:|---:|
| Alanine | 259.252 | 256.930 |
| Arginine | 55.734 | 55.636 |
| Aspartic acid | 137.817 | 132.272 |
| Cysteine | 324.298 | 323.191 |
| Glycine | 285.498 | 278.166 |

## Budget and attainable energies

All entries below are Hartree above the lowest eigenvalue of the actual tested operator. Each native-start row uses the same seed across its three optimizer budgets. The witness uses the same gates with explicit angles that prepare the reference determinant exactly in the mathematical circuit. The first glycine case reproduces the original 200-term diagnostic to the ten digits recorded in its CSV. It has four active spatial orbitals and four active electrons. Its archived 200-term SHA-256, over little-endian float64 coefficients followed by the JSON-encoded Pauli list, is `31f965423e75c844d44a43e6bb266e23c22c3f9d5919938fa3490525de98614c`. The second glycine case projects six active orbitals to eight qubits with contextual subspaces. [Original 200-term CSV](../results/profiling/paper-batches/beyond-10-gly/eight-all-200/batch_all_results.csv) · [Local states, angles and rescored energies](../results/profiling/preflight/qaoa/qaoa.json)

| Input | Seed | Exact ground (Ha) | 64 calls | 200 calls | 800 calls | Determinant witness |
|---|---:|---:|---:|---:|---:|---:|
| Glycine, four active orbitals, 200 terms | 20310 | −279.150286556 | 0.978809 | 0.895076 | 0.119920 | 0.038772 |
| Glycine, contextual subspace, 200 terms | 20310 | −279.129599412 | 1.098659 | 1.049517 | 0.082471 | 0.018085 |
| H₂, full STO-3G Hamiltonian | 42 | −1.136846575 | 0.707348 | 0.306549 | 0.020265 | 0.019341 |

Increasing the 200-call seeds by one gave errors of 0.625900, 0.839438 and 0.739186 Ha respectively. Starting at the explicit HF angles left all three energies within 10⁻¹² Ha of those witnesses, even with an 800-call allowance. At 64 calls on H₂ the native fit had 39.1% probability outside the two-electron sector. Both seed sensitivity and the reachable HF state matter when interpreting the reported high errors. [Local full results](../results/profiling/preflight/qaoa/qaoa.json)

## Circuit and expressivity checks

The reference circuit starts from the uniform product state with independently variable single-qubit phases. Each cost block evolves under the fixed open-chain generator Σᵢ Zᵢ + ½Σᵢ ZᵢZᵢ₊₁. A shared transverse X rotation follows. The optional XY setting applies single-qubit RX and RY rotations successively. The molecular coefficients enter the final energy measurement. With first-layer γ = 0 and β = π/4, setting the initial phase to +π/2 for each desired zero and −π/2 for each desired one reaches any computational-basis determinant. Set the remaining layer angles to zero. [Circuit and parameter initialization](../algorithms/vqe_qaoa_inspired.py#L78)

An independent simulation applied the diagonal Ising phase to each bitstring and multiplied each wire by the corresponding two-by-two mixer matrix. Across 36 controls covering both glycine preparations and H₂, including each mixer, the largest amplitude difference from replaying the native gate tape was 6.5 × 10⁻¹⁶, the largest squared-norm error was 2.7 × 10⁻¹⁵, and the largest dense energy discrepancy from the device was 1.03 × 10⁻¹² Ha. [Independent control](../results/profiling/preflight/qaoa/audit.py)

There is also a strict depth limit for the two glycine cases. Each commuting nearest-neighbor ZZ layer grows a single-site observable by at most one site on each side. After two layers, sites at distance greater than four have disjoint backward light cones. The product input therefore gives them zero connected correlation. The unique four-orbital prefix ground state has connected Z₀Z₅ correlation −0.101605 and spectral gap 0.238640 Ha. For pure-state trace distance D from the ground state, each one- or two-site Pauli expectation changes by at most 2D. The connected correlation changes by at most 6D. Consequently every two-layer state lies at least 0.0000684 Ha above this prefix ground energy. This is a lower bound on the best attainable error. For contextual-subspace glycine the ground-state connected Z₀Z₆ correlation is −0.003656. The archive-level error of 0.978809 Ha exceeds the attainable determinant error by 0.940037 Ha, so the observed fit already loses that much within its own ansatz. [Spectra and correlations](../results/profiling/preflight/qaoa/qaoa.json)
