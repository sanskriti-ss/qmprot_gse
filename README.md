# qmprot_gse

## Overview

qmprot_gse investigates the use of Variational Quantum Eigensolver (VQE) algorithms to compute ground-state energies (GSE) of amino acids using Hamiltonians derived from the [QMProt dataset](https://qmprot.org).

The repository serves both as a research platform for benchmarking VQE variants and as a modular framework for users to run VQE algorithms on their own data. Hamiltonians are built via an end-to-end pipeline: RHF to MP2 to CASCI active space to ordan-Wigner qubit Hamiltonian to contextual subspace reduction, all powered by PySCF and OpenFermion.

---

## Latest Results (2026-03-18)

Full benchmark: **11 algorithms × 5 amino acids** (ALA, ARG, ASP, CYS, GLY).  
Pipeline: cc-pVDZ basis, (10e,7o) active space [CYS: (12e,7o)], 14 qubits → 8 qubits after contextual subspace reduction.

![Comprehensive comparison](framework/plots/20260318_200609/comprehensive_comparison.png)

| Molecule | CASCI Ref (Ha) | HF (Ha) | Best VQE error (Ha) | Best algorithms |
|----------|---------------|---------|---------------------|-----------------|
| ALA (alanine) | −321.892 | −317.371 | 0.00232 | ADAPT, iQCC, UCC, NN-AE |
| ARG (arginine) | −602.934 | −600.089 | 0.000098 | ADAPT, iQCC, UCC, NN-AE |
| ASP (aspartic acid) | −509.517 | −506.088 | 0.00554 | ADAPT, iQCC, UCC, NN-AE |
| CYS (cysteine) | −719.414 | −716.732 | 0.00111 | ADAPT, iQCC, UCC, NN-AE |
| GLY (glycine) | −282.862 | −280.008 | 0.00733 | ADAPT, iQCC, UCC, NN-AE |

All errors are **positive** (variational principle holds throughout). ARG reaches near-chemical accuracy (0.1 mHa error).

![Error heatmap](framework/plots/20260318_200609/heatmap_error.png)

---

## Motivation

Near-term quantum hardware is constrained by qubit counts and circuit depth, making direct simulation of large molecules infeasible. This repository explores algorithmic trade-offs under these constraints — benchmarking 11 VQE variants on the same set of molecular Hamiltonians to understand accuracy, runtime, and convergence behavior.

---

## Repository Structure

```
qmprot_gse/
├── framework/                  # Main VQE benchmarking framework (see framework/README.md)
│   ├── main.py                 # Entry point: run algorithms, generate plots
│   ├── config.py               # Runtime configuration
│   ├── active_space_truncation/  # HF → CASCI → qubit Hamiltonian pipeline
│   │   ├── step1_orbitals.py   # RHF + MP2, orbital selection
│   │   ├── step2_active_space.py  # CASCI validation
│   │   └── step3_hamiltonian.py   # Spin-orbital expansion + Jordan-Wigner
│   ├── contextual_subspace/    # 14 → 8 qubit CS reduction
│   ├── core/                   # Hamiltonian loader, base VQE, HF verification
│   ├── algorithms/             # 11 VQE implementations
│   ├── datasets/               # Pre-built .h5 Hamiltonians (ala/arg/asp/cys/gly)
│   ├── plots/                  # Timestamped output plots
│   └── results/                # JSON + CSV results
└── explore_init/               # Exploratory notebooks
```

---

## Quick Start

```bash
cd framework
pip install -r requirements.txt

# Run all 11 algorithms on all 5 molecules
python main.py --all

# Run one algorithm on one molecule
python main.py --molecule ala --algorithm adapt_vqe

# Regenerate plots from existing results
python main.py --plot-only
```

Pre-built `.h5` Hamiltonians for all 5 amino acids are in `framework/datasets/`. See [framework/README.md](framework/README.md) for full documentation including algorithm descriptions, the Hamiltonian pipeline, and instructions for adding new algorithms.

---

## Supported Algorithms

| Algorithm | Description |
|-----------|-------------|
| Vanilla VQE | Hardware-efficient layered ansatz |
| ADAPT-VQE | Adaptive operator pool, grows ansatz by gradient |
| Qubit-ADAPT-VQE | ADAPT with Pauli-string pool |
| Hardware-Efficient VQE | Parameterized Ry + CNOT layers |
| QAOA-Inspired VQE | QAOA-style cost/mixer alternation |
| iQCC VQE | Iterative qubit coupled clustering |
| iQCC-Inspired VQE | ADAPT variant with iQCC operator pool |
| CB VQE | Classically boosted VQE |
| HVA VQE | Hamiltonian variational ansatz |
| UCC VQE | Unitary coupled-cluster singles & doubles |
| NN-AE VQE | Neural-network autoencoder ansatz |

---

## Practical Simulation Limits

Statevector simulation scales exponentially with qubit count:

| Qubits | Runtime | Notes |
|--------|---------|-------|
| 8–10 | Seconds | Recommended for experimentation |
| 14 | Minutes | Manageable |

This framework targets 8 qubits after contextual subspace reduction from a 14-qubit active-space Hamiltonian.

---

## Next Steps

- Compute first excited states in addition to ground-state energies
- Extend to larger amino acids and peptide chains
- Noise model simulation for hardware-realistic benchmarking


