# VQE Framework for Protein Hamiltonians

A modular framework for running and benchmarking multiple VQE (Variational Quantum Eigensolver) algorithms on protein Hamiltonians from the QMProt dataset. Enables algorithm comparison, active-space benchmarking, and rapid prototyping of new ansatz.

## Latest Complete Benchmark Results (2026-04-02)

The latest complete comparison available in this checkout is [`plots/20260402_141955/run_results.csv`](plots/20260402_141955/run_results.csv): 11 algorithms across 16 amino acids and water. The CSV records 2 qubits for every row. Glutamine has two sets of algorithm rows; the table below reports its lowest error per algorithm. Errors are absolute energy differences in Hartree from the run's `reference_energy` values.

| Molecule | Best error (Ha) | Best algorithm |
|----------|----------------:|----------------|
| ALA | 0.002322 | HVA |
| ARG | 0.000098 | HVA |
| ASP | 0.005544 | HVA |
| CYS | 0.001107 | CB |
| GLU | 0.002697 | HVA |
| GLY | 0.007332 | CB |
| HIS | 0.000318 | HVA |
| HYD | 0.019998 | CB |
| ISO | 0.001040 | HVA |
| LEU | 0.003174 | HVA |
| LYS | 0.000665 | HVA |
| MET | 0.000159 | CB |
| PHE | 0.029984 | CB |
| PRO | 0.001420 | CB |
| THR | 0.000740 | CB |
| VAL | 0.001514 | HVA |
| WAT | 0.004693 | HVA |

Nine of the 17 systems meet the 1.6 mHa chemical-accuracy threshold by their best recorded error. The later output folders in `plots/` are partial runs or exports without usable complete comparison results, so they are not used here. This summary describes the recorded run data; it does not establish that all molecules used comparable active spaces or reduction settings.

## Experiment campaigns (run from `framework/`)

One campaign folder is one version of the comparison. It holds the inputs for every experiment run that day, and `run.sh` writes that campaign's results next to those inputs. After the code changes, copy the folder to a new id and run that id. Do not reuse yesterday's folder for today's code.

```bash
campaigns/run.sh 20261105
campaigns/run.sh 20261105 --dry-run    # print the commands, do not run them
```

`campaigns/20261105/` contains `01_trainability.json` (qubit-ADAPT against hardware-efficient) and `02_noise_resilience.json` (hardware-efficient noise sweep on the same molecule, no contextual subspace). Results land in `campaigns/20261105/results/<experiment>/`. `manifest.json` records the git commit, whether the tree was dirty, and the exact command of each experiment. Each experiment also writes its own `run_config.json` in that results folder.

`config.py` is not this record. It holds solver defaults such as the optimizer name and the shot count.

## Parallel 14-qubit amino-acid comparison (WSL)

From the repository root, use the existing `framework/.venv` to run the ten
non-HVA solvers on the 19 pinned whole amino acids:

```bash
bash run_amino19_14q_parallel.sh --dry-run --workers 4
bash run_amino19_14q_parallel.sh --workers 2 --plots
```

Each worker prepares one molecule's seven active orbitals and reuses its full
14-qubit Hamiltonian across the selected solvers. Each process uses one compute
thread. The default is two concurrent molecules. Use `--workers 4` if the WSL
memory allocation can accommodate four. To try a small batch at the same fit
settings, pass `--molecules ala gly --algorithms pennylane_vqe cb_vqe`.

The launcher prints its output directory, which defaults to
`framework/results/amino19_14q_parallel_<timestamp>/`. It saves one JSON file
per completed fit and a log per molecule under `molecules/`. To continue after
an interruption or retry missing fits, point `--resume` at that directory:

```bash
bash run_amino19_14q_parallel.sh --resume /path/to/parallel-run --workers 4 --plots
```

Resume uses the saved molecule and solver lists and checks the source,
configuration, geometry hashes and solver dependencies. It skips completed fits,
including fits that reached their optimizer limit. An interrupted fit starts from
its original initialization. Completed results are collected into `results/json/`
and `results/csv/`. With `--plots`, the framework writes plots after all fits finish.
The earlier serial launcher wrote a different format, so use a new directory for
the first parallel run. Worker runtimes can include CPU and memory contention.

## Key conventions
* `target_n_params` is what was requested; `actual_n_params` + `param_match`
  (`exact` | `rounded_up` | `stopped_early`) are in every CSV row and plot label.
  Hardware-efficient VQE matches the target exactly by default (`--he-param-match exact`).
* Both algorithms share `--eval-budget` (default 2000). Qubit-ADAPT gradient screening counts, and the remaining evaluations are divided across the operators still needed. Chemical accuracy is recorded from the trace and does not stop the run unless `--he-stop-at-accuracy` is passed.
* Hardware-efficient VQE prepares the Hartree–Fock state and starts from `small_random` angles on `[-π/8, π/8]` with `RY` rotations. Qubit-ADAPT prepares Hartree–Fock and grows a real Pauli pool. Pass `--no-he-hf-init` to start hardware-efficient from `|0…0>`.
* A large pool can spend the whole budget on the first gradient screen. The row then has `actual_n_params` below the target. That is the number the energy plot uses.

## Project Structure

```
framework/
├── README.md
├── requirements.txt
├── main.py                        # Main entry point
├── config.py                      # Solver defaults (optimizer, shots), not experiment inputs
├── campaigns/                     # One folder per code version; run.sh executes its JSON inputs
├── active_space_truncation/       # Pipeline: geometry → HF → CASCI → qubit Hamiltonian
│   ├── geometry.py                # Molecule geometry dataclass
│   ├── step1_orbitals.py          # HF + MP2 + orbital diagnostics
│   ├── step2_active_space.py      # CASCI active-space validation
│   └── step3_hamiltonian.py       # Qubit Hamiltonian via PySCF + OpenFermion
├── contextual_subspace/           # Contextual subspace (CS) reduction
│   ├── core.py                    # Noncontextual ground-state solver
│   └── cs_reduction.py            # 14 → 8 qubit CS projection
├── core/
│   ├── hamiltonian_loader.py      # Load Hamiltonians from .h5 / .txt / .json
│   ├── base_vqe.py                # Base VQE class
│   ├── hf_verification.py         # ⟨HF|H|HF⟩ verification before VQE
│   └── results_manager.py         # Save/load results
├── algorithms/
│   ├── vqe_pennylane.py           # Standard VQE (hardware-efficient ansatz)
│   ├── vqe_adapt.py               # ADAPT-VQE
│   ├── vqe_qubit_adapt.py         # Qubit-ADAPT-VQE
│   ├── vqe_hardware_efficient.py  # Hardware-efficient layered ansatz
│   ├── vqe_qaoa_inspired.py       # QAOA-inspired VQE
│   ├── vqe_iqcc.py                # iQCC (iterative qubit coupled clustering) with truncated approximation of canonical transform
│   ├── vqe_iqcc_inspired.py       # ADAPT-VQE architecture with iQCC selection process
│   ├── vqe_CB.py                  # Classically boosted VQE
│   ├── vqe_hva.py                 # Hamiltonian variational ansatz
│   ├── vqe_ucc.py                 # UCC singles & doubles
│   └── vqe_nn_ae.py               # Neural-network autoencoder VQE
├── plotting/
│   └── visualizer.py
├── datasets/                      # Pre-built .h5 Hamiltonians (ala/arg/asp/cys/gly)
├── results/                       # Output JSON + CSV
└── plots/                         # Generated plots (timestamped subdirs)
```

## Hamiltonian Pipeline

The `active_space_truncation/` pipeline builds qubit Hamiltonians directly from molecular geometry without needing pre-stored Hamiltonian files:

1. **Step 1** — RHF + MP2, orbital diagnostics, propose (10e,7o) active space  
2. **Step 2** — CASCI validation: confirms `E_CASCI < E_HF`  
3. **Step 3** — PySCF `ao2mo.full()` on active-space MOs only (avoids `n_basis⁴` OOM), spin-orbital expansion via OpenFermion's `spinorb_from_spatial` convention, Jordan–Wigner transformation → 14-qubit Hamiltonian  
4. **CS reduction** — Contextual subspace projector reduces 14 → 8 qubits while preserving the ground state

Pre-built `.h5` Hamiltonians for all 5 molecules are stored in `datasets/`.



## Quick Start

### 1. Install Dependencies

```bash
cd framework
pip install -r requirements.txt
```

### 2. Run the Framework

```bash
# Run all algorithms on all molecules
python main.py --all

# Run specific algorithm on specific molecule
python main.py --molecule ala --algorithm adapt_vqe

# Run all algorithms on one molecule
python main.py --molecule arg --all-algorithms

# Generate plots only (from existing results)
python main.py --plot-only
```

### 3. Using the Hamiltonian Pipeline (optional)

Pre-built `.h5` Hamiltonians are in `datasets/`. To rebuild from scratch:

```python
from active_space_truncation.geometry import MoleculeGeometry
from active_space_truncation.step1_orbitals import run_orbital_diagnostics
from active_space_truncation.step2_active_space import validate_active_space
from active_space_truncation.step3_hamiltonian import build_qubit_hamiltonian

geom = MoleculeGeometry.from_name("alanine")
diag = run_orbital_diagnostics(geom)
active = validate_active_space(geom, diag)
ham = build_qubit_hamiltonian(geom, active, diag)
```

## Supported VQE Algorithms

| # | Name | Key feature |
|---|------|-------------|
| 1 | **Pennylane VQE** | Hardware-efficient layered ansatz |
| 2 | **ADAPT-VQE** | Adaptive operator pool, grows ansatz by gradient |
| 3 | **Qubit-ADAPT-VQE** | ADAPT with Pauli-string pool |
| 4 | **Hardware-Efficient VQE** | Parameterized layers of Ry + CNOT |
| 5 | **QAOA-Inspired VQE** | QAOA-style cost/mixer alternation |
| 6 | **iQCC VQE** | Iterative qubit coupled clustering |
| 7 | **iQCC-Inspired VQE** | ADAPT-VQE architecture using iQCC operator pool |
| 8 | **CB VQE** | Classically boosted VQE |
| 9 | **HVA VQE** | Hamiltonian variational ansatz |
| 10 | **UCC VQE** | Unitary coupled-cluster singles & doubles |
| 11 | **NN-AE VQE** | Neural-network autoencoder ansatz |

## Visualization

The framework generates multiple types of plots:

- **Per-molecule plots**: Compare all algorithms for each molecule
- **Per-algorithm plots**: Compare all molecules for each algorithm
- **Energy convergence plots**: Track optimization progress
- **Error analysis**: Compare calculated vs. reference energies
- **Heatmaps**: Algorithm × Molecule performance matrix

## Adding New Algorithms

1. Create a new file in `algorithms/` (e.g., `vqe_custom.py`)
2. Inherit from `BaseVQE` class
3. Implement the required methods:
   - `build_ansatz()`
   - `run()`
4. Register in `algorithms/__init__.py`

Example:

```python
from core.base_vqe import BaseVQE

class CustomVQE(BaseVQE):
    def __init__(self, hamiltonian, **kwargs):
        super().__init__(hamiltonian, **kwargs)
        self.name = "custom_vqe"
    
    def build_ansatz(self):
        # Your ansatz implementation
        pass
    
    def run(self):
        # Your VQE implementation
        return self.optimize()
```

## Input Format

### Hamiltonian Files (`.txt`)

```
Coefficient	Operators
0.123456	IIII
-0.234567	ZIIZ
0.345678	XXYY
...
```

### Molecule Metadata (`qmprot.json`)

```json
{
  "amino_acids": [
    {
      "abbreviation": "trp",
      "name": "tryptophan",
      "n_qubits": 148,
      "n_coefficients": 42567891,
      "hamiltonian": "hamiltonian_trp.txt",
      "energy": -672.12345
    }
  ]
}
```

## Output Format

Results are saved in `results/` as JSON:

```json
{
  "molecule": "trp",
  "algorithm": "pennylane_vqe",
  "calculated_energy": -672.12340,
  "reference_energy": -672.12345,
  "error": 0.00005,
  "n_iterations": 150,
  "runtime_seconds": 45.2,
  "convergence_history": [...]
}
```

## License

See [LICENSE.md](../LICENSE.md) for the MIT software license and the separate QMProt dataset attribution.
