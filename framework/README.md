# VQE Framework for Protein Hamiltonians

A modular framework for running and benchmarking multiple VQE (Variational Quantum Eigensolver) algorithms on protein Hamiltonians from the QMProt dataset. Enables algorithm comparison, active-space benchmarking, and rapid prototyping of new ansatz.

## 📁 Project Structure

```
framework/
├── README.md
├── requirements.txt
├── .env.example
├── .gitignore
├── main.py                     # Main entry point
├── config.py                   # Configuration settings
├── core/
│   ├── __init__.py
│   ├── hamiltonian_loader.py   # Load Hamiltonians from .txt and .json
│   ├── base_vqe.py             # Base VQE class for all algorithms
│   └── results_manager.py      # Save and load results
├── algorithms/
│   ├── __init__.py
│   ├── vqe_vanilla.py          # Standard VQE with UCCSD ansatz
│   ├── vqe_adapt.py            # ADAPT-VQE
│   ├── vqe_hardware_efficient.py # Hardware-efficient ansatz VQE
│   ├── vqe_qaoa_inspired.py    # QAOA-inspired VQE
|   ├── vqe_iqcc_inspired.py    # Variation of ADAPT-VQE based on iQCC
|   ├── vqe_iqcc_true.py        # Iterative Qubit Coupled Clustering
|   ├── vqe_CB.py               # Classically boosted VQE
|   └── vqe_hva.py              # Hamiltonian variational ansatz VQE
├── plotting/
│   ├── __init__.py
│   └── visualizer.py           # Plotting functions
├── data/
│   ├── hamiltonians/           # Store Hamiltonian .txt files
│   └── qmprot.json             # Molecule metadata
├── results/                    # Output results (JSON, CSV)
└── plots/                      # Generated plots
```

## 🚀 Quick Start

### 1. Install Dependencies

```bash
cd framework
pip install -r requirements.txt
```

### 2. Configure Environment

```bash
cp .env.example .env
# Edit .env with your settings
```

### 3. Add Hamiltonians

Place your Hamiltonian files (e.g., `hamiltonian_trp.txt`) in `data/hamiltonians/`.
In order to download them from qmprot, run the first two cells of hamiltonian_download.ipynb. Change the file names as necessary (note to devs: put the vars for the names at the top of the file to make it more user friendly)
OR instead of the .ipynb, just do run download_qmprot_hamiltonians.py

### 4. Run the Framework

```bash
# Run all algorithms on all Hamiltonians
python main.py --all

# Run specific algorithm on specific molecule
python main.py --molecule trp --algorithm vanilla_vqe

# Run all algorithms on one molecule
python main.py --molecule his --all-algorithms

# Generate plots only
python main.py --plot-only
```

## 📊 Supported VQE Algorithms

1. **Vanilla VQE** (`vqe_vanilla.py`) - Standard VQE with UCCSD ansatz
2. **ADAPT-VQE** (`vqe_adapt.py`) - Adaptive derivative-assembled pseudo-trotter VQE
3. **Hardware-Efficient VQE** (`vqe_hardware_efficient.py`) - Uses hardware-efficient ansatz
4. **QAOA-Inspired VQE** (`vqe_qaoa_inspired.py`) - QAOA-style parameterized ansatz
5. **iQCC-True VQE** (`vqe_iqcc_true.py`) - Iteratively updates circuits through canonical transformations of the Hamiltonian
6. **iQCC-Inspired VQE** (`vqe_iqcc_inspired.py`) - Adapt-VQE using iterative qubit coupled clustering
7. **CB VQE** (`vqe_CB.py`) - Classically boosted VQE
8. **HVA VQE** (`vqe_hva.py`) - Hamiltonian variational ansatz
# adding more

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

## 📤 Output Format

Results are saved in `results/` as JSON:

```json
{
  "molecule": "trp",
  "algorithm": "vanilla_vqe",
  "calculated_energy": -672.12340,
  "reference_energy": -672.12345,
  "error": 0.00005,
  "n_iterations": 150,
  "runtime_seconds": 45.2,
  "convergence_history": [...]
}
```

## License

MIT License
