# QMProt VQE Framework: Comprehensive Repository Guide

> **Purpose**: This document provides a complete technical reference for understanding, extending, and maintaining the QMProt VQE Framework. It details variable compatibility, data formats, electron mapping conventions, and module interactions.

---

## Table of Contents

1. [High-Level Overview Flow](#1-high-level-overview-flow)
2. [Configuration System (`config.py`)](#2-configuration-system-configpy)
3. [Hamiltonian Data Formats](#3-hamiltonian-data-formats)
4. [Molecule Metadata (`qmprot.json`)](#4-molecule-metadata-qmprotjson)
5. [Electron Mapping & Jordan-Wigner Convention](#5-electron-mapping--jordan-wigner-convention)
6. [The `QubitHamiltonian` Class](#6-the-qubithamiltonian-class)
7. [The `BaseVQE` Class Architecture](#7-the-basevqe-class-architecture)
8. [Backend & Noise Configuration](#8-backend--noise-configuration)
9. [Algorithm Implementation Guide](#9-algorithm-implementation-guide)
10. [Results Data Flow](#10-results-data-flow)
11. [Variable Compatibility Matrix](#11-variable-compatibility-matrix)
12. [Common Pitfalls & Debugging](#12-common-pitfalls--debugging)

---

## 1. High-Level Overview Flow

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                           USER INPUT (CLI or Script)                        │
│  python main.py --molecule ala gly --algorithm vanilla_vqe --backend-type   │
└─────────────────────────────────────────────────────────────────────────────┘
                                      │
                                      ▼
┌─────────────────────────────────────────────────────────────────────────────┐
│                              main.py                                        │
│  • Parses CLI arguments                                                     │
│  • Merges with config.py defaults                                           │
│  • Initializes VQEFramework                                                 │
└─────────────────────────────────────────────────────────────────────────────┘
                                      │
                                      ▼
┌─────────────────────────────────────────────────────────────────────────────┐
│                           VQEFramework                                      │
│  • Initializes HamiltonianLoader (points to datasets/ or data/hamiltonians/)│
│  • Initializes ResultsManager (points to results/)                          │
│  • Coordinates run_single(), run_molecule(), run_all()                      │
└─────────────────────────────────────────────────────────────────────────────┘
                                      │
                    ┌─────────────────┴─────────────────┐
                    ▼                                   ▼
┌───────────────────────────────────┐   ┌───────────────────────────────────┐
│      HamiltonianLoader            │   │      BackendConfig                │
│  • Auto-detects H5 vs legacy mode │   │  • Configures PennyLane device    │
│  • Loads from .h5 or .txt files   │   │  • Sets noise model (if any)      │
│  • Returns QubitHamiltonian       │   │  • Returns device + noise_inserter│
└───────────────────────────────────┘   └───────────────────────────────────┘
                    │                                   │
                    └─────────────────┬─────────────────┘
                                      ▼
┌─────────────────────────────────────────────────────────────────────────────┐
│                     Algorithm (e.g., VanillaVQE)                            │
│  • Inherits from BaseVQE                                                    │
│  • Receives: QubitHamiltonian, BackendConfig, optimizer params              │
│  • Implements: build_ansatz(), cost_function()                              │
│  • Runs: optimize() → returns VQEResult                                     │
└─────────────────────────────────────────────────────────────────────────────┘
                                      │
                                      ▼
┌─────────────────────────────────────────────────────────────────────────────┐
│                            VQEResult                                        │
│  • Contains: calculated_energy, reference_energy, error, n_qubits, etc.     │
│  • Serializable to JSON via to_dict()                                       │
└─────────────────────────────────────────────────────────────────────────────┘
                                      │
                                      ▼
┌─────────────────────────────────────────────────────────────────────────────┐
│                         ResultsManager                                      │
│  • Stores VQEResult objects in memory                                       │
│  • Saves to JSON (results/json/) and CSV (results/csv/)                     │
│  • Provides filtering: get_results_by_molecule(), get_results_by_algorithm()│
└─────────────────────────────────────────────────────────────────────────────┘
                                      │
                                      ▼
┌─────────────────────────────────────────────────────────────────────────────┐
│                          VQEVisualizer                                      │
│  • Reads results from ResultsManager                                        │
│  • Generates plots: molecule comparisons, convergence, heatmaps             │
│  • Saves to plots/<timestamp>/                                              │
└─────────────────────────────────────────────────────────────────────────────┘
```

---

## 2. Configuration System (`config.py`)

The `config.py` file is the **single source of truth** for default values. All other modules import from here rather than defining their own defaults.

### 2.1 Path Configuration

| Variable | Default Value | Description |
|----------|---------------|-------------|
| `FRAMEWORK_DIR` | Auto-detected | Absolute path to `framework/` directory |
| `DATA_DIR` | `framework/data/` | General data directory |
| `DATASETS_DIR` | `framework/datasets/` | **Primary**: HDF5 molecule datasets (`.h5` files) |
| `HAMILTONIANS_DIR` | `framework/data/hamiltonians/` | **Legacy**: Text-based hamiltonians (`.txt` files) |
| `MOLECULES_JSON` | `framework/data/qmprot.json` | Molecule metadata file |
| `RESULTS_DIR` | `framework/results/` | Output directory for results |
| `PLOTS_DIR` | `framework/plots/` | Output directory for visualizations |
| `LOGS_DIR` | `framework/logs/` | Log files |

**Environment Variable Override**: All paths can be overridden via environment variables (loaded from `.env`):
```bash
DATASETS_DIR=/path/to/custom/datasets
HAMILTONIANS_DIR=/path/to/custom/hamiltonians
```

### 2.2 VQE Algorithm Settings

| Variable | Default | Type | Description |
|----------|---------|------|-------------|
| `DEFAULT_OPTIMIZER` | `"COBYLA"` | `str` | Scipy optimizer method |
| `MAX_ITERATIONS` | `1000` | `int` | Maximum optimization iterations |
| `CONVERGENCE_THRESHOLD` | `1e-6` | `float` | Energy convergence threshold (Hartree) |
| `N_SHOTS` | `0` | `int` | Measurement shots (0 = analytic/exact) |
| `RANDOM_SEED` | `42` | `int` | For reproducibility |

### 2.3 Hamiltonian Truncation Settings

Large molecules (e.g., tryptophan with 42+ million terms) must be truncated for practical simulation:

| Variable | Default | Description |
|----------|---------|-------------|
| `HAMILTONIAN_MAX_TERMS` | `1000` | Maximum Pauli terms to retain |
| `HAMILTONIAN_TARGET_QUBITS` | `8` | Target qubit count after truncation |
| `HAMILTONIAN_MODE` | `"h5"` | `"h5"` (default) or `"legacy"` (txt files) |

### 2.4 Backend Settings

| Variable | Default | Options | Description |
|----------|---------|---------|-------------|
| `BACKEND` | `"pennylane"` | `pennylane`, `qiskit` | Quantum framework |
| `PENNYLANE_DEVICE` | `"default.qubit"` | See PennyLane docs | Default device name |
| `BACKEND_TYPE` | `"statevector"` | `"statevector"`, `"noisy"` | Simulation mode |
| `NOISE_MODEL` | `"depolarizing"` | See Section 8 | Noise channel type |
| `NOISE_STRENGTH` | `0.01` | `float` | Error probability per gate |

### 2.5 Supported Values

```python
SUPPORTED_OPTIMIZERS = [
    "COBYLA",      # Constrained Optimization BY Linear Approximations
    "L-BFGS-B",    # Limited-memory BFGS with bounds
    "SLSQP",       # Sequential Least Squares Programming
    "SPSA",        # Simultaneous Perturbation Stochastic Approximation
    "ADAM",        # Adaptive Moment Estimation (gradient-based)
    "GradientDescent",
    "NelderMead",  # Simplex method
]

SUPPORTED_BACKENDS = ["pennylane", "qiskit"]
```

---

## 3. Hamiltonian Data Formats

The framework supports **two data formats**, automatically detected by `HamiltonianLoader`:

### 3.1 HDF5 Format (`.h5`) — **Default/Recommended**

Located in: `framework/datasets/<molecule>/<molecule>.h5`

**Structure**:
```
<molecule>.h5
├── name                    # (bytes) Full molecule name, e.g., "alanine"
├── abbreviation            # (bytes) Short name, e.g., "ala"
├── mf                      # (bytes) Molecular formula, e.g., "C3H7NO2"
├── n_qubits                # (int) Number of qubits in full Hamiltonian
├── n_coefficients          # (int) Total number of Pauli terms
├── n_electrons             # (int) Number of electrons
├── n_orbitals              # (int) Number of spatial orbitals
├── charge                  # (int) Molecular charge
├── spin                    # (int) Spin multiplicity
├── basis                   # (bytes) Basis set, e.g., "sto-3g"
├── energy                  # (float) Reference ground state energy (Hartree)
├── hamiltonian_0           # (bytes) First chunk of Hamiltonian terms
├── hamiltonian_1           # (bytes) Second chunk...
├── hamiltonian_N           # (bytes) ... chunked to avoid memory issues
```

**Hamiltonian Chunk Format** (inside each `hamiltonian_*` key):
```
<coefficient>\t<pauli_string>\n
-0.1234567890\tIIXZYI\n
0.0987654321\tZZIIII\n
```

### 3.2 Legacy Text Format (`.txt`)

Located in: `framework/data/hamiltonians/hamiltonian_<molecule>.txt`

**Format**:
```
Coefficient	Operators
-0.8124789456	IIII
0.1713456789	ZIII
-0.2234567890	IIZI
0.0456789012	ZZII
0.1678901234	XIXI
```

- Tab-separated (`\t`)
- First line is header (skipped during parsing)
- Pauli string length = number of qubits

### 3.3 Pauli String Convention

Pauli strings follow **big-endian** qubit ordering:
- String position `i` corresponds to qubit `i`
- `"ZIII"` means: Z on qubit 0, Identity on qubits 1, 2, 3
- `"XIXI"` means: X on qubits 0 and 2, Identity on qubits 1 and 3

**Valid characters**: `I` (Identity), `X` (Pauli-X), `Y` (Pauli-Y), `Z` (Pauli-Z)

---

## 4. Molecule Metadata (`qmprot.json`)

The `qmprot.json` file provides metadata for molecules. This is used when loading Hamiltonians and for result reporting.

### 4.1 Structure

```json
{
  "test_molecules": [
    {
      "abbreviation": "h2",
      "name": "hydrogen",
      "mf": "H2",
      "n_atoms": 2,
      "charge": 0,
      "n_electrons": 2,
      "n_orbitals": 2,
      "n_qubits": 4,
      "n_coefficients": 15,
      "hamiltonian": "hamiltonian_h2.txt",
      "energy": -1.137306035753,
      "basis": "sto-3g",
      "description": "Hydrogen molecule at equilibrium bond length"
    }
  ],
  "amino_acids": [
    {
      "abbreviation": "ala",
      "name": "alanine",
      "mf": "C3H7NO2",
      "n_atoms": 13,
      "charge": 0,
      "n_electrons": 48,
      "n_orbitals": 39,
      "n_qubits": 78,
      "n_coefficients": 4200000,
      "hamiltonian": "hamiltonian_ala.txt",
      "energy": -321.75231
    }
  ]
}
```

### 4.2 Critical Fields

| Field | Type | Required | Description |
|-------|------|----------|-------------|
| `abbreviation` | `str` | ✅ Yes | Unique identifier used in CLI and filenames |
| `name` | `str` | ✅ Yes | Human-readable name |
| `n_qubits` | `int` | ✅ Yes | Number of qubits (= 2 × n_orbitals for Jordan-Wigner) |
| `n_electrons` | `int` | ⚠️ Important | Used for Hartree-Fock state initialization |
| `energy` | `float` | ✅ Yes | Reference ground state energy in Hartree |
| `hamiltonian` | `str` | Legacy only | Filename for `.txt` format |

### 4.3 Relationship: n_qubits, n_orbitals, n_electrons

Under **Jordan-Wigner transformation**:
```
n_qubits = 2 × n_orbitals
```

For a closed-shell molecule:
```
n_electrons = 2 × n_occupied_orbitals (for singlet states)
```

The `n_electrons` field is critical for:
1. Hartree-Fock state preparation (initial state in VQE)
2. Verifying electron number conservation

---

## 5. Electron Mapping & Jordan-Wigner Convention

### 5.1 Jordan-Wigner Transformation

The framework assumes Hamiltonians have already been transformed from fermionic operators to qubit operators using the **Jordan-Wigner (JW) transformation**.

**Key properties of JW mapping**:
- Each spin-orbital maps to one qubit
- Qubit `|1⟩` = orbital occupied, `|0⟩` = orbital unoccupied
- Creation/annihilation operators become Pauli strings with Z-chains

### 5.2 Spin-Orbital Ordering

Standard convention (used in QMProt data):
```
Qubit index:  0     1     2     3     4     5    ...
Orbital:      0α    0β    1α    1β    2α    2β   ...
```

Where α = spin-up, β = spin-down.

Alternatively, some datasets use:
```
Qubit index:  0     1     2    ...   N    N+1   N+2   ...
Orbital:      0α    1α    2α   ...   0β   1β    2β    ...
```

**The framework does NOT perform this mapping** — it assumes the input Hamiltonian is already in qubit form.

### 5.3 Hartree-Fock State Initialization

The Hartree-Fock (HF) state is the ground state of the non-interacting system and serves as the initial state for VQE.

**Implementation** (from `hf_verification.py`):
```python
def _hf_bitstring(n_qubits: int, n_electrons: int) -> np.ndarray:
    """
    Convention (Jordan-Wigner): the first n_electrons spin-orbitals are
    occupied → qubit register |1…1 0…0⟩ (big-endian, qubit-0 is leftmost).
    """
    n_occ = min(n_electrons, n_qubits)
    # Build the integer index of |1…1 0…0⟩
    index = 0
    for q in range(n_occ):
        index |= 1 << (n_qubits - 1 - q)
    
    state = np.zeros(2 ** n_qubits, dtype=complex)
    state[index] = 1.0 + 0.0j
    return state
```

**Example**: For `n_electrons=4, n_qubits=8`:
- HF state = `|11110000⟩`
- Qubits 0, 1, 2, 3 are `|1⟩` (occupied)
- Qubits 4, 5, 6, 7 are `|0⟩` (unoccupied)

### 5.4 Preparing HF State in Circuits

In VQE ansatz circuits, the HF state is prepared by applying X gates to occupied qubits:

```python
# From vqe_vanilla.py
n_electrons = self.hamiltonian.molecule.n_electrons or n_qubits // 2
for i in range(min(n_electrons, n_qubits)):
    qml.PauliX(wires=i)  # Flip |0⟩ to |1⟩
```

**Important**: If `n_electrons` is not set in metadata, the code defaults to `n_qubits // 2` (half-filling).

---

## 6. The `QubitHamiltonian` Class

### 6.1 Data Structure

```python
@dataclass
class QubitHamiltonian:
    molecule: Molecule        # Metadata object
    coefficients: np.ndarray  # Shape: (n_terms,), dtype: float64
    pauli_strings: List[str]  # Length: n_terms, each string has n_qubits chars
    n_qubits: int             # Number of qubits
    n_terms: int              # Number of Pauli terms
```

### 6.2 Key Methods

#### Truncation (for large molecules)

```python
def truncate(self, max_terms: int = 1000, target_qubits: int = 8) -> QubitHamiltonian:
    """
    Strategy:
    1. Sort terms by |coefficient| descending
    2. Greedily add terms that don't exceed target_qubits
    3. Remap wire indices to be consecutive (0, 1, 2, ...)
    4. Calculate ground state energy of truncated system
    """
```

**Result**: Returns a new `QubitHamiltonian` with:
- Fewer terms (≤ `max_terms`)
- Fewer qubits (≤ `target_qubits`)
- `molecule.truncated_ground_state_energy` set to the exact ground state of the truncated Hamiltonian

#### Format Conversions

```python
def to_pennylane(self):
    """Convert to PennyLane qml.Hamiltonian"""
    # Returns qml.Hamiltonian(coeffs, ops)

def to_qiskit(self):
    """Convert to Qiskit SparsePauliOp"""
    # Note: Qiskit uses REVERSE qubit ordering
    pauli_labels = [ps[::-1] for ps in self.pauli_strings]  # Reverse!

def to_openfermion(self):
    """Convert to OpenFermion QubitOperator"""
```

### 6.3 Compatibility Requirements

When creating a `QubitHamiltonian` manually:

1. **Coefficient/Pauli alignment**: `len(coefficients) == len(pauli_strings)`
2. **Qubit consistency**: All `pauli_strings` must have the same length = `n_qubits`
3. **Valid characters**: Each character in pauli_strings ∈ {`I`, `X`, `Y`, `Z`}

---

## 7. The `BaseVQE` Class Architecture

### 7.1 Constructor Parameters

```python
class BaseVQE(ABC):
    def __init__(self,
                 hamiltonian: QubitHamiltonian,  # Required
                 optimizer: str = "COBYLA",
                 max_iterations: int = 100,
                 convergence_threshold: float = 1e-6,
                 n_shots: int = 0,
                 random_seed: Optional[int] = None,
                 backend_config: Optional[BackendConfig] = None,
                 **kwargs):
```

| Parameter | Type | Source | Notes |
|-----------|------|--------|-------|
| `hamiltonian` | `QubitHamiltonian` | `HamiltonianLoader` | Must be passed, not loaded internally |
| `optimizer` | `str` | `config.DEFAULT_OPTIMIZER` | Must be in `SUPPORTED_OPTIMIZERS` |
| `max_iterations` | `int` | `config.MAX_ITERATIONS` | — |
| `convergence_threshold` | `float` | `config.CONVERGENCE_THRESHOLD` | — |
| `n_shots` | `int` | `config.N_SHOTS` | 0 = analytic mode |
| `random_seed` | `int` | `config.RANDOM_SEED` | Set via `np.random.seed()` |
| `backend_config` | `BackendConfig` | Constructed in `main.py` | If None, uses statevector |
| `**kwargs` | varies | Algorithm-specific | e.g., `n_layers` for VanillaVQE |

### 7.2 Abstract Methods (Must Implement)

```python
@abstractmethod
def build_ansatz(self) -> Any:
    """
    Build the variational ansatz circuit.
    Must set: self.n_parameters, self.device, self.cost_fn
    """
    pass

@abstractmethod
def cost_function(self, parameters: np.ndarray) -> float:
    """
    Evaluate ⟨ψ(θ)|H|ψ(θ)⟩
    """
    pass
```

### 7.3 Provided Methods

```python
def get_initial_parameters(self) -> np.ndarray:
    """Default: uniform random in [-0.1, 0.1]"""
    return np.random.uniform(-0.1, 0.1, self.n_parameters)

def callback(self, parameters: np.ndarray):
    """Called after each optimization step. Updates progress bar."""
    ...

def optimize(self, initial_parameters=None) -> Tuple[np.ndarray, float]:
    """Runs scipy.optimize.minimize with self.cost_function"""
    ...

def run(self) -> VQEResult:
    """
    Full VQE workflow:
    1. Compute HF energy (verification)
    2. Build ansatz
    3. Run optimization
    4. Package results into VQEResult
    """
    ...
```

### 7.4 Key Instance Variables Set During Execution

| Variable | Set In | Type | Description |
|----------|--------|------|-------------|
| `self.n_qubits` | `__init__` | `int` | From `hamiltonian.n_qubits` |
| `self.n_parameters` | `build_ansatz` | `int` | Total variational parameters |
| `self.device` | `build_ansatz` | PennyLane device | Quantum simulator |
| `self.cost_fn` | `build_ansatz` | `Callable` | QNode for energy evaluation |
| `self.hf_energy` | `run` | `float` | ⟨HF|H|HF⟩ energy |
| `self.convergence_history` | `optimize` | `List[float]` | Energy at each iteration |
| `self.optimal_parameters` | `optimize` | `np.ndarray` | Final optimized θ |
| `self.optimal_energy` | `optimize` | `float` | Final energy |

---

## 8. Backend & Noise Configuration

### 8.1 BackendConfig Class

```python
@dataclass
class BackendConfig:
    backend_type: str = "statevector"  # "statevector" or "noisy"
    device_name: str = "lightning.qubit"
    n_qubits: int = 0
    n_shots: int = 0  # 0 = analytic
    noise_model: Optional[str] = None
    noise_strength: float = 0.0
    extra: Dict[str, Any] = field(default_factory=dict)
```

### 8.2 Factory Methods

```python
# Noiseless statevector simulation
cfg = BackendConfig.statevector(
    n_qubits=8,
    device_name="lightning.qubit",  # Fast C++ backend
    n_shots=0,  # Exact expectation values
)

# Noisy density-matrix simulation
cfg = BackendConfig.noisy(
    n_qubits=8,
    noise_model="depolarizing",
    noise_strength=0.01,  # 1% error per gate
    device_name="default.mixed",  # Required for noise channels
)
```

### 8.3 Supported Noise Models

| Model | PennyLane Channel | Description |
|-------|-------------------|-------------|
| `"depolarizing"` | `qml.DepolarizingChannel(p)` | Random Pauli error with prob p |
| `"bitflip"` | `qml.BitFlip(p)` | X error with prob p |
| `"phaseflip"` | `qml.PhaseFlip(p)` | Z error with prob p |
| `"amplitude_damping"` | `qml.AmplitudeDamping(γ)` | T1-like decay |
| `"phase_damping"` | `qml.PhaseDamping(γ)` | T2-like dephasing |

### 8.4 Noise Insertion in Circuits

Noise is applied **after each variational layer** (not after every gate):

```python
# In VanillaVQE.build_ansatz()
insert_noise = self.noise_inserter  # From get_noise_inserter(backend_config)

@qml.qnode(self.device)
def circuit(params):
    for layer in range(n_layers):
        # ... apply rotations and CNOTs ...
        insert_noise()  # Apply noise to all qubits
    return qml.expval(H)
```

For statevector backend, `insert_noise()` is a no-op.

---

## 9. Algorithm Implementation Guide

### 9.1 Creating a New Algorithm

1. **Create file**: `framework/algorithms/vqe_custom.py`

2. **Implement class**:
```python
from core.base_vqe import BaseVQE, VQEResult
from core.hamiltonian_loader import QubitHamiltonian

class CustomVQE(BaseVQE):
    def __init__(self, hamiltonian: QubitHamiltonian, custom_param: int = 5, **kwargs):
        super().__init__(hamiltonian, **kwargs)
        self.name = "custom_vqe"  # IMPORTANT: Set unique name
        self.description = "My custom VQE algorithm"
        self.custom_param = custom_param
    
    def build_ansatz(self) -> Any:
        import pennylane as qml
        from core.backend_manager import create_device
        
        self.n_parameters = ...  # MUST set this
        self.device = create_device(self.backend_config)
        H = self.hamiltonian.to_pennylane()
        
        @qml.qnode(self.device)
        def circuit(params):
            # Prepare initial state (HF)
            n_electrons = self.hamiltonian.molecule.n_electrons or self.n_qubits // 2
            for i in range(min(n_electrons, self.n_qubits)):
                qml.PauliX(wires=i)
            
            # Your custom ansatz here
            ...
            
            self.noise_inserter()  # Apply noise if configured
            return qml.expval(H)
        
        self.cost_fn = circuit
        return circuit
    
    def cost_function(self, parameters: np.ndarray) -> float:
        return float(self.cost_fn(parameters))
```

3. **Register in `algorithms/__init__.py`**:
```python
from .vqe_custom import CustomVQE

ALGORITHMS = {
    ...
    "custom_vqe": CustomVQE,
}
```

### 9.2 Algorithm Checklist

- [ ] Set `self.name` (unique string identifier)
- [ ] Set `self.n_parameters` in `build_ansatz()`
- [ ] Create device via `create_device(self.backend_config)`
- [ ] Convert Hamiltonian via `self.hamiltonian.to_pennylane()`
- [ ] Initialize HF state in circuit
- [ ] Call `self.noise_inserter()` after each layer
- [ ] Return `qml.expval(H)` from QNode

---

## 10. Results Data Flow

### 10.1 VQEResult Dataclass

```python
@dataclass
class VQEResult:
    # Identification
    molecule_abbrev: str      # e.g., "ala"
    molecule_name: str        # e.g., "alanine"
    algorithm_name: str       # e.g., "vanilla_vqe"
    
    # Energy values (all in Hartree)
    calculated_energy: float  # VQE optimized energy
    reference_energy: float   # From qmprot.json or H5 file
    error: float              # calculated - reference
    relative_error: float     # |error| / |reference|
    
    # Optimization metadata
    n_iterations: int
    n_qubits: int
    n_parameters: int
    runtime_seconds: float
    convergence_history: List[float]
    optimal_parameters: Optional[np.ndarray]
    converged: bool
    
    # Backend info
    backend_type: str         # "statevector" or "noisy"
    noise_model: Optional[str]
    noise_strength: float
    
    # Verification
    hf_energy: Optional[float]  # ⟨HF|H|HF⟩
```

### 10.2 Serialization Flow

```
VQEResult
    │
    ▼ .to_dict()
JSON-serializable dict
    │
    ▼ json.dump()
results/json/<timestamp>.json
    │
    ▼ pandas.DataFrame()
results/csv/<timestamp>.csv
```

### 10.3 Important Notes on Energy Comparisons

When truncation is applied:
- `reference_energy`: Always the FULL system reference from metadata
- `truncated_ground_state_energy`: Exact ground state of the truncated Hamiltonian

The VQE error is computed against the **full system reference**, which means truncated results will show larger errors. This is intentional — it measures how well the truncated approximation represents the full molecule.

---

## 11. Variable Compatibility Matrix

This table shows which variables must be compatible across modules:

| Variable | Defined In | Used In | Must Match |
|----------|------------|---------|------------|
| `n_qubits` | H5/JSON metadata | `QubitHamiltonian`, `BackendConfig`, ansatz | All must agree |
| `n_electrons` | H5/JSON metadata | HF state prep, `hf_verification` | Must be ≤ n_qubits |
| `optimizer` | CLI/config | `BaseVQE.optimizer_name` | Must be in `SUPPORTED_OPTIMIZERS` |
| `backend_type` | CLI/config | `BackendConfig.backend_type` | `"statevector"` or `"noisy"` |
| `noise_model` | CLI/config | `BackendConfig.noise_model` | Must be valid channel name |
| `pauli_strings[i]` | Hamiltonian file | `QubitHamiltonian.pauli_strings` | `len(ps) == n_qubits` |
| `coefficients` | Hamiltonian file | `QubitHamiltonian.coefficients` | `len == len(pauli_strings)` |
| `reference_energy` | H5/JSON | `VQEResult.reference_energy` | Same value passed through |

### 11.1 Type Compatibility

| Parameter | Expected Type | Common Errors |
|-----------|---------------|---------------|
| `hamiltonian` | `QubitHamiltonian` | Passing file path string |
| `optimizer` | `str` | Passing optimizer class |
| `max_iterations` | `int` | Passing float |
| `n_shots` | `int` | Passing string |
| `backend_config` | `BackendConfig` or `None` | Passing dict |

---

## 12. Common Pitfalls & Debugging

### 12.1 "Molecule not found" Error

**Cause**: Abbreviation not in `qmprot.json` AND no matching H5 file.

**Fix**:
```bash
# Check available molecules
python main.py --list-molecules

# Ensure H5 file exists
ls framework/datasets/<abbrev>/<abbrev>.h5
```

### 12.2 "No valid Hamiltonian terms found"

**Cause**: Empty or malformed H5/txt file.

**Debug**:
```python
import h5py
with h5py.File("path/to/file.h5", "r") as f:
    print(list(f.keys()))
    print(f["hamiltonian_0"][()][:500])  # First 500 chars
```

### 12.3 Qubit Mismatch Errors

**Symptoms**: Shape errors in circuit, "wire X not in device"

**Cause**: `n_qubits` in metadata doesn't match Pauli string lengths.

**Fix**: Verify consistency:
```python
loader = HamiltonianLoader(DATASETS_DIR, MOLECULES_JSON)
ham = loader.load_hamiltonian("ala")
print(f"n_qubits from metadata: {ham.molecule.n_qubits}")
print(f"Pauli string length: {len(ham.pauli_strings[0])}")
# These MUST be equal
```

### 12.4 Very Slow Execution

**Cause**: Large Hamiltonian not being truncated.

**Fix**: Check truncation is being applied:
```python
# In main.py, these params control truncation:
"max_hamiltonian_terms": 1000,  # Increase if needed
"target_qubits": 8,             # Increase for more accuracy
```

### 12.5 Noisy Backend Not Working

**Symptoms**: Results identical to statevector.

**Check**:
1. Is `backend_type="noisy"` passed to algorithm?
2. Is `noise_model` set (not None)?
3. Is `noise_inserter()` called in the ansatz?

**Debug**:
```python
print(f"Backend config: {vqe.backend_config.to_dict()}")
print(f"Noise inserter callable: {vqe.noise_inserter}")
```

### 12.6 HF Energy Much Higher Than Reference

**Cause**: Usually indicates wrong `n_electrons` or different active space assumptions.

**Note**: For truncated Hamiltonians, HF energy may differ significantly from the full-system reference — this is expected.

---

## Appendix A: File Dependencies Graph

```
main.py
├── config.py (all defaults)
├── core/
│   ├── __init__.py
│   │   └── exports: HamiltonianLoader, ResultsManager
│   ├── hamiltonian_loader.py
│   │   ├── config.py (HAMILTONIAN_MAX_TERMS, HAMILTONIAN_TARGET_QUBITS)
│   │   └── h5py, numpy
│   ├── base_vqe.py
│   │   ├── hamiltonian_loader.py (QubitHamiltonian)
│   │   ├── backend_manager.py (BackendConfig, create_device)
│   │   └── hf_verification.py (compute_hf_energy)
│   ├── backend_manager.py
│   │   └── pennylane
│   ├── hf_verification.py
│   │   └── hamiltonian_loader.py (QubitHamiltonian)
│   └── results_manager.py
│       └── base_vqe.py (VQEResult)
├── algorithms/
│   ├── __init__.py
│   │   └── exports: ALGORITHMS, get_algorithm, list_algorithms
│   ├── vqe_vanilla.py
│   │   ├── core/base_vqe.py
│   │   └── core/backend_manager.py
│   ├── vqe_adapt.py
│   ├── vqe_hardware_efficient.py
│   └── vqe_qaoa_inspired.py
└── plotting/
    ├── __init__.py
    └── visualizer.py
        └── core/results_manager.py
```

---

## Appendix B: Quick Reference — Running Experiments

```bash
# Basic run
python main.py --molecule h2 --algorithm vanilla_vqe

# Multiple molecules
python main.py --molecule ala gly ser --algorithm vanilla_vqe

# All algorithms on one molecule
python main.py --molecule h2 --all-algorithms

# Noisy simulation
python main.py --molecule h2 --algorithm vanilla_vqe \
    --backend-type noisy \
    --noise-model depolarizing \
    --noise-strength 0.01

# Compare statevector vs noisy
python main.py --molecule h2 --algorithm vanilla_vqe --run-both-backends

# Custom truncation
python main.py --molecule ala --algorithm vanilla_vqe \
    --max-hamiltonian-terms 500 \
    --target-qubits 6

# Use legacy .txt files
python main.py --molecule h2 --algorithm vanilla_vqe --legacy
```

---

*Document version: 1.0 | Last updated: February 2026*
