# Explroring VQE Implementation and Excited States

This repository contains a comprehensive implementation of Variational Quantum Eigensolvers (VQE) and advanced excited states calculations, based on research from https://www.nature.com/articles/s41534-021-00368-4.

## Features

### Core VQE Implementation (`quantum_vqe_gpu.py`)
- **Advanced Ansätze**: Hardware-efficient and UCCSD-inspired circuits
- **Optimizers**: Adam, SGD, and Quantum Natural Gradients
- **Convergence Analysis**: Early stopping, patience-based optimization
- **Quantum Fisher Information**: Parameter optimization guidance
- **Energy Landscape Analysis**: Local minima and optimization paths

### Excited States Calculator (`excited_states_calculator.py`)
- **Multiple Methods**:
  - Subspace expansion method
  - Quantum Equation of Motion (qEOM)
  - Variational Quantum Deflation
  - Folded spectrum method
- **Transition Properties**: Oscillator strengths and dipole moments
- **Energy Spectrum Analysis**: Complete electronic structure

### Research Workflow (`vqe_research_runner.py`)
- **Complete Pipeline**: From Hamiltonian loading to publication-quality results
- **Comparative Analysis**: Multiple methods comparison
- **Publication-Quality Plots**: Research-grade visualizations
- **Comprehensive Logging**: Detailed execution tracking

## Installation


#### For CPU-only:
```bash
pip install pennylane numpy matplotlib scipy tqdm
```


### Basic VQE Calculation
```python
from quantum_vqe_gpu import QuantumVQE, VQEConfig, load_hamiltonian_from_pennylane

# Load molecular system
molecular_system = load_hamiltonian_from_pennylane('ala', max_terms=5000)

# Configure VQE
config = VQEConfig(
    max_iterations=200,
    optimizer="adam",
    learning_rate=0.01,
    n_layers=3,
    backend="auto",  # Automatically selects best GPU backend
    calculate_excited_states=True
)

# Run optimization
vqe = QuantumVQE(config)
ground_energy, ground_params, history = vqe.optimize_ground_state(molecular_system)
```

### Excited States Calculation
```python
from excited_states_calculator import ExcitedStatesCalculator, ExcitedStateConfig

# Configure excited states calculation
excited_config = ExcitedStateConfig(
    method="subspace_expansion",  # or "deflation", "qeom", "folded_spectrum"
    n_excited_states=5,
    max_iterations=300
)

# Calculate excited states
calculator = ExcitedStatesCalculator(excited_config)
excited_results = calculator.calculate_excited_states(
    molecular_system.hamiltonian, 
    ground_params, 
    ground_energy
)
```

### Complete Research Workflow
```bash
# Basic workflow
python vqe_research_runner.py --molecule ala --gpu --excited-states



# Advanced options
python vqe_research_runner.py \
    --molecule ala \
    --excited-states \
    --max-terms 10000 \
    --max-iterations 500 \
    --n-layers 4 \
    --excited-method subspace_expansion

python vqe_research_runner.py --molecule water --max-iterations 50 --excited-states --n-layers 4

Feel free to experiment!
```

## Advanced Features


### Ansatz Comparison
```python
from quantum_vqe_gpu import compare_ansatz_performance

results = compare_ansatz_performance(molecular_system, config)
```

### Energy Landscape Analysis
```python
from quantum_vqe_gpu import analyze_energy_landscape

param_variations, energies = analyze_energy_landscape(
    vqe, molecular_system, optimal_params, n_samples=100
)
```

### Method Comparison for Excited States
```python
from excited_states_calculator import compare_excited_state_methods

comparison = compare_excited_state_methods(
    hamiltonian, ground_params, ground_energy
)
```

## Implementation Details

### VQE Algorithm Improvements
1. **Adaptive Learning Rates**: Adam optimizer with momentum
2. **Early Stopping**: Prevents overtraining with patience mechanism
3. **Parameter Initialization**: Smart initialization strategies
4. **Convergence Monitoring**: Real-time convergence tracking

### Excited States Methods

#### Subspace Expansion
- Creates orthogonal subspace of trial states
- Diagonalizes Hamiltonian within subspace
- Best for systems with well-separated states

#### Variational Quantum Deflation
- Uses penalty functions to enforce orthogonality
- Sequential optimization of excited states
- Robust for complex energy landscapes

#### Quantum Equation of Motion (qEOM)
- Based on excitation operators from ground state
- Efficient for molecular systems
- Captures dominant excitations

#### Folded Spectrum Method
- Targets specific energy regions
- Useful for finding states in dense spectra
- Can locate multiple states simultaneously

### GPU Optimization Strategies
1. **Batch Processing**: Vectorized operations across parameters
2. **Memory Management**: Efficient gradient computation
3. **Device Selection**: Automatic optimal device selection
4. **Precision Control**: Mixed precision for speed vs accuracy

## Research Applications

### Molecular Electronic Structure
- Ground state energies for drug discovery
- Excited states for photochemistry
- Transition properties for spectroscopy

### Quantum Algorithm Development
- Benchmarking new ansatz designs
- Optimization algorithm comparison
- Hardware efficiency analysis

### Quantum Hardware Studies
- Noise resilience analysis
- Circuit depth optimization
- Qubit connectivity requirements

## Output and Results

### Comprehensive Results File
Each run generates a JSON file with:
- Molecular system information
- Optimization parameters and history
- Ground and excited state energies
- Accuracy metrics and timing
- Method comparison data

### Plots
- VQE convergence analysis
- Energy level diagrams
- Error convergence plots
- Transition energy spectra
- Computational performance metrics

### Research Logs
Detailed logging includes:
- Device and backend information
- Optimization progress
- Error handling and warnings
- Performance benchmarks

## Performance Considerations


### Computational Scaling
- Ground state VQE: O(N²) for N-qubit systems
- Excited states: Additional factor of number of states
- GPU acceleration: 10-100x speedup depending on system size

### Accuracy vs Speed Trade-offs
- Exact simulation (shots=None): Highest accuracy
- Finite shots: Realistic quantum hardware simulation
- Reduced Hamiltonian terms: Faster but less accurate

## Troubleshooting

### Common Issues
1. **GPU Not Detected**: Check CUDA installation and drivers
2. **Memory Errors**: Reduce system size or increase GPU memory
3. **Convergence Issues**: Adjust learning rate or increase iterations
4. **Import Errors**: Verify all dependencies are installed

### Performance Optimization
1. **Use GPU acceleration** when available
2. **Limit Hamiltonian terms** for faster convergence
3. **Adjust ansatz depth** based on system complexity
4. **Monitor memory usage** for large systems

## Contributing

This implementation is designed for research purposes. Contributions welcome:
- New ansatz implementations
- Additional excited states methods
- GPU optimization improvements
- Benchmarking new molecular systems

## Citation

If you use this implementation in your research, please cite:
- The original VQE papers
- The Nature quantum computing paper: https://www.nature.com/articles/s41534-021-00368-4
- PennyLane library for quantum machine learning

## License

This implementation is provided for research and educational purposes. See MIT LICENSE file for details.
