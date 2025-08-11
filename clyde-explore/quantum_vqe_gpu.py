#!/usr/bin/env python3
"""

This implementation tries to extend the basic VQE approach to include:
1. GPU acceleration using PennyLane with JAX/TensorFlow backends (optional)
2. Ground state and excited state calculations
3. Advanced ansatz circuits (UCCSD-inspired)
4. Quantum natural gradients
5. Energy landscape analysis
6. Molecular orbital visualization capabilities

Based on concepts from:
- https://www.nature.com/articles/s41534-021-00368-4
- Variational quantum eigensolvers for electronic structure calculations
"""

import os
import sys
import time
import numpy as np
import matplotlib.pyplot as plt
from typing import List, Tuple, Dict, Optional, Union
from dataclasses import dataclass
from pathlib import Path
import json
import pickle
from tqdm import tqdm
import logging

# Quantum computing and ML libraries
import pennylane as qml
from pennylane import numpy as qml_numpy

# Import large system handler
try:
    from large_system_handler import LargeSystemHandler, LargeSystemConfig, create_reduced_vqe_config
    LARGE_SYSTEM_HANDLER_AVAILABLE = True
except ImportError:
    LARGE_SYSTEM_HANDLER_AVAILABLE = False
    print("⚠ Large system handler not available")

# Try to import GPU-accelerated backends
try:
    import jax
    import jax.numpy as jnp
    from jax import grad, jit, random
    JAX_AVAILABLE = True
    print("✓ JAX detected - GPU acceleration available")
except ImportError:
    JAX_AVAILABLE = False
    print("⚠ JAX not available - using CPU backend")

try:
    import tensorflow as tf
    TF_AVAILABLE = True
    print("✓ TensorFlow detected - GPU acceleration available")
except ImportError:
    TF_AVAILABLE = False
    print("⚠ TensorFlow not available")

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

@dataclass
class VQEConfig:
    """Configuration class for VQE calculations"""
    max_iterations: int = 500
    convergence_threshold: float = 1e-6
    patience: int = 50
    optimizer: str = "adam"  # adam, sgd, qng
    learning_rate: float = 0.01
    n_layers: int = 3
    shots: Optional[int] = None  # None for exact simulation
    backend: str = "lightning.qubit"  # lightning.qubit, lightning.qubit.jax, lightning.qubit.tf
    device_name: str = "auto"  # auto, cpu, gpu
    save_results: bool = True
    plot_convergence: bool = True
    calculate_excited_states: bool = True
    n_excited_states: int = 3

@dataclass
class MolecularSystem:
    """Container for molecular system information"""
    name: str
    n_qubits: int
    n_electrons: int
    hamiltonian: qml.Hamiltonian
    exact_energy: Optional[float] = None
    molecular_data: Optional[Dict] = None

class QuantumVQE:
    """Advanced VQE implementation with GPU support and excited states"""
    
    def __init__(self, config: VQEConfig, large_system_config: Optional[LargeSystemConfig] = None):
        self.config = config
        self.large_system_config = large_system_config or LargeSystemConfig()
        self.large_system_handler = LargeSystemHandler(self.large_system_config) if LARGE_SYSTEM_HANDLER_AVAILABLE else None
        self.setup_device()
        self.energies_history = []
        self.excited_states_history = []
        self.quantum_fisher_info = []
        self.processing_results = None  # Store large system processing results
        
    def setup_device(self):
        """Setup quantum device with GPU acceleration if available"""
        logger.info(f"Setting up quantum device: {self.config.backend}")
        
        # Get available PennyLane devices
        available_devices = list(qml.plugin_devices.keys())
        
        # Determine the best available backend
        if self.config.backend == "auto":
            if JAX_AVAILABLE and "lightning.qubit.jax" in available_devices and self.config.device_name in ["auto", "gpu"]:
                try:
                    # Check if GPU is available
                    devices = jax.devices()
                    gpu_devices = [d for d in devices if d.device_kind == 'gpu']
                    if gpu_devices:
                        self.config.backend = "lightning.qubit.jax"
                        logger.info(f"Using JAX GPU backend with {len(gpu_devices)} GPU(s)")
                    else:
                        self.config.backend = "lightning.qubit.jax"
                        logger.info("Using JAX CPU backend")
                except Exception as e:
                    logger.warning(f"JAX setup failed: {e}, falling back to default")
                    self.config.backend = "lightning.qubit"
            elif TF_AVAILABLE and "lightning.qubit.tf" in available_devices and self.config.device_name in ["auto", "gpu"]:
                try:
                    gpus = tf.config.list_physical_devices('GPU')
                    if gpus:
                        self.config.backend = "lightning.qubit.tf"
                        logger.info(f"Using TensorFlow GPU backend with {len(gpus)} GPU(s)")
                    else:
                        self.config.backend = "lightning.qubit.tf"
                        logger.info("Using TensorFlow CPU backend")
                except Exception as e:
                    logger.warning(f"TensorFlow setup failed: {e}, falling back to default")
                    self.config.backend = "lightning.qubit"
            else:
                self.config.backend = "lightning.qubit"
                logger.info("Using default PennyLane backend (GPU backends not available)")
        
        # Fallback if the selected backend is not available
        if self.config.backend not in available_devices:
            logger.warning(f"Backend {self.config.backend} not available, using default.qubit")
            self.config.backend = "lightning.qubit"

    def create_device(self, n_qubits: int):
        """Create quantum device for given number of qubits"""
        device_kwargs = {"wires": n_qubits}
        if self.config.shots is not None:
            device_kwargs["shots"] = self.config.shots
            
        self.device = qml.device(self.config.backend, **device_kwargs)
        logger.info(f"Created device: {n_qubits} qubits, backend: {self.config.backend}")
        return self.device
    
    def hardware_efficient_ansatz(self, params: np.ndarray, n_qubits: int, n_layers: int):
        """Hardware-efficient ansatz with parameterized rotations and entangling gates"""
        param_idx = 0
        
        for layer in range(n_layers):
            # Single-qubit rotations
            for qubit in range(n_qubits):
                qml.RY(params[param_idx], wires=qubit)
                param_idx += 1
                qml.RZ(params[param_idx], wires=qubit)
                param_idx += 1
            
            # Entangling gates (circular connectivity)
            for qubit in range(n_qubits):
                qml.CNOT(wires=[qubit, (qubit + 1) % n_qubits])
        
        # Final layer of single-qubit rotations
        for qubit in range(n_qubits):
            qml.RY(params[param_idx], wires=qubit)
            param_idx += 1
    
    def uccsd_inspired_ansatz(self, params: np.ndarray, n_qubits: int, n_electrons: int):
        """UCCSD-inspired ansatz for molecular systems"""
        # Initialize with Hartree-Fock state
        for i in range(n_electrons):
            qml.PauliX(wires=i)
        
        param_idx = 0
        
        # Single excitations 
        for i in range(n_electrons):
            for a in range(n_electrons, n_qubits):
                if param_idx < len(params):
                    # Apply parameterized single excitation using SingleExcitation
                    qml.SingleExcitation(params[param_idx], wires=[i, a])
                    param_idx += 1
                
        # Double excitations
        for i in range(n_electrons):
            for j in range(i + 1, n_electrons):
                for a in range(n_electrons, n_qubits):
                    for b in range(a + 1, n_qubits):
                        if param_idx < len(params):
                            qml.DoubleExcitation(params[param_idx], wires=[i, j, a, b])
                            param_idx += 1
    
    def create_cost_function(self, hamiltonian: qml.Hamiltonian, ansatz_type: str = "hardware_efficient"):
        """Create cost function for VQE optimization"""
        n_qubits = hamiltonian.num_wires
        
        @qml.qnode(self.device, interface="autograd")
        def cost_function(params):
            if ansatz_type == "hardware_efficient":
                self.hardware_efficient_ansatz(params, n_qubits, self.config.n_layers)
            elif ansatz_type == "uccsd":
                # Estimate number of electrons (simplified)
                n_electrons = n_qubits // 2
                self.uccsd_inspired_ansatz(params, n_qubits, n_electrons)
            
            return qml.expval(hamiltonian)
        
        return cost_function
    
    def create_excited_state_cost_function(self, hamiltonian: qml.Hamiltonian, 
                                         ground_state_params: np.ndarray,
                                         excited_level: int = 1,
                                         penalty_weight: float = 10.0):
        """Create cost function for excited state calculation using penalty method"""
        n_qubits = hamiltonian.num_wires
        
        @qml.qnode(self.device, interface="autograd")
        def ground_state_overlap(params):
            self.hardware_efficient_ansatz(params, n_qubits, self.config.n_layers)
            return qml.state()
        
        @qml.qnode(self.device, interface="autograd")
        def excited_cost_function(params):
            self.hardware_efficient_ansatz(params, n_qubits, self.config.n_layers)
            
            # Get current state
            current_state = qml.state()
            ground_state = ground_state_overlap(ground_state_params)
            
            # Calculate overlap penalty
            overlap = qml.math.abs(qml.math.dot(qml.math.conj(current_state), ground_state))**2
            penalty = penalty_weight * overlap
            
            # Hamiltonian expectation value
            energy = qml.expval(hamiltonian)
            
            return energy + penalty
        
        return excited_cost_function
    
    def setup_optimizer(self, params: np.ndarray):
        """Setup optimizer based on configuration"""
        if self.config.optimizer == "adam":
            return qml.AdamOptimizer(stepsize=self.config.learning_rate)
        elif self.config.optimizer == "sgd":
            return qml.GradientDescentOptimizer(stepsize=self.config.learning_rate)
        elif self.config.optimizer == "qng":
            return qml.QNGOptimizer(stepsize=self.config.learning_rate)
        else:
            raise ValueError(f"Unknown optimizer: {self.config.optimizer}")
    
    def calculate_quantum_fisher_information(self, cost_function, params: np.ndarray):
        """Calculate quantum Fisher information matrix for parameter optimization"""
        try:
            # This is a simplified QFI calculation
            grad_fn = qml.grad(cost_function)
            gradient = grad_fn(params)
            
            # Approximate QFI as outer product of gradients
            qfi = np.outer(gradient, gradient)
            return qfi
        except Exception as e:
            logger.warning(f"QFI calculation failed: {e}")
            return None
    
    def prepare_molecular_system(self, molecular_system: MolecularSystem) -> MolecularSystem:
        """Prepare molecular system for computation, handling large systems"""
        if not LARGE_SYSTEM_HANDLER_AVAILABLE or molecular_system.n_qubits <= self.large_system_config.max_qubits:
            logger.info(f"System size ({molecular_system.n_qubits} qubits) suitable for direct simulation")
            return molecular_system
        
        logger.info(f"Large system detected ({molecular_system.n_qubits} qubits) - applying reduction strategies")
        
        # Process large system
        self.processing_results = self.large_system_handler.process_large_system(
            molecular_system.hamiltonian, molecular_system.name
        )
        
        strategy = self.processing_results['strategy_used']
        
        if strategy == 'classical_approximation':
            # System too large for quantum simulation
            logger.error("System too large for quantum simulation")
            raise RuntimeError(f"System with {molecular_system.n_qubits} qubits is too large for classical simulation. "
                             f"Consider using classical quantum chemistry methods instead.")
        
        elif self.processing_results.get('reduced_hamiltonian') is not None:
            # Use reduced Hamiltonian
            reduced_h = self.processing_results['reduced_hamiltonian']
            reduced_system = MolecularSystem(
                name=f"{molecular_system.name}_reduced",
                n_qubits=reduced_h.num_wires,
                n_electrons=reduced_h.num_wires // 2,  # Estimate
                hamiltonian=reduced_h,
                exact_energy=molecular_system.exact_energy,  # Keep original for comparison
                molecular_data=molecular_system.molecular_data
            )
            
            # Update VQE config for reduced system
            if self.processing_results.get('mapping_info'):
                original_config = self.config
                self.config = create_reduced_vqe_config(original_config, reduced_system.n_qubits)
                logger.info(f"Updated VQE config for reduced system: {self.config.n_layers} layers, "
                           f"{self.config.max_iterations} iterations")
            
            logger.info(f"Using reduced system: {reduced_system.n_qubits} qubits "
                       f"(reduced from {molecular_system.n_qubits})")
            return reduced_system
        
        else:
            # Fallback - try with original system but warn user
            logger.warning("Could not reduce system size - attempting with original (may fail)")
            return molecular_system

    # def optimize_ground_state(self, molecular_system: MolecularSystem) -> Tuple[float, np.ndarray, List[float]]:
    def optimize_ground_state(self, molecular_system: MolecularSystem) -> Tuple[float, np.ndarray, List[float]]:
        """Optimize ground state energy using VQE"""
        # Prepare system (handle large systems)
        prepared_system = self.prepare_molecular_system(molecular_system)
        
        logger.info(f"Starting ground state optimization for {prepared_system.name}")
        
        # Create device and cost function
        self.create_device(prepared_system.n_qubits)
        cost_function = self.create_cost_function(prepared_system.hamiltonian)
        
        # Initialize parameters
        n_params = self.get_n_parameters(prepared_system.n_qubits)
        np.random.seed(42)
        params = qml_numpy.array(np.random.uniform(0, 2*np.pi, n_params), requires_grad=True)
        
        # Setup optimizer
        optimizer = self.setup_optimizer(params)
        
        # Optimization loop
        energies = []
        best_energy = float('inf')
        best_params = params.copy()
        patience_counter = 0
        
        logger.info(f"Starting optimization with {n_params} parameters")
        progress_bar = tqdm(range(self.config.max_iterations), desc="Ground State VQE")
        
        start_time = time.time()
        
        for step in progress_bar:
            try:
                # Optimization step
                params, energy = optimizer.step_and_cost(cost_function, params)
                energies.append(float(energy))
                
                # Track best parameters
                if energy < best_energy:
                    best_energy = float(energy)
                    best_params = params.copy()
                    patience_counter = 0
                else:
                    patience_counter += 1
                
                # Update progress bar
                progress_bar.set_postfix({
                    'Energy': f'{energy:.6f}',
                    'Best': f'{best_energy:.6f}',
                    'Patience': f'{patience_counter}/{self.config.patience}'
                })
                
                # Check convergence
                if step > 10:
                    recent_energies = energies[-10:]
                    if max(recent_energies) - min(recent_energies) < self.config.convergence_threshold:
                        logger.info(f"Converged after {step} iterations")
                        break
                
                # Early stopping
                if patience_counter >= self.config.patience:
                    logger.info(f"Early stopping after {step} iterations")
                    break
                    
                # Calculate QFI periodically
                if step % 50 == 0:
                    qfi = self.calculate_quantum_fisher_information(cost_function, params)
                    if qfi is not None:
                        self.quantum_fisher_info.append(qfi)
                        
            except Exception as e:
                logger.error(f"Error at step {step}: {e}")
                break
        
        optimization_time = time.time() - start_time
        logger.info(f"Ground state optimization completed in {optimization_time:.2f} seconds")
        
        self.energies_history = energies
        return best_energy, best_params, energies
    
    def calculate_excited_states(self, molecular_system: MolecularSystem, 
                               ground_state_params: np.ndarray) -> List[Tuple[float, np.ndarray]]:
        """Calculate excited states using orthogonality constraints"""
        if not self.config.calculate_excited_states:
            return []
        
        # Use the same prepared system as ground state optimization
        prepared_system = molecular_system
        if hasattr(self, 'processing_results') and self.processing_results:
            if self.processing_results.get('reduced_hamiltonian') is not None:
                reduced_h = self.processing_results['reduced_hamiltonian']
                prepared_system = MolecularSystem(
                    name=f"{molecular_system.name}_reduced",
                    n_qubits=reduced_h.num_wires,
                    n_electrons=reduced_h.num_wires // 2,
                    hamiltonian=reduced_h,
                    exact_energy=molecular_system.exact_energy,
                    molecular_data=molecular_system.molecular_data
                )
        
        logger.info(f"Calculating {self.config.n_excited_states} excited states for {prepared_system.name}")
        excited_states = []
        previous_states_params = [ground_state_params]
        
        for level in range(1, self.config.n_excited_states + 1):
            logger.info(f"Optimizing excited state level {level}")
            
            # Create cost function with orthogonality constraints
            cost_function = self.create_excited_state_cost_function(
                prepared_system.hamiltonian, 
                ground_state_params,
                excited_level=level
            )
            
            # Initialize with random parameters
            n_params = len(ground_state_params)
            np.random.seed(42 + level)
            params = qml_numpy.array(np.random.uniform(0, 2*np.pi, n_params), requires_grad=True)
            
            # Setup optimizer
            optimizer = self.setup_optimizer(params)
            
            # Optimization loop for excited state
            best_energy = float('inf')
            best_params = params.copy()
            
            for step in range(self.config.max_iterations // 2):  # Fewer iterations for excited states
                try:
                    params, energy = optimizer.step_and_cost(cost_function, params)
                    
                    if energy < best_energy:
                        best_energy = float(energy)
                        best_params = params.copy()
                        
                except Exception as e:
                    logger.warning(f"Error in excited state optimization: {e}")
                    break
            
            excited_states.append((best_energy, best_params))
            previous_states_params.append(best_params)
            logger.info(f"Excited state {level} energy: {best_energy:.6f} Hartree")
        
        return excited_states
    
    def get_n_parameters(self, n_qubits: int, ansatz_type: str = "hardware_efficient") -> int:
        """Calculate number of parameters for the ansatz"""
        if ansatz_type == "hardware_efficient":
            # For hardware-efficient ansatz: (2 rotations per qubit per layer) + final rotations
            return self.config.n_layers * n_qubits * 2 + n_qubits
        elif ansatz_type == "uccsd":
            # For UCCSD ansatz: single excitations + double excitations
            n_electrons = n_qubits // 2  # Rough estimate
            n_single_excitations = n_electrons * (n_qubits - n_electrons)
            n_double_excitations = (n_electrons * (n_electrons - 1) // 2) * ((n_qubits - n_electrons) * (n_qubits - n_electrons - 1) // 2)
            return n_single_excitations + n_double_excitations
        else:
            return self.config.n_layers * n_qubits * 2 + n_qubits
    
    def plot_results(self, molecular_system: MolecularSystem, 
                    ground_energy: float, excited_energies: List[float] = None):
        """Plot optimization results and energy levels"""
        if not self.config.plot_convergence:
            return
        
        fig, axes = plt.subplots(2, 2, figsize=(15, 12))
        
        # Plot 1: Ground state convergence
        axes[0, 0].plot(self.energies_history, 'b-', alpha=0.7, label='VQE Energy')
        if len(self.energies_history) > 20:
            window = 20
            moving_avg = np.convolve(self.energies_history, np.ones(window)/window, mode='valid')
            axes[0, 0].plot(range(window-1, len(self.energies_history)), moving_avg, 
                           'r-', linewidth=2, label='Moving Average')
        
        if molecular_system.exact_energy is not None:
            axes[0, 0].axhline(y=molecular_system.exact_energy, color='orange', 
                              linestyle='--', label='Exact Energy')
        
        axes[0, 0].set_xlabel('Iteration')
        axes[0, 0].set_ylabel('Energy (Hartree)')
        axes[0, 0].set_title(f'Ground State Convergence - {molecular_system.name}')
        axes[0, 0].legend()
        axes[0, 0].grid(True, alpha=0.3)
        
        # Plot 2: Energy improvement
        if len(self.energies_history) > 1:
            improvements = [self.energies_history[0] - e for e in self.energies_history]
            axes[0, 1].plot(improvements, 'g-', linewidth=2)
            axes[0, 1].set_xlabel('Iteration')
            axes[0, 1].set_ylabel('Energy Improvement (Hartree)')
            axes[0, 1].set_title('Energy Improvement Over Time')
            axes[0, 1].grid(True, alpha=0.3)
        
        # Plot 3: Energy level diagram
        if excited_energies:
            energy_levels = [ground_energy] + excited_energies
            level_labels = ['Ground'] + [f'Excited {i}' for i in range(1, len(excited_energies) + 1)]
            
            y_positions = range(len(energy_levels))
            axes[1, 0].barh(y_positions, energy_levels, alpha=0.7)
            axes[1, 0].set_yticks(y_positions)
            axes[1, 0].set_yticklabels(level_labels)
            axes[1, 0].set_xlabel('Energy (Hartree)')
            axes[1, 0].set_title('Energy Level Diagram')
            
            if molecular_system.exact_energy is not None:
                axes[1, 0].axvline(x=molecular_system.exact_energy, color='orange', 
                                  linestyle='--', label='Exact Ground State')
                axes[1, 0].legend()
        
        # Plot 4: Quantum Fisher Information (if available)
        if self.quantum_fisher_info:
            # Plot trace of QFI matrix over time
            qfi_traces = [np.trace(qfi) for qfi in self.quantum_fisher_info]
            axes[1, 1].plot(qfi_traces, 'purple', linewidth=2)
            axes[1, 1].set_xlabel('Optimization Step (×50)')
            axes[1, 1].set_ylabel('Tr(QFI)')
            axes[1, 1].set_title('Quantum Fisher Information Trace')
            axes[1, 1].grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        if self.config.save_results:
            plt.savefig(f'vqe_results_{molecular_system.name}_{int(time.time())}.png', 
                       dpi=300, bbox_inches='tight')
        
        plt.show()
    
    def save_results(self, molecular_system: MolecularSystem, 
                    ground_energy: float, ground_params: np.ndarray,
                    excited_states: List[Tuple[float, np.ndarray]]):
        """Save optimization results to file"""
        if not self.config.save_results:
            return
        
        results = {
            'molecular_system': {
                'name': molecular_system.name,
                'n_qubits': molecular_system.n_qubits,
                'n_electrons': molecular_system.n_electrons,
                'exact_energy': molecular_system.exact_energy
            },
            'ground_state': {
                'energy': ground_energy,
                'parameters': ground_params.tolist()
            },
            'excited_states': [
                {'energy': energy, 'parameters': params.tolist()}
                for energy, params in excited_states
            ],
            'optimization_history': self.energies_history,
            'config': {
                'max_iterations': self.config.max_iterations,
                'optimizer': self.config.optimizer,
                'learning_rate': self.config.learning_rate,
                'n_layers': self.config.n_layers,
                'backend': self.config.backend
            }
        }
        
        timestamp = int(time.time())
        filename = f'vqe_results_{molecular_system.name}_{timestamp}.json'
        
        with open(filename, 'w') as f:
            json.dump(results, f, indent=2)
        
        logger.info(f"Results saved to {filename}")

def load_hamiltonian_from_pennylane(molecule_name: str = 'ala', max_terms: int = 100) -> MolecularSystem:
    """Load molecular Hamiltonian from PennyLane datasets"""
    logger.info(f"Loading {molecule_name} dataset from PennyLane...")
    
    try:
        # Handle different molecule types and datasets
        if molecule_name.lower() in ['water', 'h2o']:
            # Load H2O from qchem dataset
            ds = qml.data.load("qchem", molname="H2O", bondlength=0.958, basis="STO-3G") # or 6-31G++
            dataset = ds[0] if isinstance(ds, list) else ds
            
            # Extract Hamiltonian and energy directly from qchem dataset
            hamiltonian = dataset.hamiltonian
            exact_energy = dataset.fci_energy if hasattr(dataset, 'fci_energy') else None
            
            # Get molecular parameters
            n_qubits = hamiltonian.num_wires
            n_electrons = 10  # Water has 10 electrons (8 from O + 2 from H atoms)
            
            logger.info(f"Loaded {molecule_name}: {n_qubits} qubits, {len(hamiltonian.coeffs)} terms")
            
            return MolecularSystem(
                name=molecule_name,
                n_qubits=n_qubits,
                n_electrons=n_electrons,
                hamiltonian=hamiltonian,
                exact_energy=exact_energy
            )
        
        else:
            # Load other molecules from 'other' dataset
            ds = qml.data.load('other', name=molecule_name)
            
            if isinstance(ds, list):
                dataset = ds[0]
            else:
                dataset = ds
            
            # Extract exact energy
            exact_energy = dataset.energy if hasattr(dataset, 'energy') else None
            
            # Extract Hamiltonian chunks
            hamiltonian_chunks = []
            if hasattr(dataset, 'list_attributes'):
                for key in dataset.list_attributes():
                    if "hamiltonian" in key:
                        hamiltonian_chunks.append(getattr(dataset, key))
            
            if not hamiltonian_chunks:
                raise ValueError("No Hamiltonian data found in dataset")
            
            # Combine and parse Hamiltonian
            full_hamiltonian = "".join(hamiltonian_chunks)
            coefficients, operators = parse_hamiltonian_string(full_hamiltonian, max_terms)
            
            # Build PennyLane Hamiltonian
            hamiltonian = qml.Hamiltonian(coefficients, operators)
            
            # Estimate number of electrons (simplified)
            n_qubits = hamiltonian.num_wires
            n_electrons = n_qubits // 2  # Rough estimate
            
            logger.info(f"Loaded {molecule_name}: {n_qubits} qubits, {len(coefficients)} terms")
            
            return MolecularSystem(
                name=molecule_name,
                n_qubits=n_qubits,
                n_electrons=n_electrons,
                hamiltonian=hamiltonian,
                exact_energy=exact_energy
            )
        
    except Exception as e:
        logger.error(f"Failed to load {molecule_name}: {e}")
        raise

def parse_hamiltonian_string(hamiltonian_string: str, max_terms: int) -> Tuple[List[float], List]:
    """Parse Hamiltonian string into coefficients and operators"""
    def string_to_operator(op_string: str):
        """Convert string representation to PennyLane operator"""
        if "Identity" in op_string:
            return qml.Identity(0)
        
        terms = op_string.split(" @ ")
        ops = []
        
        for term in terms:
            try:
                op, wire = term.split("(")
                wire = int(wire.strip(")"))
                if op == "X":
                    ops.append(qml.PauliX(wire))
                elif op == "Y":
                    ops.append(qml.PauliY(wire))
                elif op == "Z":
                    ops.append(qml.PauliZ(wire))
            except ValueError:
                continue
        
        return qml.prod(*ops) if len(ops) > 1 else ops[0]
    
    coefficients = []
    operators = []
    
    lines = hamiltonian_string.split("\n")
    valid_lines = [line.strip() for line in lines if line.strip() and 
                   "Coefficient" not in line and "Operators" not in line]
    
    if len(valid_lines) > max_terms:
        logger.info(f"Limiting to first {max_terms} terms out of {len(valid_lines)}")
        valid_lines = valid_lines[:max_terms]
    
    for line in tqdm(valid_lines, desc="Parsing Hamiltonian"):
        parts = line.split()
        try:
            coeff = float(parts[0])
            op_string = " ".join(parts[1:])
            coefficients.append(coeff)
            operators.append(string_to_operator(op_string))
        except (ValueError, IndexError):
            continue
    
    return coefficients, operators

def analyze_energy_landscape(vqe: QuantumVQE, molecular_system: MolecularSystem, 
                           optimal_params: np.ndarray, n_samples: int = 100):
    """Analyze the energy landscape around the optimal parameters"""
    logger.info("Analyzing energy landscape...")
    
    cost_function = vqe.create_cost_function(molecular_system.hamiltonian)
    
    # Sample parameters around the optimal point
    param_variations = []
    energies = []
    
    for i in range(n_samples):
        # Add small random perturbations
        noise_scale = 0.1
        perturbed_params = optimal_params + np.random.normal(0, noise_scale, len(optimal_params))
        
        try:
            energy = cost_function(perturbed_params)
            param_variations.append(perturbed_params)
            energies.append(float(energy))
        except Exception as e:
            continue
    
    # Analyze landscape properties
    energy_std = np.std(energies)
    energy_range = max(energies) - min(energies)
    
    logger.info(f"Energy landscape analysis:")
    logger.info(f"  Energy std: {energy_std:.6f} Hartree")
    logger.info(f"  Energy range: {energy_range:.6f} Hartree")
    
    return param_variations, energies

def compare_ansatz_performance(molecular_system: MolecularSystem, config: VQEConfig):
    """Compare different ansatz architectures"""
    logger.info("Comparing ansatz performance...")
    
    results = {}
    ansatz_types = ["hardware_efficient", "uccsd"]  # Can add "uccsd" when implemented
    
    for ansatz_type in ansatz_types:
        logger.info(f"Testing {ansatz_type} ansatz...")
        
        vqe = QuantumVQE(config)
        try:
            ground_energy, ground_params, energies = vqe.optimize_ground_state(molecular_system)
            results[ansatz_type] = {
                'ground_energy': ground_energy,
                'convergence_steps': len(energies),
                'final_gradient_norm': np.linalg.norm(np.gradient(energies[-10:]))
            }
        except Exception as e:
            logger.error(f"Failed to optimize with {ansatz_type}: {e}")
            results[ansatz_type] = None
    
    return results

def main():
    """Main function to run VQE calculations"""
    print("VQE Implementation")
    print("=" * 60)
    
    # Configuration
    config = VQEConfig(
        max_iterations=200,
        convergence_threshold=1e-6,
        patience=30,
        optimizer="adam",
        learning_rate=0.01,
        n_layers=3,
        shots=None,  # Exact simulation
        backend="auto",
        device_name="auto",
        save_results=True,
        plot_convergence=True,
        calculate_excited_states=True,
        n_excited_states=2
    )
    
    try:
        # Load molecular system
        molecular_system = load_hamiltonian_from_pennylane('ala', max_terms=5000)
        
        # Initialize VQE
        vqe = QuantumVQE(config)
        
        # Optimize ground state
        print("\nOptimizing Ground State...")
        ground_energy, ground_params, energy_history = vqe.optimize_ground_state(molecular_system)
        
        # Calculate excited states
        excited_states = []
        if config.calculate_excited_states:
            print("\nCalculating Excited States...")
            excited_states = vqe.calculate_excited_states(molecular_system, ground_params)
        
        # Analysis and results
        print("\nRESULTS SUMMARY")
        print("=" * 60)
        print(f"Molecular system: {molecular_system.name}")
        print(f"Number of qubits: {molecular_system.n_qubits}")
        print(f"Ground state energy: {ground_energy:.6f} Hartree")
        
        if molecular_system.exact_energy is not None:
            error = abs(ground_energy - molecular_system.exact_energy)
            print(f"Exact energy: {molecular_system.exact_energy:.6f} Hartree")
            print(f"Absolute error: {error:.6f} Hartree")
            print(f"Relative error: {(error/abs(molecular_system.exact_energy)*100):.3f}%")
        
        if excited_states:
            print(f"\nExcited state energies:")
            for i, (energy, _) in enumerate(excited_states):
                print(f"  Excited state {i+1}: {energy:.6f} Hartree")
        
        # Energy landscape analysis
        if len(ground_params) <= 50:  # Only for small parameter sets
            param_variations, landscape_energies = analyze_energy_landscape(
                vqe, molecular_system, ground_params, n_samples=50
            )
        
        # Plot results
        excited_energies = [energy for energy, _ in excited_states]
        vqe.plot_results(molecular_system, ground_energy, excited_energies)
        
        # Save results
        vqe.save_results(molecular_system, ground_energy, ground_params, excited_states)
        
        print("\nVQE calculation completed successfully!")
        
        # Comparing Ansatz
        print("\nComparing Ansatz Performance...")
        compare_ansatz_performance(vqe, molecular_system, ground_params)
        
    except Exception as e:
        logger.error(f"VQE calculation failed: {e}")
        raise

if __name__ == "__main__":
    main()
