#!/usr/bin/env python3
"""
Large Quantum System Handler

This module provides strategies for handling large molecular systems (>20-30 qubits)
that cannot be simulated classically in their full form.

Strategies Trying to be implemented:
1. Hamiltonian truncation and importance sampling
2. Active space reduction
3. Fragment-based calculations
"""

import numpy as np
import pennylane as qml
from typing import List, Tuple, Dict, Optional, Union
import logging
from dataclasses import dataclass
import time
from pathlib import Path
import json

logger = logging.getLogger(__name__)

@dataclass
class LargeSystemConfig:
    """Configuration for large system handling"""
    max_qubits: int = 20  # Maximum qubits for full simulation
    truncation_threshold: float = 1e-6  # Threshold for Hamiltonian term truncation
    active_space_size: int = 12  # Size of active space for reduction
    fragment_size: int = 8  # Size of molecular fragments
    use_importance_sampling: bool = True
    classical_shadows: bool = False  # Use classical shadow tomography
    tensor_network: bool = False  # Use tensor network approximation
    save_intermediate: bool = True

class LargeSystemHandler:
    """Handler for large molecular quantum systems"""
    
    def __init__(self, config: LargeSystemConfig):
        self.config = config
        self.original_hamiltonian = None
        self.reduced_hamiltonian = None
        self.qubit_mapping = None
        
    def analyze_system_size(self, hamiltonian: qml.Hamiltonian) -> Dict:
        """Analyze the molecular system and determine best strategy"""
        n_qubits = hamiltonian.num_wires
        n_terms = len(hamiltonian.coeffs)
        
        analysis = {
            'n_qubits': n_qubits,
            'n_terms': n_terms,
            'classical_simulation_feasible': n_qubits <= self.config.max_qubits,
            'recommended_strategy': None,
            'memory_estimate_gb': (2**n_qubits * 16) / (1024**3) if n_qubits <= 50 else float('inf'),
            'parameter_count_estimate': None
        }
        
        # Determine strategy based on system size
        if n_qubits <= 12:
            analysis['recommended_strategy'] = 'full_simulation'
        elif n_qubits <= 20:
            analysis['recommended_strategy'] = 'optimized_simulation'
        elif n_qubits <= 30:
            analysis['recommended_strategy'] = 'active_space_reduction'
        elif n_qubits <= 50:
            analysis['recommended_strategy'] = 'fragment_based'
        elif n_qubits <= 80:  # Allow larger systems to use fragment approach
            analysis['recommended_strategy'] = 'fragment_based'
        else:
            analysis['recommended_strategy'] = 'classical_approximation'
            
        # Estimate parameter count for hardware-efficient ansatz
        if n_qubits <= self.config.max_qubits:
            # Standard hardware-efficient ansatz
            analysis['parameter_count_estimate'] = 3 * n_qubits * 2 + n_qubits  # 3 layers
        else:
            # Reduced system
            analysis['parameter_count_estimate'] = 3 * self.config.active_space_size * 2 + self.config.active_space_size
            
        logger.info(f"System analysis: {n_qubits} qubits, {n_terms} terms")
        logger.info(f"Strategy: {analysis['recommended_strategy']}")
        logger.info(f"Memory estimate: {analysis['memory_estimate_gb']:.2f} GB")
        
        return analysis
    
    def truncate_hamiltonian(self, hamiltonian: qml.Hamiltonian, 
                           max_terms: Optional[int] = None) -> qml.Hamiltonian:
        """Truncate Hamiltonian by removing small terms"""
        coeffs = np.array(hamiltonian.coeffs)
        ops = hamiltonian.ops
        
        # Sort by absolute coefficient magnitude
        sorted_indices = np.argsort(np.abs(coeffs))[::-1]
        
        # Apply threshold truncation
        threshold_mask = np.abs(coeffs) >= self.config.truncation_threshold
        
        # Apply max terms limit if specified
        if max_terms is not None:
            max_terms = min(max_terms, len(coeffs))
            top_indices = sorted_indices[:max_terms]
            term_limit_mask = np.zeros(len(coeffs), dtype=bool)
            term_limit_mask[top_indices] = True
            final_mask = threshold_mask & term_limit_mask
        else:
            final_mask = threshold_mask
            
        # Create truncated Hamiltonian
        truncated_coeffs = coeffs[final_mask]
        truncated_ops = [ops[i] for i in range(len(ops)) if final_mask[i]]
        
        logger.info(f"Truncated Hamiltonian: {len(truncated_ops)} terms (from {len(ops)})")
        logger.info(f"Kept {np.sum(np.abs(truncated_coeffs))/np.sum(np.abs(coeffs))*100:.1f}% of total weight")
        
        return qml.Hamiltonian(truncated_coeffs, truncated_ops)
    
    def select_active_space(self, hamiltonian: qml.Hamiltonian) -> Tuple[qml.Hamiltonian, Dict]:
        """Select most important qubits for active space calculation"""
        n_qubits = hamiltonian.num_wires
        
        # Count qubit importance based on coefficient weights
        qubit_importance = np.zeros(n_qubits)
        
        for coeff, op in zip(hamiltonian.coeffs, hamiltonian.ops):
            if hasattr(op, 'wires'):
                for wire in op.wires:
                    qubit_importance[wire] += abs(coeff)
        
        # Select most important qubits
        active_qubits = np.argsort(qubit_importance)[-self.config.active_space_size:]
        active_qubits = sorted(active_qubits)
        
        # Create mapping from original to reduced system
        qubit_mapping = {orig: new for new, orig in enumerate(active_qubits)}
        
        # Filter Hamiltonian terms that only involve active qubits
        active_coeffs = []
        active_ops = []
        
        for coeff, op in zip(hamiltonian.coeffs, hamiltonian.ops):
            if hasattr(op, 'wires'):
                if all(wire in active_qubits for wire in op.wires):
                    # Remap wires to reduced system
                    new_op = self._remap_operator(op, qubit_mapping)
                    active_coeffs.append(coeff)
                    active_ops.append(new_op)
            else:
                # Identity terms
                active_coeffs.append(coeff)
                active_ops.append(op)
        
        reduced_hamiltonian = qml.Hamiltonian(active_coeffs, active_ops)
        
        mapping_info = {
            'active_qubits': active_qubits,
            'qubit_mapping': qubit_mapping,
            'qubit_importance': qubit_importance,
            'reduction_factor': len(active_qubits) / n_qubits
        }
        
        logger.info(f"Active space: {len(active_qubits)} qubits from {n_qubits}")
        logger.info(f"Active qubits: {active_qubits}")
        logger.info(f"Kept {len(active_ops)} terms from {len(hamiltonian.ops)}")
        
        return reduced_hamiltonian, mapping_info
    
    def _remap_operator(self, op, qubit_mapping):
        """Remap operator wires according to qubit mapping"""
        if not hasattr(op, 'wires') or len(op.wires) == 0:
            return op
            
        new_wires = [qubit_mapping[wire] for wire in op.wires if wire in qubit_mapping]
        
        # Handle different operator types
        if isinstance(op, qml.PauliX):
            return qml.PauliX(wires=new_wires[0])
        elif isinstance(op, qml.PauliY):
            return qml.PauliY(wires=new_wires[0])
        elif isinstance(op, qml.PauliZ):
            return qml.PauliZ(wires=new_wires[0])
        elif isinstance(op, qml.Identity):
            return qml.Identity(wires=new_wires[0] if new_wires else 0)
        elif hasattr(op, '_name') and 'Tensor' in op._name:
            # Handle tensor products
            remapped_factors = []
            for factor in op.obs:
                remapped_factor = self._remap_operator(factor, qubit_mapping)
                remapped_factors.append(remapped_factor)
            return qml.operation.Tensor(*remapped_factors)
        else:
            # Generic operator - try to create new instance with remapped wires
            try:
                op_class = type(op)
                if hasattr(op, 'parameters'):
                    return op_class(*op.parameters, wires=new_wires)
                else:
                    return op_class(wires=new_wires)
            except:
                logger.warning(f"Could not remap operator {op}, skipping")
                return None
    
    def fragment_molecule(self, hamiltonian: qml.Hamiltonian) -> List[Tuple[qml.Hamiltonian, List[int]]]:
        """Fragment large molecule into smaller pieces for separate calculation"""
        n_qubits = hamiltonian.num_wires
        fragments = []
        
        # Simple fragmentation: divide qubits into overlapping groups
        fragment_size = self.config.fragment_size
        overlap = fragment_size // 4  # 25% overlap between fragments
        
        start = 0
        while start < n_qubits:
            end = min(start + fragment_size, n_qubits)
            fragment_qubits = list(range(start, end))
            
            # Extract fragment Hamiltonian
            fragment_coeffs = []
            fragment_ops = []
            
            for coeff, op in zip(hamiltonian.coeffs, hamiltonian.ops):
                if hasattr(op, 'wires'):
                    if all(wire in fragment_qubits for wire in op.wires):
                        # Remap to local fragment numbering
                        local_mapping = {q: i for i, q in enumerate(fragment_qubits)}
                        local_op = self._remap_operator(op, local_mapping)
                        if local_op is not None:
                            fragment_coeffs.append(coeff)
                            fragment_ops.append(local_op)
                else:
                    # Identity terms
                    fragment_coeffs.append(coeff)
                    fragment_ops.append(op)
            
            if fragment_ops:  # Only add non-empty fragments
                fragment_hamiltonian = qml.Hamiltonian(fragment_coeffs, fragment_ops)
                fragments.append((fragment_hamiltonian, fragment_qubits))
            
            start += fragment_size - overlap
            
        logger.info(f"Created {len(fragments)} molecular fragments")
        for i, (frag_h, qubits) in enumerate(fragments):
            logger.info(f"Fragment {i}: qubits {qubits[0]}-{qubits[-1]}, {len(frag_h.ops)} terms")
        
        return fragments
    
    def estimate_ground_state_classically(self, hamiltonian: qml.Hamiltonian) -> float:
        """Estimate ground state energy using classical approximations"""
        logger.info("Computing classical energy estimate...")
        
        # Use truncated diagonalization for small systems
        if hamiltonian.num_wires <= 12:
            try:
                matrix = qml.utils.sparse_hamiltonian(hamiltonian)
                eigenvals = np.linalg.eigvals(matrix.toarray())
                return min(eigenvals).real
            except:
                pass
        
        # Fallback: sum of all negative terms (very rough lower bound)
        negative_sum = sum(coeff for coeff in hamiltonian.coeffs if coeff < 0)
        logger.info(f"Classical lower bound estimate: {negative_sum:.6f} Hartree")
        
        return negative_sum
    
    def process_large_system(self, hamiltonian: qml.Hamiltonian, 
                           molecule_name: str) -> Dict:
        """Main processing function for large molecular systems"""
        start_time = time.time()
        
        # Analyze system
        analysis = self.analyze_system_size(hamiltonian)
        strategy = analysis['recommended_strategy']
        
        results = {
            'original_system': analysis,
            'strategy_used': strategy,
            'processing_time': 0,
            'reduced_hamiltonian': None,
            'mapping_info': None,
            'fragments': None,
            'classical_estimate': None
        }
        
        self.original_hamiltonian = hamiltonian
        
        try:
            if strategy == 'full_simulation':
                # No reduction needed
                results['reduced_hamiltonian'] = hamiltonian
                logger.info("Using full simulation - no reduction needed")
                
            elif strategy == 'optimized_simulation':
                # Just truncate small terms
                truncated_h = self.truncate_hamiltonian(hamiltonian, max_terms=500)
                results['reduced_hamiltonian'] = truncated_h
                logger.info("Using optimized simulation with term truncation")
                
            elif strategy == 'active_space_reduction':
                # Truncate and reduce active space
                truncated_h = self.truncate_hamiltonian(hamiltonian, max_terms=1000)
                reduced_h, mapping_info = self.select_active_space(truncated_h)
                results['reduced_hamiltonian'] = reduced_h
                results['mapping_info'] = mapping_info
                logger.info("Using active space reduction")
                
            elif strategy == 'fragment_based':
                # Fragment the molecule
                fragments = self.fragment_molecule(hamiltonian)
                results['fragments'] = fragments
                # Use largest fragment as representative
                if fragments:
                    largest_fragment = max(fragments, key=lambda x: len(x[1]))
                    results['reduced_hamiltonian'] = largest_fragment[0]
                logger.info("Using fragment-based approach")
                
            elif strategy == 'classical_approximation':
                # Classical estimates only
                classical_estimate = self.estimate_ground_state_classically(hamiltonian)
                results['classical_estimate'] = classical_estimate
                logger.warning("System too large for quantum simulation - classical estimate only")
                
            # Save processing results
            if self.config.save_intermediate:
                self._save_processing_results(results, molecule_name)
                
        except Exception as e:
            logger.error(f"Error processing large system: {e}")
            results['error'] = str(e)
        
        results['processing_time'] = time.time() - start_time
        logger.info(f"Large system processing completed in {results['processing_time']:.2f} seconds")
        
        return results
    
    def _save_processing_results(self, results: Dict, molecule_name: str):
        """Save processing results to file"""
        save_data = {
            'molecule_name': molecule_name,
            'timestamp': time.strftime('%Y-%m-%d %H:%M:%S'),
            'strategy_used': results['strategy_used'],
            'original_qubits': results['original_system']['n_qubits'],
            'original_terms': results['original_system']['n_terms'],
            'processing_time': results['processing_time']
        }
        
        if results.get('mapping_info'):
            save_data['active_space_info'] = {
                'active_qubits': results['mapping_info']['active_qubits'],
                'reduction_factor': results['mapping_info']['reduction_factor']
            }
        
        filename = f"large_system_processing_{molecule_name}_{int(time.time())}.json"
        with open(filename, 'w') as f:
            json.dump(save_data, f, indent=2)
            
        logger.info(f"Processing results saved to {filename}")

def create_reduced_vqe_config(original_config, reduced_system_size: int):
    """Create VQE configuration optimized for reduced system"""
    # Import here to avoid circular imports
    from quantum_vqe_gpu import VQEConfig
    
    # Adjust parameters for smaller system
    max_iter_scale = min(2.0, 30 / reduced_system_size)  # More iterations for smaller systems
    
    reduced_config = VQEConfig(
        max_iterations=int(original_config.max_iterations * max_iter_scale),
        convergence_threshold=original_config.convergence_threshold,
        patience=max(20, original_config.patience // 2),  # Less patience for smaller systems
        optimizer=original_config.optimizer,
        learning_rate=original_config.learning_rate * 1.5,  # Higher learning rate for smaller systems
        n_layers=max(2, min(4, reduced_system_size // 4)),  # Adaptive layer count
        shots=original_config.shots,
        backend=original_config.backend,
        device_name=original_config.device_name,
        save_results=original_config.save_results,
        plot_convergence=original_config.plot_convergence,
        calculate_excited_states=reduced_system_size <= 15,  # Only for small enough systems
        n_excited_states=min(2, max(1, reduced_system_size // 6))  # Adaptive excited states
    )
    
    logger.info(f"Created reduced VQE config: {reduced_config.n_layers} layers, {reduced_config.max_iterations} iterations")
    return reduced_config
