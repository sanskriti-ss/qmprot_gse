"""
Excited States Module - First Excited State via VQD

Implements Variational Quantum Deflation (VQD) for calculating
the first excited state of molecular Hamiltonians from QMProt.

Based on: Higgott et al., "Variational Quantum Computation of Excited States" (2019)
"""
import numpy as np
import pennylane as qml
from typing import List, Optional, Tuple, Any, Dict
from dataclasses import dataclass, field
import time
import logging
from tqdm import tqdm

from .base_vqe import BaseVQE, VQEResult
from .hamiltonian_loader import QubitHamiltonian

logger = logging.getLogger(__name__)


@dataclass
class ExcitedStateResult:
    """Data class for excited state calculation results"""
    state_index: int  # 0 = ground, 1 = first excited, etc.
    energy: float
    gap_from_ground: Optional[float]
    optimal_parameters: np.ndarray
    convergence_history: List[float] = field(default_factory=list)
    n_iterations: int = 0
    runtime_seconds: float = 0.0
    converged: bool = False
    
    def to_dict(self) -> Dict:
        """Convert to dictionary for JSON serialization"""
        return {
            "state_index": self.state_index,
            "energy": float(self.energy),
            "gap_from_ground": float(self.gap_from_ground) if self.gap_from_ground else None,
            "optimal_parameters": self.optimal_parameters.tolist(),
            "convergence_history": [float(e) for e in self.convergence_history],
            "n_iterations": self.n_iterations,
            "runtime_seconds": float(self.runtime_seconds),
            "converged": self.converged,
        }


@dataclass 
class VQDResult:
    """Combined results for VQD calculation (ground + excited states)"""
    molecule_abbrev: str
    molecule_name: str
    n_qubits: int
    n_parameters: int
    ground_state: ExcitedStateResult
    excited_states: List[ExcitedStateResult]
    total_runtime_seconds: float
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    def to_dict(self) -> Dict:
        return {
            "molecule_abbrev": self.molecule_abbrev,
            "molecule_name": self.molecule_name,
            "n_qubits": self.n_qubits,
            "n_parameters": self.n_parameters,
            "ground_state": self.ground_state.to_dict(),
            "excited_states": [es.to_dict() for es in self.excited_states],
            "total_runtime_seconds": float(self.total_runtime_seconds),
            "metadata": self.metadata,
        }
    
    @property
    def first_excited_energy(self) -> Optional[float]:
        """Get the first excited state energy"""
        if self.excited_states:
            return self.excited_states[0].energy
        return None
    
    @property
    def first_gap(self) -> Optional[float]:
        """Get the first excitation gap (E1 - E0)"""
        if self.excited_states:
            return self.excited_states[0].gap_from_ground
        return None


class VQD(BaseVQE):
    """
    Variational Quantum Deflation (VQD) for excited states.
    
    Finds excited states by adding penalty terms to the cost function
    that penalize overlap with previously found states.
    
    Cost function for k-th state:
        L_k(θ) = ⟨ψ(θ)|H|ψ(θ)⟩ + Σᵢ βᵢ|⟨ψᵢ|ψ(θ)⟩|²
    
    where ψᵢ are previously found eigenstates and βᵢ are penalty coefficients.
    
    Args:
        hamiltonian: QubitHamiltonian object
        n_states: Total number of states to find (1 = ground only, 2 = ground + 1st excited)
        beta: Penalty coefficient (should be larger than expected energy gap)
        n_layers: Number of variational ansatz layers
        **kwargs: Additional arguments passed to BaseVQE
    
    Example:
        >>> from core.hamiltonian_loader import HamiltonianLoader
        >>> loader = HamiltonianLoader()
        >>> H = loader.load("ala")
        >>> vqd = VQD(H, n_states=2, beta=2.0)
        >>> results = vqd.run()
        >>> print(f"Ground state: {results.ground_state.energy:.6f}")
        >>> print(f"First excited: {results.first_excited_energy:.6f}")
        >>> print(f"Gap: {results.first_gap:.6f}")
    """
    
    def __init__(self,
                 hamiltonian: QubitHamiltonian,
                 n_states: int = 2,
                 beta: float = 2.0,
                 n_layers: int = 2,
                 **kwargs):
        super().__init__(hamiltonian, **kwargs)
        
        self.name = "vqd"
        self.description = "Variational Quantum Deflation"
        self.n_states = n_states
        self.beta = beta
        self.n_layers = n_layers
        
        # Storage for found states (parameters and energies)
        self.found_states: List[Tuple[np.ndarray, float]] = []
        
        # Circuit functions (set in build_ansatz)
        self.energy_circuit = None
        self.overlap_circuit = None
        self.ansatz_fn = None
        self.device = None
        
        # Current state being optimized
        self._current_state_idx = 0
        
    def build_ansatz(self) -> Any:
        """
        Build the variational ansatz and overlap measurement circuits.
        
        Returns:
            The energy measurement circuit (QNode)
        """
        n_qubits = self.n_qubits
        n_layers = self.n_layers
        
        # Parameters: 3 rotation angles per qubit per layer
        self.n_parameters = n_qubits * 3 * n_layers
        
        # Create PennyLane device
        self.device = qml.device("lightning.qubit", wires=n_qubits)
        
        # Get Hamiltonian in PennyLane format
        H = self.hamiltonian.to_pennylane()
        
        def apply_ansatz(params, wires):
            """
            Apply the variational ansatz circuit.
            
            Uses a hardware-efficient ansatz with:
            - Hartree-Fock initial state
            - Layers of single-qubit rotations (RX, RY, RZ)
            - Entangling CNOT layers
            """
            params = params.reshape(n_layers, n_qubits, 3)
            
            # Hartree-Fock initial state preparation
            n_electrons = self.hamiltonian.molecule.n_electrons or n_qubits // 2
            for i in range(min(n_electrons, n_qubits)):
                qml.PauliX(wires=i)
            
            # Variational layers
            for layer in range(n_layers):
                # Single-qubit rotations
                for qubit in range(n_qubits):
                    qml.RX(params[layer, qubit, 0], wires=qubit)
                    qml.RY(params[layer, qubit, 1], wires=qubit)
                    qml.RZ(params[layer, qubit, 2], wires=qubit)
                
                # Entangling layer (nearest-neighbor CNOTs)
                for i in range(0, n_qubits - 1, 2):
                    qml.CNOT(wires=[i, i + 1])
                for i in range(1, n_qubits - 1, 2):
                    qml.CNOT(wires=[i, i + 1])
        
        self.ansatz_fn = apply_ansatz
        
        @qml.qnode(self.device)
        def energy_circuit(params):
            """Measure energy expectation value ⟨ψ(θ)|H|ψ(θ)⟩"""
            apply_ansatz(params, range(n_qubits))
            return qml.expval(H)
        
        @qml.qnode(self.device)
        def overlap_circuit(params_new, params_old):
            """
            Measure overlap |⟨ψ_old|ψ_new⟩|² using compute-uncompute method.
            
            Circuit:
                |0⟩ → U_new(θ_new) → U†_old(θ_old) → measure
                
            The probability of measuring |0...0⟩ equals |⟨ψ_old|ψ_new⟩|²
            """
            # Prepare |ψ_new⟩ = U_new|0⟩
            apply_ansatz(params_new, range(n_qubits))
            # Apply U†_old (adjoint of old state preparation)
            qml.adjoint(lambda: apply_ansatz(params_old, range(n_qubits)))()
            # Return probabilities
            return qml.probs(wires=range(n_qubits))
        
        self.energy_circuit = energy_circuit
        self.overlap_circuit = overlap_circuit
        
        logger.info(f"Built VQD ansatz: {self.n_parameters} parameters, {n_layers} layers, {n_qubits} qubits")
        
        return energy_circuit
    
    def cost_function(self, parameters: np.ndarray) -> float:
        """
        VQD cost function with deflation penalties.
        
        For the k-th state:
            L_k(θ) = ⟨ψ(θ)|H|ψ(θ)⟩ + Σᵢ βᵢ|⟨ψᵢ|ψ(θ)⟩|²
        
        Args:
            parameters: Current variational parameters
            
        Returns:
            Cost value (energy + penalty)
        """
        if self.energy_circuit is None:
            raise RuntimeError("Must call build_ansatz() first")
        
        # Base energy expectation value
        energy = float(self.energy_circuit(parameters))
        
        # Add penalty terms for overlap with previously found states
        penalty = 0.0
        for prev_params, prev_energy in self.found_states:
            probs = self.overlap_circuit(parameters, prev_params)
            overlap_squared = float(probs[0])  # |⟨ψ_old|ψ_new⟩|²
            penalty += self.beta * overlap_squared
        
        return energy + penalty
    
    def get_energy(self, parameters: np.ndarray) -> float:
        """
        Get the actual energy (without penalty) for given parameters.
        
        Args:
            parameters: Variational parameters
            
        Returns:
            Energy expectation value
        """
        if self.energy_circuit is None:
            raise RuntimeError("Must call build_ansatz() first")
        return float(self.energy_circuit(parameters))
    
    def get_overlap(self, params1: np.ndarray, params2: np.ndarray) -> float:
        """
        Compute overlap squared |⟨ψ1|ψ2⟩|² between two states.
        
        Args:
            params1: Parameters for first state
            params2: Parameters for second state
            
        Returns:
            Overlap squared
        """
        if self.overlap_circuit is None:
            raise RuntimeError("Must call build_ansatz() first")
        probs = self.overlap_circuit(params1, params2)
        return float(probs[0])
    
    def run_single_state(self, state_index: int) -> ExcitedStateResult:
        """
        Run VQD optimization for a single state.
        
        Args:
            state_index: Index of state to find (0=ground, 1=first excited, ...)
            
        Returns:
            ExcitedStateResult for this state
        """
        self._current_state_idx = state_index
        state_name = "ground state" if state_index == 0 else f"excited state {state_index}"
        
        logger.info(f"Finding {state_name}...")
        
        # Reset tracking for this state
        self.convergence_history = []
        self.iteration_count = 0
        
        start_time = time.time()
        
        # Get initial parameters
        initial_params = self.get_initial_parameters()
        
        # Run optimization
        optimal_params, optimal_cost = self.optimize(initial_params)
        
        runtime = time.time() - start_time
        
        # Get the actual energy (without penalty)
        actual_energy = self.get_energy(optimal_params)
        
        # Calculate gap from ground state
        gap_from_ground = None
        if state_index > 0 and self.found_states:
            ground_energy = self.found_states[0][1]
            gap_from_ground = actual_energy - ground_energy
        
        # Check convergence
        converged = len(self.convergence_history) > 1 and \
                   abs(self.convergence_history[-1] - self.convergence_history[-2]) < self.convergence_threshold
        
        result = ExcitedStateResult(
            state_index=state_index,
            energy=actual_energy,
            gap_from_ground=gap_from_ground,
            optimal_parameters=optimal_params.copy(),
            convergence_history=self.convergence_history.copy(),
            n_iterations=self.iteration_count,
            runtime_seconds=runtime,
            converged=converged,
        )
        
        # Store state for deflation of subsequent states
        self.found_states.append((optimal_params.copy(), actual_energy))
        
        # Log overlaps with previous states (for verification)
        if state_index > 0:
            for i, (prev_params, _) in enumerate(self.found_states[:-1]):
                overlap = self.get_overlap(optimal_params, prev_params)
                logger.info(f"  Overlap with state {i}: {overlap:.6f}")
        
        logger.info(f"  Energy: {actual_energy:.8f} Hartree, Runtime: {runtime:.2f}s")
        if gap_from_ground is not None:
            logger.info(f"  Gap from ground state: {gap_from_ground:.6f} Hartree")
        
        return result
    
    def run(self) -> VQDResult:
        """
        Run complete VQD calculation for ground and excited states.
        
        Returns:
            VQDResult containing all state results
        """
        logger.info(f"Running VQD on {self.hamiltonian.molecule.name}")
        logger.info(f"  Finding {self.n_states} states (ground + {self.n_states-1} excited)")
        logger.info(f"  Penalty coefficient β = {self.beta}")
        
        total_start = time.time()
        
        # Build ansatz
        self.build_ansatz()
        
        # Reset found states
        self.found_states = []
        
        # Find all states sequentially
        all_results = []
        for state_idx in range(self.n_states):
            result = self.run_single_state(state_idx)
            all_results.append(result)
        
        total_runtime = time.time() - total_start
        
        # Separate ground and excited states
        ground_state = all_results[0]
        excited_states = all_results[1:] if len(all_results) > 1 else []
        
        vqd_result = VQDResult(
            molecule_abbrev=self.hamiltonian.molecule.abbreviation,
            molecule_name=self.hamiltonian.molecule.name,
            n_qubits=self.n_qubits,
            n_parameters=self.n_parameters,
            ground_state=ground_state,
            excited_states=excited_states,
            total_runtime_seconds=total_runtime,
            metadata={
                "algorithm": "VQD",
                "n_states": self.n_states,
                "beta": self.beta,
                "n_layers": self.n_layers,
                "optimizer": self.optimizer_name,
                "max_iterations": self.max_iterations,
            }
        )
        
        # Summary log
        logger.info(f"\nVQD Results for {self.hamiltonian.molecule.name}:")
        logger.info(f"  Ground state energy: {ground_state.energy:.8f} Hartree")
        for i, es in enumerate(excited_states):
            logger.info(f"  Excited state {i+1}: {es.energy:.8f} Hartree (gap: {es.gap_from_ground:.6f})")
        logger.info(f"  Total runtime: {total_runtime:.2f}s")
        
        return vqd_result


class SSVQE(BaseVQE):
    """
    Subspace-Search VQE (SSVQE) for simultaneous excited states.
    
    Finds multiple eigenstates in a single optimization by minimizing
    a weighted sum of energies for different reference states.
    
    Cost function:
        L(θ) = Σᵢ wᵢ ⟨φᵢ|U†(θ) H U(θ)|φᵢ⟩
    
    where {|φᵢ⟩} are orthonormal reference states and wᵢ are weights.
    
    Args:
        hamiltonian: QubitHamiltonian object
        n_states: Number of states to find simultaneously
        weights: List of weights for each state (default: descending weights)
        n_layers: Number of variational ansatz layers
        **kwargs: Additional arguments passed to BaseVQE
    """
    
    def __init__(self,
                 hamiltonian: QubitHamiltonian,
                 n_states: int = 2,
                 weights: Optional[List[float]] = None,
                 n_layers: int = 2,
                 **kwargs):
        super().__init__(hamiltonian, **kwargs)
        
        self.name = "ssvqe"
        self.description = "Subspace-Search VQE"
        self.n_states = n_states
        self.n_layers = n_layers
        
        # Default weights: descending (prioritize lower energy states)
        if weights is None:
            self.weights = [1.0 / (i + 1) for i in range(n_states)]
        else:
            self.weights = weights
        
        # Normalize weights
        total = sum(self.weights)
        self.weights = [w / total for w in self.weights]
        
        self.device = None
        self.state_circuits = []
        
    def build_ansatz(self) -> Any:
        """Build ansatz circuits for each reference state"""
        n_qubits = self.n_qubits
        n_layers = self.n_layers
        
        self.n_parameters = n_qubits * 3 * n_layers
        self.device = qml.device("lightning.qubit", wires=n_qubits)
        H = self.hamiltonian.to_pennylane()
        
        def apply_ansatz(params, wires):
            """Apply variational ansatz"""
            params = params.reshape(n_layers, n_qubits, 3)
            for layer in range(n_layers):
                for qubit in range(n_qubits):
                    qml.RX(params[layer, qubit, 0], wires=qubit)
                    qml.RY(params[layer, qubit, 1], wires=qubit)
                    qml.RZ(params[layer, qubit, 2], wires=qubit)
                for i in range(0, n_qubits - 1, 2):
                    qml.CNOT(wires=[i, i + 1])
                for i in range(1, n_qubits - 1, 2):
                    qml.CNOT(wires=[i, i + 1])
        
        self.state_circuits = []
        
        for ref_idx in range(self.n_states):
            @qml.qnode(self.device)
            def state_circuit(params, ref_idx=ref_idx):
                """Circuit for reference state |ref_idx⟩"""
                # Prepare reference state (computational basis)
                # Use binary representation of ref_idx
                for qubit in range(n_qubits):
                    if (ref_idx >> qubit) & 1:
                        qml.PauliX(wires=qubit)
                
                apply_ansatz(params, range(n_qubits))
                return qml.expval(H)
            
            self.state_circuits.append(state_circuit)
        
        logger.info(f"Built SSVQE ansatz: {self.n_parameters} parameters, {self.n_states} states")
        return self.state_circuits[0]
    
    def cost_function(self, parameters: np.ndarray) -> float:
        """SSVQE cost: weighted sum of energies for all reference states"""
        total = 0.0
        for i, (w, circuit) in enumerate(zip(self.weights, self.state_circuits)):
            energy = float(circuit(parameters))
            total += w * energy
        return total
    
    def get_state_energies(self, parameters: np.ndarray) -> List[float]:
        """Get individual state energies from optimized parameters"""
        return [float(circuit(parameters)) for circuit in self.state_circuits]
    
    def run(self) -> VQDResult:
        """Run SSVQE optimization"""
        logger.info(f"Running SSVQE on {self.hamiltonian.molecule.name}")
        logger.info(f"  Finding {self.n_states} states simultaneously")
        logger.info(f"  Weights: {self.weights}")
        
        total_start = time.time()
        
        self.build_ansatz()
        self.convergence_history = []
        self.iteration_count = 0
        
        initial_params = self.get_initial_parameters()
        optimal_params, _ = self.optimize(initial_params)
        
        total_runtime = time.time() - total_start
        
        # Extract individual energies
        energies = self.get_state_energies(optimal_params)
        
        # Create results
        ground_state = ExcitedStateResult(
            state_index=0,
            energy=energies[0],
            gap_from_ground=None,
            optimal_parameters=optimal_params,
            convergence_history=self.convergence_history,
            n_iterations=self.iteration_count,
            runtime_seconds=total_runtime,
        )
        
        excited_states = []
        for i, energy in enumerate(energies[1:], 1):
            excited_states.append(ExcitedStateResult(
                state_index=i,
                energy=energy,
                gap_from_ground=energy - energies[0],
                optimal_parameters=optimal_params,
                convergence_history=[],
                n_iterations=0,
                runtime_seconds=0,
            ))
        
        result = VQDResult(
            molecule_abbrev=self.hamiltonian.molecule.abbreviation,
            molecule_name=self.hamiltonian.molecule.name,
            n_qubits=self.n_qubits,
            n_parameters=self.n_parameters,
            ground_state=ground_state,
            excited_states=excited_states,
            total_runtime_seconds=total_runtime,
            metadata={
                "algorithm": "SSVQE",
                "n_states": self.n_states,
                "weights": self.weights,
                "n_layers": self.n_layers,
                "optimizer": self.optimizer_name,
            }
        )
        
        logger.info(f"\nSSVQE Results:")
        for i, e in enumerate(energies):
            gap = f" (gap: {e - energies[0]:.6f})" if i > 0 else ""
            logger.info(f"  State {i}: {e:.8f} Hartree{gap}")
        
        return result


# Convenience function for quick first excited state calculation
def compute_first_excited_state(hamiltonian: QubitHamiltonian,
                                 method: str = "vqd",
                                 **kwargs) -> VQDResult:
    """
    Convenience function to compute ground and first excited state.
    
    Args:
        hamiltonian: QubitHamiltonian object
        method: Either "vqd" or "ssvqe"
        **kwargs: Additional arguments passed to the algorithm
        
    Returns:
        VQDResult with ground and first excited state
    """
    if method.lower() == "vqd":
        solver = VQD(hamiltonian, n_states=2, **kwargs)
    elif method.lower() == "ssvqe":
        solver = SSVQE(hamiltonian, n_states=2, **kwargs)
    else:
        raise ValueError(f"Unknown method: {method}. Use 'vqd' or 'ssvqe'")
    
    return solver.run()
