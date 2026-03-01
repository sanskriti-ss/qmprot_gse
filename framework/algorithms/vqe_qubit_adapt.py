"""
QUBIT-ADAPT-VQE Implementation

Adaptive Derivative-Assembled Pseudo-Trotter VQE using a Pauli Operator Pool.
Grows the ansatz adaptively based on gradient information of Pauli strings.
"""
import numpy as np
from typing import Optional, Any, List, Tuple
import logging

import sys
sys.path.append('..')
from core.base_vqe import BaseVQE, VQEResult
from core.hamiltonian_loader import QubitHamiltonian

logger = logging.getLogger(__name__)


class QubitAdaptVQE(BaseVQE):
    """
    Qubit-ADAPT-VQE implementation.
    
    Adaptively grows the ansatz by selecting Pauli operators directly 
    from a pool based on their gradient magnitudes.
    """
    
    def __init__(self,
                 hamiltonian: QubitHamiltonian,
                 max_operators: int = 20,
                 gradient_threshold: float = 1e-4,
                 use_restricted_pool: bool = True,
                 **kwargs):
        """
        Initialize Qubit-ADAPT-VQE.
        
        Args:
            hamiltonian: QubitHamiltonian object
            max_operators: Maximum number of operators to add
            gradient_threshold: Minimum gradient to add an operator
            use_restricted_pool: If False (default), generate all odd-Y Pauli strings
                           over all qubit combinations. If True, restrict to
                           occupied->virtual excitations based on n_electrons
                           (smaller pool, faster gradient screening).
            **kwargs: Additional arguments passed to BaseVQE
        """
        super().__init__(hamiltonian, **kwargs)
        
        self.name = "qubit_adapt_vqe"
        self.description = "Qubit-Adaptive Derivative-Assembled Pseudo-Trotter VQE"
        self.max_operators = max_operators
        self.gradient_threshold = gradient_threshold
        self.use_restricted_pool = use_restricted_pool

        # Operator pool is now a list of Pauli strings (e.g., "Y0 X1")
        self.operator_pool: List[str] = []
        self.selected_operators: List[int] = []
        self.parameters: np.ndarray = np.array([])
        
        # Will be set in build_ansatz
        self.device = None
        
    def _generate_full_pool(self, n_qubits: int) -> List[str]:
        """
        Generate the full Pauli operator pool using all qubit combinations.
        
        Includes all odd-Y Pauli strings for every pair (singles) and
        every quartet (doubles) of qubits, regardless of occupation.
        Larger pool but more thorough exploration.
        """
        pool = []
        
        # Singles (2-qubit) - all odd-Y combinations
        for i in range(n_qubits):
            for j in range(i + 1, n_qubits):
                pool.append(f"Y{i} X{j}")
                pool.append(f"X{i} Y{j}")
        
        # Doubles (4-qubit) - all odd-Y permutations
        if n_qubits >= 4:
            for i in range(n_qubits):
                for j in range(i + 1, n_qubits):
                    for k in range(j + 1, n_qubits):
                        for l in range(k + 1, n_qubits):
                            # 1-Y terms
                            pool.append(f"Y{i} X{j} X{k} X{l}")
                            pool.append(f"X{i} Y{j} X{k} X{l}")
                            pool.append(f"X{i} X{j} Y{k} X{l}")
                            pool.append(f"X{i} X{j} X{k} Y{l}")
                            # 3-Y terms
                            pool.append(f"Y{i} Y{j} Y{k} X{l}")
                            pool.append(f"Y{i} Y{j} X{k} Y{l}")
                            pool.append(f"Y{i} X{j} Y{k} Y{l}")
                            pool.append(f"X{i} Y{j} Y{k} Y{l}")

        return pool

    def _generate_restricted_pool(self, n_qubits: int, n_electrons: int) -> List[str]:
        """
        Generate a restricted Pauli operator pool using occupied/virtual partitioning.
        
        Only generates excitations from occupied orbitals (0..n_electrons-1)
        to virtual orbitals (n_electrons..n_qubits-1). Smaller pool for faster
        gradient screening, physically motivated by Hartree-Fock reference.
        """
        pool = []

        occ_indices = range(n_electrons)
        vir_indices = range(n_electrons, n_qubits)

        # Singles: one occupied -> one virtual
        for p in occ_indices:
            for q in vir_indices:
                pool.append(f"Y{p} X{q}")
                pool.append(f"X{p} Y{q}")

        # Doubles: two occupied -> two virtual
        if n_qubits >= 4:
            for p in occ_indices:
                for q in range(p + 1, n_electrons):
                    for r in vir_indices:
                        for s in range(r + 1, n_qubits):
                            wires = sorted([p, q, r, s])
                            a, b, c, d = wires

                            # 1-Y terms
                            pool.append(f"Y{a} X{b} X{c} X{d}")
                            pool.append(f"X{a} Y{b} X{c} X{d}")
                            pool.append(f"X{a} X{b} Y{c} X{d}")
                            pool.append(f"X{a} X{b} X{c} Y{d}")
                            
                            # 3-Y terms
                            # pool.append(f"Y{a} Y{b} Y{c} X{d}")
                            # pool.append(f"Y{a} Y{b} X{c} Y{d}")
                            # pool.append(f"Y{a} X{b} Y{c} Y{d}")
                            # pool.append(f"X{a} Y{b} Y{c} Y{d}")

        return pool

    def build_ansatz(self) -> Any:
        """Build the Pauli operator pool"""
        from core.backend_manager import create_device
        
        n_qubits = self.n_qubits
        
        # Create device via backend manager
        self.device = create_device(self.backend_config)
        
        # Generate Pauli Pool
        if not self.use_restricted_pool:
            self.operator_pool = self._generate_full_pool(n_qubits)
            logger.info(f"Built FULL Pauli operator pool with {len(self.operator_pool)} operators")
        else:
            n_electrons = self.hamiltonian.molecule.n_electrons
            if n_electrons is None or n_electrons >= n_qubits:
                n_electrons = n_qubits // 2
            n_electrons = max(n_electrons, 1)
            self.operator_pool = self._generate_restricted_pool(n_qubits, n_electrons)
            logger.info(f"Built RESTRICTED Pauli operator pool with {len(self.operator_pool)} operators "
                       f"(n_electrons={n_electrons}, n_virtual={n_qubits - n_electrons})")
        
        # Initialize with empty ansatz
        self.selected_operators = []
        self.parameters = np.array([])
        self.n_parameters = 0
        
        return self.operator_pool
    
    def _build_circuit(self, params: np.ndarray):
        """Build circuit with currently selected Pauli operators"""
        import pennylane as qml
        
        n_qubits = self.n_qubits
        insert_noise = self.noise_inserter

        def parse_pauli_string(pauli_str):
            """
            Converts 'Y0 X1' -> ('YX', [0, 1])
            """
            # Split "Y0 X1" into ["Y0", "X1"]
            terms = pauli_str.split()
            
            pauli_word = ""
            wires = []
            
            for term in terms:
                # The first character is the Pauli (X, Y, Z)
                pauli_char = term[0]
                # The rest is the wire index
                wire_idx = int(term[1:])
                
                pauli_word += pauli_char
                wires.append(wire_idx)
                
            return pauli_word, wires
        
        @qml.qnode(self.device)
        def circuit(params):
            # Initial state (Hartree-Fock)
            n_electrons = min(self.hamiltonian.molecule.n_electrons or n_qubits // 2, n_qubits)
            for i in range(n_electrons):
                qml.PauliX(wires=i)
            
            # Apply selected operators
            for idx, op_idx in enumerate(self.selected_operators):
                pauli_string_raw = self.operator_pool[op_idx]
                
                # PARSING STEP
                pauli_word, target_wires = parse_pauli_string(pauli_string_raw)

                theta = params[idx] if idx < len(params) else 0.0
                
                # Apply Pauli Rotation
                # U = exp(-i * theta * P). PennyLane implements exp(-i * phi/2 * P).
                # So we pass phi = 2 * theta.
                qml.PauliRot(2 * theta, pauli_word, wires=target_wires)

                # Apply noise
                insert_noise()
            
            H = self.hamiltonian.to_pennylane()
            return qml.expval(H)
        
        return circuit
    
    def cost_function(self, parameters: np.ndarray) -> float:
        """Evaluate the cost function"""
        circuit = self._build_circuit(parameters)
        return float(circuit(parameters))
    
    def _prescreen_pool(self) -> List[int]:
        """
        Pre-screen the operator pool against the Hamiltonian to find operators
        that could have non-zero gradient from a computational basis state.
        
        An operator P can only produce a non-zero gradient if some Hamiltonian
        term flips the exact same qubits (X/Y positions match). This is exact
        for the first ADAPT iteration (HF state) and a useful heuristic for
        later iterations.
        """
        h_flip_patterns = set()
        for pauli_str in self.hamiltonian.pauli_strings:
            flips = frozenset(i for i, c in enumerate(pauli_str) if c in ('X', 'Y'))
            if flips:
                h_flip_patterns.add(flips)

        viable = []
        for op_idx, op_str in enumerate(self.operator_pool):
            terms = op_str.split()
            op_flips = frozenset(int(t[1:]) for t in terms)
            if op_flips in h_flip_patterns:
                viable.append(op_idx)

        logger.info(f"Pre-screening: {len(viable)}/{len(self.operator_pool)} operators viable "
                    f"({len(self.operator_pool) - len(viable)} skipped, "
                    f"{len(h_flip_patterns)} unique off-diagonal patterns in H)")
        return viable

    def _compute_gradients(self) -> np.ndarray:
        """Compute gradients for all operators in the pool via Finite Difference"""
        gradients = np.zeros(len(self.operator_pool))
        delta = 1e-5
        
        n_qubits = self.n_qubits
        n_elec = min(self.hamiltonian.molecule.n_electrons or n_qubits // 2, n_qubits // 2)
        logger.info(f"Gradient computation: n_qubits={n_qubits}, n_electrons_used={n_elec}, "
                     f"HF state={'|' + '1'*n_elec + '0'*(n_qubits-n_elec) + '>'}, "
                     f"pool_size={len(self.operator_pool)}")
        
        viable_ops = self._prescreen_pool()
        
        for op_idx in viable_ops:
            if op_idx in self.selected_operators:
                continue
            
            # Temporarily add operator at the END of the ansatz
            self.selected_operators.append(op_idx)
            test_params = np.append(self.parameters, delta)
            
            # Compute gradient via finite difference at theta=0
            energy_plus = self.cost_function(test_params)
            
            test_params[-1] = -delta
            energy_minus = self.cost_function(test_params)
            
            gradient = (energy_plus - energy_minus) / (2 * delta)
            gradients[op_idx] = abs(gradient)

            self.selected_operators.pop()
        
        return gradients
    
    def run(self) -> VQEResult:
        """Run Qubit-ADAPT-VQE"""
        import time
        from scipy.optimize import minimize
        
        logger.info(f"Running {self.name} on {self.hamiltonian.molecule.name}")
        start_time = time.time()
        
        # HF verification
        # self._perform_hf_verification()
        
        # Build pool
        self.build_ansatz()
        self.convergence_history = []
        
        # ADAPT loop
        for iteration in range(self.max_operators):
            # 1. Compute Gradients
            gradients = self._compute_gradients()
            max_grad_idx = np.argmax(gradients)
            max_grad = gradients[max_grad_idx]
            
            logger.debug(f"Iter {iteration}: Max grad {max_grad:.6f} for op {self.operator_pool[max_grad_idx]}")
            
            # 2. Check Convergence
            if max_grad < self.gradient_threshold:
                logger.info(f"Converged at iteration {iteration} (gradient {max_grad:.2e} < {self.gradient_threshold})")
                break
            
            # 3. Add Operator
            self.selected_operators.append(max_grad_idx)
            self.parameters = np.append(self.parameters, 0.0)
            self.n_parameters = len(self.parameters)
            
            # 4. Re-optimize all parameters
            if len(self.parameters) > 0:
                result = minimize(
                    self.cost_function,
                    self.parameters,
                    method=self.optimizer_name,
                    options={"maxiter": 100}
                )
                self.parameters = result.x
                current_energy = result.fun
            else:
                current_energy = self.cost_function(self.parameters)
            
            self.convergence_history.append(current_energy)
            logger.info(f"Iter {iteration}: Energy = {current_energy:.8f}, Added {self.operator_pool[max_grad_idx]}")
        
        runtime = time.time() - start_time
        
        # Calculate final stats
        optimal_energy = self.cost_function(self.parameters)
        ref_energy = self.hamiltonian.molecule.reference_energy
        error = optimal_energy - ref_energy
        
        result = VQEResult(
            molecule_abbrev=self.hamiltonian.molecule.abbreviation,
            molecule_name=self.hamiltonian.molecule.name,
            algorithm_name=self.name,
            calculated_energy=optimal_energy,
            reference_energy=ref_energy,
            error=error,
            relative_error=abs(error / ref_energy) if ref_energy else 0.0,
            n_iterations=len(self.selected_operators),
            n_qubits=self.n_qubits,
            n_parameters=self.n_parameters,
            runtime_seconds=runtime,
            convergence_history=self.convergence_history,
            optimal_parameters=self.parameters,
            converged=True,
            metadata={
                "optimizer": self.optimizer_name,
                "n_operators_selected": len(self.selected_operators),
                "selected_operators_str": [self.operator_pool[i] for i in self.selected_operators]
            },
            backend_type=self.backend_config.backend_type,
            noise_model=self.backend_config.noise_model,
            noise_strength=self.backend_config.noise_strength,
            hf_energy=self.hf_energy,
        )
        
        return result