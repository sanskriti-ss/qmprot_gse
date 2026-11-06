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
from core.base_vqe import (
    BaseVQE,
    EvaluationBudgetExhausted,
    VQEResult,
    incumbent_history,
    inner_solve_iteration_cap,
)
from core.hamiltonian_loader import QubitHamiltonian

logger = logging.getLogger(__name__)


def _pauli_string_from_pool_label(label: str, n_qubits: int) -> str:
    """Expand ``'Y0 X1'`` to a length-``n_qubits`` string, index 0 on wire 0."""
    chars = ["I"] * n_qubits
    for term in label.split():
        chars[int(term[1:])] = term[0]
    return "".join(chars)


def _paulis_commute(left: str, right: str) -> bool:
    """Two Pauli strings commute when they anticommute on an even number of wires."""
    anticommuting_sites = 0
    for left_op, right_op in zip(left, right):
        if left_op != "I" and right_op != "I" and left_op != right_op:
            anticommuting_sites += 1
    return anticommuting_sites % 2 == 0


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
                 max_gradient_candidates: Optional[int] = None,
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
            max_gradient_candidates: If set, evaluate gradients on at most this
                           many viable pool operators per ADAPT iteration
                           (random subset, deterministic under random_seed).
            **kwargs: Additional arguments passed to BaseVQE
        """
        super().__init__(hamiltonian, **kwargs)
        
        self.name = "qubit_adapt_vqe"
        self.description = "Qubit-Adaptive Derivative-Assembled Pseudo-Trotter VQE"
        self.max_operators = max_operators
        self.gradient_threshold = gradient_threshold
        self.use_restricted_pool = use_restricted_pool
        self.max_gradient_candidates = (
            int(max_gradient_candidates)
            if max_gradient_candidates is not None and int(max_gradient_candidates) > 0
            else None
        )

        # Operator pool is now a list of Pauli strings (e.g., "Y0 X1")
        self.operator_pool: List[str] = []
        self.selected_operators: List[int] = []
        self.parameters: np.ndarray = np.array([])
        # Operators that fail to commute with H.  Exact on every state, so
        # it can be cached.  Gradients of this set are recomputed each iteration.
        self._noncommuting_ops: Optional[List[int]] = None
        self._rng = np.random.default_rng(kwargs.get("random_seed", None))
        
        # Will be set in build_ansatz
        self.device = None
        self.cost_fn = None
        self._cost_fn_signature: Optional[Tuple[int, ...]] = None
        
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
        self.cost_fn = None
        self._cost_fn_signature = None
        
        # Generate Pauli Pool
        # Use full pool when CS reduction is active (no meaningful occupied/virtual split)
        cs_active = getattr(self.hamiltonian, "cs_initial_state", None) is not None
        if not self.use_restricted_pool or cs_active:
            if cs_active:
                logger.info("CS reduction active; using FULL pool (occupied/virtual split not applicable)")
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
        self._noncommuting_ops = None
        
        return self.operator_pool
    
    def _build_circuit(self):
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
            # Initial state preparation
            self._prepare_initial_state()
            
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
        signature = tuple(self.selected_operators)
        if self.cost_fn is None or self._cost_fn_signature != signature:
            self.cost_fn = self._build_circuit()
            self._cost_fn_signature = signature
        return float(self.cost_fn(parameters))
    
    def _noncommuting_pool_indices(self) -> List[int]:
        """Pool operators whose commutator with H is not identically zero.

        ``[H, P] = 0`` means the gradient of ``exp(-i θ P)`` vanishes on
        every state, so those operators can be dropped for the whole run.
        Matching X/Y supports against the Hartree--Fock determinant is a
        stronger screen and is exact only on that first state.  It is not
        used here: an operator that is flat on Hartree--Fock can be the
        largest gradient after the first accepted rotation.
        """
        h_terms = [
            pauli for pauli in self.hamiltonian.pauli_strings
            if any(char != "I" for char in pauli)
        ]
        kept: List[int] = []
        for op_idx, op_str in enumerate(self.operator_pool):
            op_pauli = _pauli_string_from_pool_label(op_str, self.n_qubits)
            if any(not _paulis_commute(op_pauli, term) for term in h_terms):
                kept.append(op_idx)

        logger.info(
            "Commutator screen: %d/%d pool operators fail to commute with H "
            "(%d dropped)",
            len(kept),
            len(self.operator_pool),
            len(self.operator_pool) - len(kept),
        )
        return kept

    def _compute_gradients(self) -> np.ndarray:
        """Finite-difference gradients of the pool on the current ansatz.

        The difference uses step ``1e-5`` and the parameters already stored
        on the ansatz.  Already selected operators are left at zero so the
        same Pauli is not appended twice.  ``max_gradient_candidates``, when
        set, draws a random subset of this list; the default scores every
        operator that does not commute with H.
        """
        gradients = np.zeros(len(self.operator_pool))
        delta = 1e-5

        if self._noncommuting_ops is None:
            self._noncommuting_ops = self._noncommuting_pool_indices()

        viable_ops = [
            op_idx for op_idx in self._noncommuting_ops
            if op_idx not in self.selected_operators
        ]
        logger.info(
            "Gradient computation: scoring %d/%d non-commuting operators "
            "on the current %d-parameter ansatz",
            len(viable_ops),
            len(self._noncommuting_ops),
            len(self.parameters),
        )

        if (
            self.max_gradient_candidates is not None
            and len(viable_ops) > self.max_gradient_candidates
        ):
            n_total = len(viable_ops)
            viable_ops = self._rng.choice(
                viable_ops,
                size=self.max_gradient_candidates,
                replace=False,
            ).tolist()
            logger.info(
                "Gradient screening cap active: evaluating %d/%d viable operators",
                len(viable_ops),
                n_total,
            )
        
        for op_idx in viable_ops:
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
        
        logger.info(f"Running {self.name} on {self.hamiltonian.molecule.name}")
        start_time = time.time()
        
        # HF verification
        # self._perform_hf_verification()
        
        # Build pool
        self.build_ansatz()
        self.convergence_history = []
        
        # ADAPT loop.  Inner solves use BaseVQE.optimize so the configured
        # iteration budget and the Bayesian optimizer both apply.
        # ``update_history=False`` keeps one energy per accepted operator.
        # A shared evaluation budget, when the trainability counter is
        # attached, stops screening or the inner solve without a further
        # energy call.  An operator whose re-optimization did not return
        # is removed again: theta=0 would have left the state unchanged,
        # and the partial COBYLA point is not a finished solve.
        grad_converged = False
        budget_limited = None
        for iteration in range(self.max_operators):
            try:
                gradients = self._compute_gradients()
            except EvaluationBudgetExhausted:
                budget_limited = "gradient screening"
                logger.info("Evaluation budget exhausted during gradient screening")
                break
            max_grad_idx = np.argmax(gradients)
            max_grad = gradients[max_grad_idx]
            
            logger.debug(f"Iter {iteration}: Max grad {max_grad:.6f} for op {self.operator_pool[max_grad_idx]}")
            
            if max_grad < self.gradient_threshold:
                logger.info(f"Converged at iteration {iteration} (gradient {max_grad:.2e} < {self.gradient_threshold})")
                grad_converged = True
                break

            operators_left = self.max_operators - len(self.selected_operators)
            step_cap = inner_solve_iteration_cap(
                self, operators_left, self.max_iterations,
            )
            if step_cap < 1:
                budget_limited = "re-optimization"
                logger.info(
                    "Evaluation budget cannot give each of the %d remaining "
                    "operators an iteration",
                    operators_left,
                )
                break
            
            self.selected_operators.append(max_grad_idx)
            self.parameters = np.append(self.parameters, 0.0)
            self.n_parameters = len(self.parameters)

            saved_max = self.max_iterations
            self.max_iterations = step_cap
            try:
                self.parameters, current_energy = self.optimize(
                    initial_parameters=self.parameters,
                    update_history=False,
                )
            except EvaluationBudgetExhausted:
                self.selected_operators.pop()
                self.parameters = np.asarray(self.parameters[:-1], dtype=float)
                self.n_parameters = len(self.parameters)
                budget_limited = "re-optimization"
                logger.info("Evaluation budget exhausted during re-optimization")
                break
            finally:
                self.max_iterations = saved_max
            self.convergence_history.append(float(current_energy))
            logger.info(f"Iter {iteration}: Energy = {current_energy:.8f}, Added {self.operator_pool[max_grad_idx]}")
        
        runtime = time.time() - start_time
        
        # The last completed inner solve already returned this energy.
        # Calling the cost again would spend one evaluation past the budget.
        if self.convergence_history:
            optimal_energy = float(self.convergence_history[-1])
        else:
            try:
                optimal_energy = float(self.cost_function(self.parameters))
            except EvaluationBudgetExhausted:
                optimal_energy = float("nan")
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
            best_so_far_history=incumbent_history(self.convergence_history),
            optimal_parameters=self.parameters,
            converged=grad_converged,
            metadata={
                "optimizer": self.optimizer_name,
                "n_operators_selected": len(self.selected_operators),
                "selected_operators_str": [self.operator_pool[i] for i in self.selected_operators],
                "max_gradient_candidates": self.max_gradient_candidates,
                "evaluation_budget": budget_limited,
            },
            backend_type=self.backend_config.backend_type,
            noise_model=self.backend_config.noise_model,
            noise_strength=self.backend_config.noise_strength,
            hf_energy=self.hf_energy,
        )
        
        return result