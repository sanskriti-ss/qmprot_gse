# ABOUTME: Verifies exact energy reuse for successive analytic ADAPT screens and optimizer fits.
# ABOUTME: Exercises distinct state amplitudes, changing operator lists, resets and sampled measurements.
import unittest
from pathlib import Path

import numpy as np
import pennylane as qml

from algorithms.vqe_adapt import AdaptVQE
from algorithms.vqe_qubit_adapt import QubitAdaptVQE
from core.backend_manager import BackendConfig
from core.base_vqe import recent_state_energies
from experiments.provenance import load_problem


class AdaptiveEnergy(unittest.TestCase):
    def test_only_actual_state_bytes_identify_measured_states(self):
        cache = recent_state_energies(3)
        values = np.arange(8, dtype=float)
        states = np.eye(8, dtype=complex)
        calls = []
        for index in (0, 5, 7, 5, 1, 0):
            state = states[index]
            def expectation(observable):
                calls.append(index)
                return float(np.vdot(state, observable * state).real)
            self.assertEqual(cache(state, expectation, values), index)
        self.assertEqual(calls, [0, 5, 7, 1])
        state = states[0].copy()
        state[7] = 1e-21j
        self.assertEqual(cache(state, expectation, values), float(np.vdot(state, values * state).real))
        self.assertEqual(len(calls), 5)
        self.assertIsNone(recent_state_energies(20))

    def test_dynamic_pool_states_match_original_qnodes_bit_for_bit(self):
        source = Path(__file__).parent / "reproductions/layered_vqe/hydrogen.npz"
        problem = load_problem(source)
        for cls in (QubitAdaptVQE, AdaptVQE):
            solver = cls(problem.hamiltonian)
            for trial in range(2):
                solver.build_ansatz()
                for indices in ([0], [1], [2], [0, 2], [], [0]):
                    solver.selected_operators = list(indices)
                    for sign in (1., -1.):
                        angles = sign*np.array([.1, 1e-5][:len(indices)])
                        ordinary = solver._build_circuit() if cls is QubitAdaptVQE else solver._build_circuit(angles)
                        self.assertEqual(solver.cost_function(angles).hex(), float(ordinary(angles)).hex())

    def test_repeated_finite_shot_measurements_execute_on_device(self):
        source = Path(__file__).parent / "reproductions/layered_vqe/hydrogen.npz"
        problem = load_problem(source)
        backend = BackendConfig.statevector(4, device_name="default.qubit", n_shots=64, seed=42)
        for cls in (QubitAdaptVQE, AdaptVQE):
            solver = cls(problem.hamiltonian, backend_config=backend)
            solver.build_ansatz()
            solver.selected_operators = [0]
            with qml.Tracker(solver.device) as tracker:
                solver.cost_function(np.array([0.31]))
                first = tracker.totals["executions"]
                solver.cost_function(np.array([0.31]))
            self.assertGreater(first, 0)
            self.assertEqual(tracker.totals["executions"], 2 * first)

    def test_support_keeps_native_order_on_cancellation_complex_tails_and_dense_states(self):
        import itertools
        import os
        from threadpoolctl import threadpool_limits
        from pennylane_lightning.core._serialize import QuantumScriptSerializer
        from pennylane_lightning.lightning_qubit_ops import MeasurementsC128
        from core.pauli_measurement import adaptive_state_measurement, lightning_thread_count

        rng = np.random.default_rng(42)
        n = 5
        names = ["".join(p) for p in itertools.product("IXYZ", repeat=n)]
        words = ["IIIII", "ZIIII", "IIZII", *rng.choice(names[1:], 128, replace=False), "IIIII"]
        coefficients = np.array([2.**40, .0001, -2.**40, *rng.normal(size=128), .25])
        ops = [qml.pauli.string_to_pauli_word(w) for w in words]
        observable = qml.Hamiltonian(coefficients, ops)
        full = QuantumScriptSerializer("lightning.qubit", False)._ob(observable)
        device = qml.device("lightning.qubit", wires=n)
        native = MeasurementsC128(device._statevector.state_vector)
        previous = os.environ.get("OMP_NUM_THREADS")
        os.environ["OMP_NUM_THREADS"] = "1"
        try:
            with threadpool_limits(limits=1, user_api="openmp"):
                measure = adaptive_state_measurement(n, observable)
                if lightning_thread_count() is None:
                    self.assertIsNone(measure)
                    return
                self.assertIsNotNone(measure)
                for support in ([0], [2, 5], [0, 4, 18], [1, 2, 3, 11, 12], list(range(32))):
                    state = np.zeros(1 << n, dtype=complex)
                    state[support] = rng.normal(size=len(support)) + 1j*rng.normal(size=len(support))
                    state[support[-1]] *= 1e-24
                    state /= np.linalg.norm(state)
                    device._statevector.reset_state()
                    device._statevector.get_final_state(qml.tape.QuantumScript([qml.StatePrep(state, wires=range(n))]))
                    selected = []
                    def energy(terms):
                        selected.append(len(terms.get_ops()))
                        return native.expval(terms)
                    result = measure(device._statevector.state, energy, full)
                    self.assertEqual(result.hex(), native.expval(full).hex())
                    if len(support) > 16:
                        self.assertEqual(selected, [len(full.get_ops())])
                    else:
                        self.assertLess(selected[0], len(full.get_ops()))
                os.environ["OMP_NUM_THREADS"] = "2"
                self.assertIsNone(adaptive_state_measurement(n, observable))
        finally:
            if previous is None:
                del os.environ["OMP_NUM_THREADS"]
            else:
                os.environ["OMP_NUM_THREADS"] = previous
