# ABOUTME: Checks fixed-angle circuit reuse against each original Lightning QNode and its public tape.
# ABOUTME: Exercises changing references, live entanglers, inspection, and the gradient study's forward circuit.
import unittest
from pathlib import Path

import numpy as np
import pennylane as qml

from algorithms import get_algorithm
from experiments.barren_plateau import _gradient_cost
from experiments.provenance import load_problem


class RotationLayout(unittest.TestCase):
    def solver(self, method, **settings):
        problem = load_problem(Path(__file__).parent / "reproductions/layered_vqe/hydrogen.npz")
        solver = get_algorithm(method)(problem.hamiltonian, n_layers=2, **settings)
        solver.build_ansatz()
        return solver

    def compare(self, solver, vector):
        value = solver.cost_function(vector)
        observed = solver.cost_fn.qtape
        state = solver.device._statevector.state.copy()
        wanted = float(solver.cost_fn(vector))
        self.assertEqual(value.hex(), wanted.hex())
        self.assertEqual(state.tobytes(), solver.device._statevector.state.tobytes())
        self.assertTrue(qml.equal(observed, solver.cost_fn.qtape))
        return observed

    def test_registered_rotations_match_inspectable_tape_and_native_state(self):
        for gate_set, gate in (("RY", "CNOT"), ("RY_RZ", "CZ"), ("full", "CNOT")):
            settings = dict(max_free_parameters=10, rotation_gates=gate_set, entangling_gate=gate)
            for solver in (self.solver("hardware_efficient_vqe", **settings), self.solver("pennylane_vqe"),
                           self.solver("cb_vqe")):
                with self.subTest(method=solver.name, gates=gate_set):
                    vectors = np.random.default_rng(43).normal(size=(4, solver.n_parameters))
                    saved = self.compare(solver, vectors[0])
                    saved_values = [op.data for op in saved.operations]
                    for vector in vectors[1:]:
                        current = self.compare(solver, vector)
                        self.assertIs(current.measurements[0], saved.measurements[0])
                    self.assertEqual([op.data for op in saved.operations], saved_values)

    def test_reference_and_entangler_changes_use_the_current_gates(self):
        solver = self.solver("hardware_efficient_vqe")
        h = solver.hamiltonian
        vectors = np.random.default_rng(43).normal(size=(8, solver.n_parameters))
        self.compare(solver, vectors[0])
        solver.entangling_gate = "CZ"
        self.compare(solver, vectors[1])
        solver.prepare_hf = False
        self.compare(solver, vectors[2])
        solver.prepare_hf = True
        h.cs_initial_state = np.eye(16, dtype=complex)[1].copy()
        self.compare(solver, vectors[3])
        self.compare(solver, vectors[4])
        h.cs_initial_state[:] = np.eye(16)[8]
        self.compare(solver, vectors[5])
        self.compare(solver, vectors[6])
        with qml.Tracker(solver.device) as tracker:
            self.compare(solver, vectors[7])
        self.assertEqual(tracker.totals["executions"], 2)

    def test_forward_gradient_cost_preserves_signed_zero_and_noncontiguous_parameters(self):
        for name in ("hardware_efficient_vqe", "pennylane_vqe"):
            solver = self.solver(name)
            gradient_cost = _gradient_cost(solver.cost_fn)
            ordinary = qml.QNode(solver.cost_fn.func, solver.device, interface=None, diff_method=None)
            source = np.random.default_rng(0).normal(size=2 * solver.n_parameters)
            for vector in (source[::2], np.full(solver.n_parameters, -0.), source[1::2], source[::2].astype("float32")):
                energy = float(gradient_cost(vector))
                state = solver.device._statevector.state.copy()
                self.assertEqual(energy.hex(), float(ordinary(vector)).hex())
                self.assertEqual(state.tobytes(), solver.device._statevector.state.tobytes())

    def test_unusable_contextual_state_keeps_the_per_call_hf_diagnostic(self):
        solver = self.solver("hardware_efficient_vqe")
        solver.hamiltonian.cs_initial_state = np.full(16, np.nan, dtype=complex)
        with self.assertLogs("core.base_vqe", level="WARNING") as logged:
            solver.cost_function(np.zeros(solver.n_parameters))
            solver.cost_function(np.ones(solver.n_parameters))
        self.assertEqual(len(logged.records), 2)
        self.assertTrue(all("falling back to HF initial state" in r.getMessage() for r in logged.records))
