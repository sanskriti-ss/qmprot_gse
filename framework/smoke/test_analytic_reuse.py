# ABOUTME: Counts physical hardware-efficient energy evaluations on real PennyLane simulators.
# ABOUTME: Checks repeated parameters, bounded eviction, rebuilt circuits and finite shots.
import unittest

import numpy as np
import pennylane as qml

from algorithms.vqe_hardware_efficient import HardwareEfficientVQE
from core.backend_manager import BackendConfig
from core.hamiltonian_loader import Molecule, QubitHamiltonian


class AnalyticReuse(unittest.TestCase):
    def solver(self, shots=0):
        molecule = Molecule("orbital", "One orbital", 1, 1, -1., "", n_electrons=1)
        hamiltonian = QubitHamiltonian.from_dict({"Z": 1.}, molecule)
        config = BackendConfig.statevector(1, n_shots=shots)
        solver = HardwareEfficientVQE(hamiltonian, n_layers=1, rotation_gates="RY", backend_config=config)
        solver.build_ansatz()
        return solver

    def test_four_recent_vectors_are_reused_bit_for_bit(self):
        solver = self.solver()
        vectors = [np.array([.2, i / 10]) for i in range(5)]
        direct = [float(solver.cost_fn(v)).hex() for v in vectors]
        order = (0, 1, 2, 3, 0, 4, 0, 1)
        executions = (1, 2, 3, 4, 4, 5, 5, 6)
        with qml.Tracker(solver.device) as tracker:
            for index, expected in zip(order, executions, strict=True):
                self.assertEqual(solver.cost_function(vectors[index].copy()).hex(), direct[index])
                self.assertEqual(tracker.totals["executions"], expected)

    def test_rebuilding_the_reference_invalidates_previous_energies(self):
        solver = self.solver()
        vector = np.array([.2, .3])
        with qml.Tracker(solver.device) as tracker:
            occupied = solver.cost_function(vector)
            solver.cost_function(vector)
        self.assertEqual(tracker.totals["executions"], 1)
        solver.prepare_hf = False
        solver.build_ansatz()
        with qml.Tracker(solver.device) as tracker:
            vacuum = solver.cost_function(vector)
            solver.cost_function(vector)
        self.assertEqual(tracker.totals["executions"], 1)
        self.assertAlmostEqual(occupied, -np.cos(.5), places=14)
        self.assertAlmostEqual(vacuum, np.cos(.5), places=14)

    def test_finite_shots_execute_each_request(self):
        solver = self.solver(shots=64)
        with qml.Tracker(solver.device) as tracker:
            energies = [solver.cost_function(np.array([.2, .3])) for _ in range(3)]
        self.assertEqual(tracker.totals["executions"], 3)
        self.assertTrue(all(-1 <= e <= 1 for e in energies))
