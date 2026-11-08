# ABOUTME: Compares grouped noisy Pauli measurements and final density matrices with PennyLane.
# ABOUTME: Counts native basis rotations and exercises the distinct-preparation and tracker paths.
import unittest

import numpy as np
import pennylane as qml
from pennylane.devices import DefaultMixed

from core.mixed_measurements import GroupedMixedState


class Readouts(unittest.TestCase):
    def test_reused_rotations_preserve_all_group_values_and_last_state(self):
        class CountRotations(GroupedMixedState):
            rotations = 0

            def _apply_operation(self, op):
                if type(op) is qml.Hadamard:
                    self.rotations += 1
                return super()._apply_operation(op)

        operations = [qml.RY(.6, 0), qml.RX(.13, 1), qml.CNOT([0, 2]), qml.DepolarizingChannel(.027, 1)]
        rs = np.random.default_rng(42)
        words = [rs.choice([qml.X, qml.Y, qml.Z], size=4) for _ in range(55)]
        tapes = [qml.tape.QuantumScript(operations, [qml.expval(qml.prod(*(f(i) for i, f in enumerate(word))))])
                 for word in words]
        stock = DefaultMixed(wires=4)
        candidate = CountRotations(wires=4)
        expected = stock.batch_execute(tapes)
        actual = candidate.batch_execute(tapes)
        self.assertEqual(np.asarray(actual).tobytes(), np.asarray(expected).tobytes())
        self.assertEqual(candidate._state.tobytes(), stock._state.tobytes())
        self.assertEqual(candidate.state.tobytes(), stock.state.tobytes())
        total = sum(op is not qml.Z for w in words for op in w)
        self.assertLess(candidate.rotations, total)
        candidate.rotations = 0
        with qml.Tracker(candidate) as tracker:
            tracked = candidate.batch_execute(tapes)
        self.assertEqual(tracker.totals["executions"], len(tapes))
        self.assertEqual(candidate.rotations, total)
        self.assertEqual(np.asarray(tracked).tobytes(), np.asarray(expected).tobytes())

        tapes[5] = qml.tape.QuantumScript([qml.RX(-.61, 0)], tapes[5].measurements)
        self.assertEqual(np.asarray(candidate.batch_execute(tapes)).tobytes(),
                         np.asarray(stock.batch_execute(tapes)).tobytes())
