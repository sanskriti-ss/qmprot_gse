# ABOUTME: Verifies real-state ordered Pauli expectations against the original Lightning measurement.
# ABOUTME: Exercises cancellation, signed and tiny amplitudes, serialization checks and memory limits.
import unittest

import numpy as np
import pennylane as qml
from pennylane_lightning.core._serialize import QuantumScriptSerializer
from pennylane_lightning.lightning_qubit_ops import MeasurementsC128, StateVectorC128

from core.ordered_measurement import ordered_measurement, _ordered_rows, _row_dtype


class OrderedMeasurement(unittest.TestCase):
    def setUp(self):
        self.n = 8
        self.rng = np.random.default_rng(1921)
        self.state = StateVectorC128(self.n)
        self.expectation = MeasurementsC128(self.state).expval
        self.serialize = QuantumScriptSerializer("lightning.qubit", False)._ob
        words = self.rng.choice(list("IXYZ"), size=(128, self.n))
        ops = [qml.pauli.string_to_pauli_word("".join(w)) for w in words]
        coeffs = self.rng.uniform(-2, 2, len(ops))
        coeffs[1:4] = [2.**-52, 1.+2.**-27, -1.]
        self.observable = qml.dot(coeffs, ops)
        self.native = self.serialize(self.observable)
        self.measure = ordered_measurement(self.n, self.observable)

    def test_finite_dense_and_sparse_inputs(self):
        for i in range(72):
            x = self.rng.normal(size=1 << self.n).astype(complex)
            if i % 3 == 0:
                x[self.rng.uniform(size=x.size) > .04] = -0.
            if i % 3 == 1:
                x.real = np.ldexp(x.real, self.rng.integers(-680, 20, x.size))
            if i % 3 == 2:
                x[0] = x[-1] = 2.**-1074
            x /= np.linalg.norm(x)
            self.state.UpdateData(x)
            expected = self.expectation(self.native)
            if self.measure is not None:
                self.assertEqual(self.measure(x, self.expectation, self.native).hex(), expected.hex())
        if _row_dtype():
            import os
            if os.environ.get("OMP_NUM_THREADS") == "1":
                self.assertIsNotNone(self.measure)

    def test_complex_and_different_native_operator(self):
        x = self.rng.normal(size=256) + 1j * self.rng.normal(size=256)
        x /= np.linalg.norm(x)
        self.state.UpdateData(x)
        if self.measure is not None:
            self.assertEqual(self.measure(x, self.expectation, self.native).hex(),
                             self.expectation(self.native).hex())
            other = self.serialize(qml.Z(0))
            real = x.real.astype(complex)
            self.state.UpdateData(real)
            self.assertEqual(self.measure(real, self.expectation, other).hex(), self.expectation(other).hex())

    def test_small_and_oversized_plans(self):
        self.assertIsNone(ordered_measurement(4, qml.Z(0)))
        full = np.arange(65536, dtype=np.int32)
        self.assertIsNone(_ordered_rows(12, full.tobytes(), full.tobytes(), np.ones(full.size).tobytes()))
