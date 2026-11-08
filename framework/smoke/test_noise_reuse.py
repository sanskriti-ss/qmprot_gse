# ABOUTME: Checks deterministic energy reuse on an actual noisy PennyLane circuit.
# ABOUTME: Verifies exact parameter keys, bounded reuse and fresh finite-shot measurements.
import unittest

import numpy as np
import pennylane as qml

from core.backend_manager import BackendConfig, create_device
from experiments.noise_resilience import _noise_objective


class NoiseObjectiveTest(unittest.TestCase):
    def circuit(self, config):
        device = create_device(config)

        @qml.qnode(device)
        def measure(parameters):
            qml.RY(parameters.reshape(-1)[0], wires=0)
            qml.AmplitudeDamping(config.noise_strength, wires=0)
            return qml.expval(qml.PauliZ(0))

        return measure

    def test_exact_values_and_distinct_vectors(self):
        config = BackendConfig.noisy(1, "amplitude_damping", 0.05)
        circuit = self.circuit(config)
        energy = _noise_objective(circuit, config)
        p = np.array([0.7])
        vectors = [p, p.copy(), np.nextafter(p, np.inf), p.astype(np.float32), p.reshape(1, 1),
                   np.array([0.0]), np.array([-0.0]), p]
        expected = [float(circuit(x)).hex() for x in vectors]
        with qml.Tracker(circuit.device) as tracker:
            actual = [float(energy(x)).hex() for x in vectors]
        self.assertEqual(expected, actual)
        self.assertEqual(tracker.totals["executions"], len(vectors) - 1)

    def test_sampled_circuit_executes_each_call(self):
        config = BackendConfig.noisy(1, "amplitude_damping", 0.05, n_shots=100)
        circuit = self.circuit(config)
        energy = _noise_objective(circuit, config)
        with qml.Tracker(circuit.device) as tracker:
            for _ in range(3):
                energy(np.array([0.7]))
        self.assertEqual(tracker.totals["executions"], 3)
