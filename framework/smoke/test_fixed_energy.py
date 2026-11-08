# ABOUTME: Checks fixed molecular energy evaluation against each registered circuit on real inputs.
# ABOUTME: Covers ordered native measurements, sampled calls and changing adaptive circuit operators.
from pathlib import Path
import unittest

import numpy as np
import pennylane as qml

from algorithms import ALGORITHMS
from core.backend_manager import BackendConfig, analytic_energy, create_device
from experiments.provenance import load_problem


class FixedEnergy(unittest.TestCase):
    def test_fixed_families_preserve_all_energy_bits(self):
        path = Path(__file__).parent / "reproductions/layered_vqe/hydrogen.npz"
        ham = load_problem(path).hamiltonian
        algorithms = ("hardware_efficient_vqe", "pennylane_vqe", "hva_vqe", "ucc_vqe",
                      "qaoa_inspired_vqe", "nn_ae_vqe", "cb_vqe")
        for name in algorithms:
            with self.subTest(name=name):
                solver = ALGORITHMS[name](ham, n_layers=2, random_seed=42)
                solver.build_ansatz()
                rng = np.random.default_rng(42)
                for theta in rng.uniform(-2., 2., (4, solver.n_parameters)):
                    self.assertEqual(solver.cost_function(theta).hex(), float(solver.cost_fn(theta)).hex())

    def test_dynamic_operations_and_observable_replacement(self):
        device = create_device(BackendConfig.statevector(2))
        operators = [qml.Hamiltonian([1., -.2], [qml.X(0), qml.Z(1)])]
        active = []
        @qml.qnode(device)
        def circuit(theta):
            for wire in active:
                qml.RY(theta[wire], wires=wire)
            return qml.expval(operators[0])
        measured = []
        def observe(state, energy, observable):
            measured.append(state.copy())
            return energy(observable)
        evaluate = analytic_energy(circuit, operators[0], measure_state=observe)
        theta = np.array([.3, .7])
        for wire in (0, 1):
            active.append(wire)
            self.assertEqual(float(evaluate(theta)).hex(), float(circuit(theta)).hex())
        if device.name == "lightning.qubit":
            self.assertEqual(len(measured), 2)
            self.assertFalse(np.array_equal(*measured))
        operators[0] = qml.Hamiltonian([1.], [qml.X(1)])
        self.assertEqual(float(evaluate(theta)).hex(), float(circuit(theta)).hex())

    def test_finite_shots_and_default_device_use_the_original_qnode(self):
        obs = qml.Hamiltonian([1.], [qml.Z(0)])
        for config in (BackendConfig.statevector(1, n_shots=32),
                       BackendConfig.statevector(1, device_name="default.qubit")):
            dev = create_device(config)
            @qml.qnode(dev)
            def circuit(theta):
                qml.RY(theta[0], 0)
                return qml.expval(obs)
            self.assertIs(analytic_energy(circuit, obs), circuit)

    def test_mixed_groups_preserve_each_native_energy_and_shot_behavior(self):
        from core.mixed_measurements import GroupedMixedState

        path = Path(__file__).parent / "reproductions/layered_vqe/hydrogen.npz"
        ham = load_problem(path).hamiltonian.to_pennylane()
        for model in ("amplitude_damping", "phaseflip", "depolarizing"):
            device = create_device(BackendConfig.noisy(4, model, .027))
            self.assertIsInstance(device.target_device, GroupedMixedState)
            stock = qml.device("default.mixed", wires=4)
            def trial(params):
                qml.RY(params[0], 0)
                qml.CNOT([0, 1])
                qml.RZ(params[1], 1)
                channel = {"amplitude_damping": qml.AmplitudeDamping, "phaseflip": qml.PhaseFlip,
                           "depolarizing": qml.DepolarizingChannel}[model]
                for wire in range(4):
                    channel(.027, wire)
                return qml.expval(ham)
            candidate, baseline = qml.QNode(trial, device), qml.QNode(trial, stock)
            for theta in np.random.default_rng(42).uniform(-1, 1, (3, 2)):
                self.assertEqual(float(candidate(theta)).hex(), float(baseline(theta)).hex())
        sampled = create_device(BackendConfig.noisy(4, model, .027, n_shots=8))
        self.assertNotIsInstance(sampled.target_device, GroupedMixedState)

    def test_mixed_batch_keeps_distinct_preparations_and_readout(self):
        from pennylane.devices import DefaultMixed
        from core.mixed_measurements import GroupedMixedState

        group = GroupedMixedState(wires=2, readout_prob=.11)
        stock = DefaultMixed(wires=2, readout_prob=.11)
        first = [qml.RY(.65, 0), qml.CNOT([0, 1]), qml.PhaseFlip(.1, 0)]
        second = [qml.RY(-.9, 0), qml.CNOT([0, 1]), qml.PhaseFlip(.1, 0)]
        z, x = qml.expval(qml.Z(0)), qml.expval(qml.X(0) @ qml.X(1))
        batch = [qml.tape.QuantumScript(ops, [term]) for ops, term in
                 ((first, z), (first, x), (second, x), (second, z), (first, x))]
        expected = stock.batch_execute(batch)
        result = group.batch_execute(batch)
        self.assertEqual([float(x).hex() for x in result], [float(x).hex() for x in expected])

    def test_additional_lightning_callback_reuse_requires_verified_single_thread(self):
        import os
        import subprocess

        framework = Path(__file__).resolve().parents[1]
        source = """import numpy as np, pennylane as qml
from algorithms import get_algorithm
from experiments.provenance import load_problem
h = load_problem('smoke/reproductions/layered_vqe/hydrogen.npz').hamiltonian
s = get_algorithm('pennylane_vqe')(h)
s.build_ansatz()
theta = np.zeros(s.n_parameters)
with qml.Tracker(s.device) as t:
    s.cost_function(theta)
    s.cost_function(theta.copy())
print(t.totals['executions'])
"""
        import sys
        runner = subprocess.run([sys.executable, "-c", source], cwd=framework, capture_output=True,
                                text=True, env={**os.environ, "OMP_NUM_THREADS": "2"}, check=True)
        self.assertEqual(runner.stderr, "")
        self.assertEqual(runner.stdout.strip(), "2")
