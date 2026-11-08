# ABOUTME: Exercises the independent frozen glycine replay against native objectives and physical endpoints.
# ABOUTME: Checks channel location, the parent-spin reconstruction and sensitivity to corrupted scientific records.
import unittest

import numpy as np

from algorithms import get_algorithm
from experiments._common import make_backend_config
from experiments.endpoint_sector import summarize_endpoint
from experiments.provenance import load_problem
from smoke.noise_glycine_oracle import (FROZEN, GlycineCircuit, PAULI, check_endpoint, depolarize,
                                       parent_isometry, physical_endpoint)


class NoiseGlycineOracle(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.problem = load_problem(FROZEN)
        cls.circuit = GlycineCircuit(FROZEN)
        cls.embedding = parent_isometry(cls.circuit)

    def test_asymmetric_complex_state_agrees_with_explicit_kraus_operators(self):
        ket = np.array([1., .2j, -.3, .5+.7j])
        ket /= np.linalg.norm(ket)
        rho = np.outer(ket, ket.conj())
        exact = rho.copy()
        p = .19
        for wire in (0, 1):
            weighted = np.zeros_like(rho)
            for letter, weight in (("I", 1-p), ("X", p/3), ("Y", p/3), ("Z", p/3)):
                matrices = [PAULI["I"], PAULI["I"]]
                matrices[wire] = PAULI[letter]
                unitary = np.kron(*matrices)
                weighted += weight * unitary @ exact @ unitary.conj().T
            exact = weighted
        np.testing.assert_allclose(depolarize(rho, p), exact, atol=2e-16, rtol=0)

    def test_exact_native_channel_sequence_and_oracle(self):
        theta = np.random.default_rng(42).uniform(-np.pi/8, np.pi/8, 24)
        for p in (0., .01):
            with self.subTest(strength=p):
                backend = make_backend_config(8, "depolarizing" if p else None, p)
                model = get_algorithm("hardware_efficient_vqe")(self.problem.hamiltonian, n_layers=2,
                                                                random_seed=42, backend_config=backend)
                model.build_ansatz()
                model.cost_fn.construct((theta,), {})
                gates = model.cost_fn.qtape.operations
                wire = range(8)
                layout = [("StatePrep", tuple(wire))] + [("RY", (q,)) for q in wire]
                for _ in range(2):
                    layout.extend(("CNOT", (q, (q+1)%8)) for q in wire)
                    layout.extend(("RY", (q,)) for q in wire)
                    if p:
                        layout.extend(("DepolarizingChannel", (q,)) for q in wire)
                self.assertEqual([(g.name, tuple(g.wires)) for g in gates], layout)
                np.testing.assert_array_equal([float(g.data[0]) for g in gates if g.name == "RY"], theta)
                self.assertTrue(all(float(g.data[0]) == p for g in gates if g.name == "DepolarizingChannel"))
                state = self.circuit.state(theta, p)
                endpoint = physical_endpoint(self.circuit, self.embedding, state)
                self.assertLess(abs(endpoint["rescored_energy"] - model.cost_function(theta)), 5e-10)
                native = summarize_endpoint(self.problem, model, theta)
                check_endpoint(endpoint, native)
                for field in ("wrong_electron_probability", "mean_spin_squared", "rescored_energy"):
                    with self.assertRaisesRegex(ValueError, field):
                        check_endpoint(endpoint, {**native, field: native[field] + .002})
