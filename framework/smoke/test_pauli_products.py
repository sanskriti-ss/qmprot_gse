# ABOUTME: Checks ordered Pauli products against PennyLane's pairwise composition.
# ABOUTME: Exercises both molecule Hamiltonian encodings and the OpenFermion conversion on real devices.
import contextlib
from functools import reduce
import io
from operator import matmul
from pathlib import Path
import unittest

import numpy as np
import pennylane as qml
from pyscf import lib

from active_space_truncation.run_pipeline import run_pipeline
from algorithms.vqe_qubit_adapt import QubitAdaptVQE
from core.backend_manager import BackendConfig
from core.hamiltonian_loader import Molecule, QubitHamiltonian, pauli_product
from core.iqcc_helpers import of_to_pennylane


class PauliProducts(unittest.TestCase):
    def test_adaptive_cost_uses_the_current_problem_across_builds(self):
        molecule = Molecule("h2", "Hydrogen", 2, 1, -0.75, "", n_electrons=1)
        hamiltonian = QubitHamiltonian.from_dict({"ZI": 0.75}, molecule)
        solver = QubitAdaptVQE(hamiltonian)
        for coefficient in (0.75, 0.5):
            hamiltonian.coefficients[0] = coefficient
            solver.build_ansatz()
            self.assertEqual(solver.cost_function(np.array([])).hex(), (-coefficient).hex())
            solver.selected_operators = [0]
            solver.cost_function(np.array([0.2]))
            self.assertIs(solver.cost_fn.qtape.measurements[0].obs, solver._energy_observable)
        sampled = QubitAdaptVQE(hamiltonian, backend_config=BackendConfig.statevector(2, n_shots=100))
        sampled.build_ansatz()
        self.assertIsNone(sampled._energy_observable)
        self.assertEqual(sampled.cost_function(np.array([])), -0.5)

    def test_composing_a_pauli_word_preserves_factor_order(self):
        for mode in (qml.operation.enable_new_opmath_cm, qml.operation.disable_new_opmath_cm):
            with self.subTest(mode=mode.__name__), mode():
                for word in ((qml.X(2),), (qml.Y(2), qml.Z(0), qml.X(3), qml.Y(1))):
                    expected = reduce(matmul, word)
                    actual = pauli_product(word)
                    self.assertIs(type(actual), type(expected))
                    self.assertEqual(actual.wires, expected.wires)
                    self.assertTrue(qml.equal(actual, expected))

    def test_measurement_tapes_and_binary64_energies(self):
        path = Path(__file__).resolve().parent / "geometries/hydrogen/hydrogen.h5"
        log = io.StringIO()
        with contextlib.redirect_stdout(log), lib.temporary_env(lib.StreamObject, stdout=log):
            problem = run_pipeline("hydrogen", basis="sto-3g", h5_path=str(path), quiet=True)
        dense = problem["hamiltonian"].qubit_hamiltonian
        labels = [" ".join(f"{p}({i})" for i, p in enumerate(word) if p != "I") for word in dense.pauli_strings]
        sparse = QubitHamiltonian(dense.molecule, dense.coefficients, labels, dense.n_qubits, dense.n_terms)
        operator = dense.to_openfermion()
        def reference():
            ops = []
            for word in dense.pauli_strings:
                factors = [getattr(qml, p)(i) for i, p in enumerate(word) if p != "I"]
                ops.append(reduce(matmul, factors) if factors else qml.Identity(0))
            return qml.Hamiltonian(dense.coefficients, ops)
        def tape(make, theta):
            with qml.queuing.AnnotatedQueue() as queue:
                qml.RY(theta, 0)
                for i in range(3):
                    qml.CNOT([i, i + 1])
                qml.expval(make())
            return qml.tape.QuantumScript.from_queue(queue)

        device = qml.device("default.qubit", wires=4)
        for mode in (qml.operation.enable_new_opmath_cm, qml.operation.disable_new_opmath_cm):
            with self.subTest(mode=mode.__name__), mode():
                for theta in (-0.75, 0.0, 0.375):
                    expected = tape(reference, theta)
                    energy = float(device.execute(expected)).hex()
                    for make in (dense.to_pennylane, sparse.to_pennylane, lambda: of_to_pennylane(operator)):
                        actual = tape(make, theta)
                        self.assertEqual(len(actual.operations), len(expected.operations))
                        self.assertTrue(all(qml.equal(left, right) for left, right in zip(actual.operations, expected.operations)))
                        self.assertTrue(qml.equal(actual.measurements[0].obs, expected.measurements[0].obs))
                        self.assertEqual(float(device.execute(actual)).hex(), energy)
