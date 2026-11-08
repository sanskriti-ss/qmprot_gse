# ABOUTME: Checks true iQCC screening against device measurements of every generator on a frozen glycine input.
# ABOUTME: Verifies ordered Pauli products, exact zero slopes and the screening conditions for sampled and rotated states.
import json
import unittest

import numpy as np
from openfermion import QubitOperator

from algorithms.vqe_iqcc_true import _dress, _optimal_angle, _pauli_hamiltonian, _rotation_terms, iQCC_true_VQE
from core.backend_manager import BackendConfig
from core.hamiltonian_loader import Molecule, QubitHamiltonian
from core.iqcc_helpers import of_to_pennylane
from smoke.iqcc_control import FIXTURE, problem


class IQCCReferenceScreen(unittest.TestCase):
    def test_glycine_screen_matches_all_device_slopes(self):
        h = problem(FIXTURE / "glycine.json")
        settings = json.loads((FIXTURE / "observations.json").read_text())["requested_params"]
        solver = iQCC_true_VQE(h, backend_config=BackendConfig.statevector(4, device_name="lightning.qubit"),
                               **settings)
        result = solver.run()
        self.assertEqual(solver.device.name, "lightning.qubit")
        dressed = _pauli_hamiltonian(h)
        zero_count = 0
        measured = 0
        for step in range(result.metadata["n_screen_sweeps"]):
            best = None
            best_gradient = 0.
            for word in solver.operator_pool:
                generator = QubitOperator(word)
                anti = QubitOperator()
                slope = QubitOperator()
                for term, coefficient in dressed.terms.items():
                    (product, phase), = (generator * QubitOperator(term)).terms.items()
                    if phase.imag:
                        anti.terms[term] = coefficient
                        slope.terms[product] = -phase.imag * coefficient
                found_anti, found_slope = _rotation_terms(dressed, generator)
                for expected, actual in ((anti, found_anti), (slope, found_slope)):
                    self.assertEqual(list(expected.terms), list(actual.terms))
                    self.assertEqual(np.asarray(list(expected.terms.values())).tobytes(),
                                     np.asarray(list(actual.terms.values())).tobytes())
                if not slope.terms:
                    continue
                measured += 1
                gradient = solver.compute_hf_energy(of_to_pennylane(slope))
                if all(any(pauli in "XY" for _, pauli in term) for term in slope.terms):
                    self.assertEqual(gradient.hex(), (0.).hex())
                    zero_count += 1
                if abs(gradient) > abs(best_gradient):
                    best_gradient = gradient
                    best = word, anti, slope
            if step == result.n_iterations:
                self.assertLess(abs(best_gradient), solver.gradient_threshold)
                break
            word, anti, slope = best
            angle = _optimal_angle(solver.compute_hf_energy(of_to_pennylane(anti)), best_gradient, solver.bound)
            self.assertEqual(word, result.metadata["selected_generators"][step])
            self.assertEqual(angle.hex(), result.optimal_parameters[step].hex())
            dressed = _dress(dressed, anti, slope, angle)
        self.assertGreater(zero_count, 0)
        self.assertEqual(zero_count, result.metadata["n_screen_basis_zeros"])
        self.assertEqual(measured, result.metadata["n_screen_expectations"])

    def test_sampling_other_device_and_superpositions_keep_device_screening(self):
        molecule = Molecule("one", "one", 1, 1, -1., "", n_electrons=1)
        for device, shots, reference, expect_skip in (
            ("lightning.qubit", 0, None, True),
            ("lightning.qubit", 0, np.array([0., 1j]), True),
            ("default.qubit", 0, None, False),
            ("lightning.qubit", 32, None, False),
            ("lightning.qubit", 0, np.array([1e-18j, 1.]), False),
        ):
            with self.subTest(device=device, shots=shots, reference=reference):
                h = QubitHamiltonian.from_dict({"Z": 1.}, molecule)
                h.cs_initial_state = reference
                config = BackendConfig.statevector(1, device_name=device, n_shots=shots)
                result = iQCC_true_VQE(h, max_operators=1, gradient_threshold=0., backend_config=config).run()
                self.assertEqual(result.metadata["n_screen_expectations"], 2)
                self.assertEqual(result.metadata["n_screen_basis_zeros"], 2 if expect_skip else 0)
