# ABOUTME: Checks particle populations, spin and independently scored benchmark endpoints.
# ABOUTME: Replays physical hydrogen and saved contextual glycine fits in their parent orbitals.
import json
from pathlib import Path
import unittest

import numpy as np
from openfermion import get_sparse_operator, s_squared_operator

from algorithms import get_algorithm
from core.hamiltonian_loader import Molecule, QubitHamiltonian
from experiments._common import LoadedHamiltonian
from core.backend_manager import BackendConfig
from experiments.endpoint_sector import model_reference, summarize_endpoint
from experiments.provenance import load_problem


DATA = Path(__file__).resolve().parent / "reproductions"
FITS = DATA / "layered_vqe"


class EndpointSector(unittest.TestCase):
    def test_ordered_molecular_operator_reuse_and_prefix_mutation(self):
        problem = load_problem(FITS / "hydrogen.npz")
        ham = problem.hamiltonian
        def check():
            fresh = ham._extend_sparse_matrix(None, ham.coefficients, ham.pauli_strings, ham.n_qubits)
            matrix = ham.ordered_sparse_matrix()
            self.assertIs(matrix, ham.ordered_sparse_matrix())
            for name in ("data", "indices", "indptr"):
                self.assertEqual(getattr(matrix, name).tobytes(), getattr(fresh, name).tobytes())
                self.assertFalse(getattr(matrix, name).flags.writeable)
            return matrix

        complete = check()
        ham.coefficients[0] += .125
        changed = check()
        self.assertIsNot(changed, complete)
        ham.pauli_strings[1] = ham.pauli_strings[2]
        edited = check()
        self.assertIsNot(edited, changed)
        ham.pauli_strings = ham.pauli_strings[:3]
        ham.coefficients = ham.coefficients[:3]
        ham.n_terms = 3
        self.assertIsNot(check(), edited)
        solver = get_algorithm("hardware_efficient_vqe")(ham, n_layers=1)
        solver.build_ansatz()
        angles = np.linspace(-.2, .4, solver.n_parameters)
        first = summarize_endpoint(problem, solver, angles)
        self.assertEqual(first, summarize_endpoint(problem, solver, angles))
        self.assertAlmostEqual(first["rescored_energy"], float(solver.cost_function(angles)), places=13)

    def test_glycine_endpoints_keep_the_measured_leakage(self):
        problem = load_problem(FITS / "contextual_gly.npz")
        records = json.loads((FITS / "fits.json").read_text())
        case = next(row for row in records if row["input"] == "contextual_gly.npz")
        reference = json.loads((DATA / "study_checks/semantic_scope.json").read_text())["contextual_glycine"]
        for fit, expected in zip(case["rows"], reference["rows"], strict=True):
            solver = get_algorithm(fit["algorithm"])(problem.hamiltonian, n_layers=2, rotation_gates="RY_RZ")
            solver.build_ansatz()
            actual = summarize_endpoint(problem, solver, fit["final_angles"])
            self.assertAlmostEqual(actual["wrong_electron_probability"], expected["wrong_N_probability"], places=11)
            self.assertAlmostEqual(actual["rescored_energy"], expected["lifted_parent_energy_ha"], places=9)
            self.assertAlmostEqual(sum(actual["electron_count_probabilities"].values()), 1.)
        self.assertAlmostEqual(model_reference(problem)["minimum_possible_abs_error_to_casci"],
                               reference["attainable_floor_mha"] / 1000, places=10)

    def test_spin_and_charge_use_the_scored_state(self):
        problem = load_problem(FITS / "hydrogen.npz")
        solver = get_algorithm("ucc_vqe")(problem.hamiltonian)
        solver.build_ansatz()
        summary = summarize_endpoint(problem, solver, [np.pi / 2, 0.])
        self.assertAlmostEqual(summary["wrong_electron_probability"], 0.)
        self.assertAlmostEqual(summary["mean_spin_squared"], 1.)
        self.assertAlmostEqual(summary["target_spin_z_probability"], 1.)

    def test_density_and_projected_density_preserve_physical_probabilities(self):
        for filename in ("hydrogen.npz", "contextual_gly.npz"):
            problem = load_problem(FITS / filename)
            size = problem.n_qubits_final
            pure = get_algorithm("hardware_efficient_vqe")(problem.hamiltonian, n_layers=1)
            mixed = get_algorithm("hardware_efficient_vqe")(problem.hamiltonian, n_layers=1,
                backend_config=BackendConfig.noisy(size, noise_model="phaseflip", noise_strength=0.))
            pure.build_ansatz()
            mixed.build_ansatz()
            theta = np.linspace(.1, .35, pure.n_parameters)
            a = summarize_endpoint(problem, pure, theta)
            b = summarize_endpoint(problem, mixed, theta)
            self.assertEqual(b["state_kind"], "density_matrix")
            for field in ("wrong_electron_probability", "mean_spin_z", "mean_spin_squared", "rescored_energy"):
                self.assertAlmostEqual(a[field], b[field], places=9)

    def test_noisy_hydrogen_matches_explicit_kraus_charge_and_spin(self):
        from smoke.test_noise_physics import density_score

        problem = load_problem(FITS / "hydrogen.npz")
        solver = get_algorithm("hardware_efficient_vqe")(problem.hamiltonian, n_layers=1,
            backend_config=BackendConfig.noisy(4, noise_model="amplitude_damping", noise_strength=.17))
        solver.build_ansatz()
        theta = np.linspace(-.6, .85, solver.n_parameters)
        actual = summarize_endpoint(problem, solver, theta)
        wrong_n = np.diag([complex(i.bit_count() != 2) for i in range(16)])
        spin = get_sparse_operator(s_squared_operator(2), n_qubits=4).toarray()
        for field, operator in (("wrong_electron_probability", wrong_n), ("mean_spin_squared", spin)):
            expected = density_score(solver, theta, operator, "amplitude_damping", .17, 1)
            self.assertAlmostEqual(actual[field], expected, places=10)
        self.assertGreater(actual["wrong_electron_probability"], .1)


    def test_reference_above_the_entire_spectrum(self):
        molecule = Molecule("trial", "trial", 2, 2, -1., "", n_electrons=1)
        ham = QubitHamiltonian.from_dict({"II": -2., "IX": .1}, molecule)
        problem = LoadedHamiltonian(ham, "trial", "", 2, 2, False, casci_energy=-1.)
        ref = model_reference(problem)
        self.assertAlmostEqual(ref["operator_ground_energy"], -2.1)
        self.assertAlmostEqual(ref["minimum_possible_abs_error_to_casci"], .9)
