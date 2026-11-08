# ABOUTME: Verifies molecular electron and frozen-core counts for sulfur-containing QMProt inputs.
# ABOUTME: Checks the selected active-space orbital ranks using the pinned chemical geometries.
from contextlib import redirect_stdout
from io import StringIO
from pathlib import Path
import unittest

from pyscf import lib

from active_space_truncation.geometry import load_geometry_from_h5
from active_space_truncation.step1_orbitals import run_orbital_analysis


class SulfurInputs(unittest.TestCase):
    def test_electron_count_core_and_active_ranks(self):
        for molecule, electrons, core, start in (("cys", 64, 11, 30), ("met", 80, 13, 38)):
            with self.subTest(molecule=molecule):
                path = Path(__file__).resolve().parent / "geometries" / molecule / f"{molecule}.h5"
                geometry = load_geometry_from_h5(str(path))
                self.assertEqual(geometry.n_electrons, electrons)
                log = StringIO()
                with redirect_stdout(log), lib.temporary_env(lib.StreamObject, stdout=log):
                    diag = run_orbital_analysis(geometry, basis="sto-3g")
                self.assertEqual(diag.mol.nelectron, electrons)
                self.assertEqual(diag.n_core_orbitals, core)
                self.assertEqual(diag.proposed_active_indices, list(range(start, start + 6)))


class ContextualProjection(unittest.TestCase):
    def test_operator_and_state_use_the_same_unitary(self):
        import numpy as np
        from scipy.linalg import expm
        from core.hamiltonian_loader import Molecule, QubitHamiltonian
        from contextual_subspace.cs_reduction import apply_contextual_subspace_reduction

        terms = {"YYY": .9223398493293176, "XYZ": .5946420154403149,
                 "ZYY": -.9084551514332808, "ZXZ": .02270746516308031}
        ham = QubitHamiltonian.from_dict(terms, Molecule("probe", "Probe", 3, 4, 0., "", n_electrons=1))
        reduced, meta = apply_contextual_subspace_reduction(ham, 2, dfs_cutoff_seconds=.1)
        self.assertTrue(meta["reduced"])
        paulis = dict(I=np.eye(2), X=np.array([[0, 1], [1, 0]]),
                      Y=np.array([[0, -1j], [1j, 0]]), Z=np.diag([1, -1]))

        def matrix(word):
            op = np.array([[1.]])
            for letter in word:
                op = np.kron(op, paulis[letter])
            return op

        # This two-clique noncontextual model fixes Z on the middle rotated wire at +1.
        rotations = [(.7806381478047378, "XII"), (np.pi / 2, "YYZ"),
                     (np.pi / 2, "IZY"), (np.pi / 2, "ZYI")]
        unitary = np.eye(8)
        for angle, generator in rotations:
            unitary = expm(1j * angle / 2 * matrix(generator)) @ unitary
        projection = np.eye(8)[:, [0, 1, 4, 5]]
        lifted = unitary.conj().T @ projection
        full_matrix = sum(c * matrix(p) for p, c in terms.items())
        reduced_matrix = sum(c * matrix(p) for p, c in reduced.to_dict().items())
        np.testing.assert_allclose(reduced_matrix, lifted.conj().T @ full_matrix @ lifted, atol=1e-8)
        reference = lifted.conj().T @ np.eye(8)[:, 4]
        self.assertAlmostEqual(np.vdot(reference, reference).real, .5)
        reference /= np.linalg.norm(reference)
        self.assertAlmostEqual(abs(np.vdot(reference, reduced.cs_initial_state)) ** 2, 1., places=12)
        self.assertAlmostEqual(meta["hf_projection_probability"], .5)
        self.assertEqual(meta["stabilizers"], [{"pauli": "IZI", "eigenvalue": 1}])
        for saved, (angle, generator) in zip(meta["rotations"], rotations, strict=True):
            self.assertEqual(saved["pauli"], generator)
            self.assertAlmostEqual(saved["angle_rad"], angle, places=7)

    def test_anticommuting_terms_without_commuting_generators(self):
        from contextual_subspace.cs_reduction import apply_contextual_subspace_reduction
        from core.hamiltonian_loader import Molecule, QubitHamiltonian
        from math import sqrt
        for offset in (0., 1.):
            terms = {"XI": .2, "ZI": .3}
            if offset:
                terms["II"] = offset
            ham = QubitHamiltonian.from_dict(terms, Molecule("pair", "Pair", 2, len(terms), 0., "", n_electrons=1))
            reduced, meta = apply_contextual_subspace_reduction(ham, 1, dfs_cutoff_seconds=.1)
            self.assertEqual(reduced.n_qubits, 1)
            expected = offset - sqrt(.2**2 + .3**2)
            self.assertAlmostEqual(reduced.to_dict()["I"], expected)
            self.assertAlmostEqual(meta["selected_noncontextual_energy"], expected)

    def test_scalar_operator_has_no_stabilizers(self):
        from contextual_subspace.cs_reduction import apply_contextual_subspace_reduction
        from core.hamiltonian_loader import Molecule, QubitHamiltonian
        ham = QubitHamiltonian.from_dict({"II": 1.}, Molecule("pair", "Pair", 2, 1, 1., "", n_electrons=1))
        result, meta = apply_contextual_subspace_reduction(ham, 1, dfs_cutoff_seconds=.1)
        self.assertIs(result, ham)
        self.assertFalse(meta["reduced"])

    def test_metadata_records_the_actual_hf_compatible_sector(self):
        from contextual_subspace.cs_reduction import apply_contextual_subspace_reduction
        from core.hamiltonian_loader import Molecule, QubitHamiltonian
        ham = QubitHamiltonian.from_dict({"ZI": -1., "II": 2.}, Molecule("pair", "Pair", 2, 2, 1., "", n_electrons=1))
        result, meta = apply_contextual_subspace_reduction(ham, 1, dfs_cutoff_seconds=.1)
        self.assertEqual(meta["noncontextual_ground_state_energy"], 1.)
        self.assertEqual(meta["selected_noncontextual_energy"], 3.)
        self.assertEqual(meta["stabilizers"], [{"pauli": "ZI", "eigenvalue": -1}])
        self.assertEqual(meta["hf_projection_probability"], 1.)
        self.assertEqual(result.to_dict(), {"I": 3.})
