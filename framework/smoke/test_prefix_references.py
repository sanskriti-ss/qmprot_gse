# ABOUTME: Checks ordered prefix matrices and their dense and sparse ground-state references.
# ABOUTME: Verifies per-molecule isolation and rejection of unavailable or failed exact references.
import json
from itertools import product
import os
from pathlib import Path
import subprocess
import sys
import unittest

import numpy as np
from scipy.sparse import csr_matrix, kron

from core.hamiltonian_loader import Molecule, QubitHamiltonian
from core.hf_verification import verify_hf_energy
from experiments.experiment5_term_count_sweep import _HamiltonianPrefixes, _prefix_hamiltonian


def hamiltonian(n_qubits, terms):
    molecule = Molecule("sample", "sample", n_qubits, len(terms), 0.0, "")
    coefficients, strings = zip(*terms)
    return QubitHamiltonian(molecule, np.array(coefficients), list(strings), n_qubits, len(terms))


def sparse_references(reuse):
    full = hamiltonian(11, [(1.0, "Z" + "I"*10), (-0.5, "X" + "I"*10), (0.25, "I"*10 + "Y")])
    prefixes = _HamiltonianPrefixes(full)
    energies = []
    for size in (1, 2, 3):
        energy = (prefixes.at(size).molecule.reference_energy if reuse else
                  full._calculate_ground_state_energy(full.coefficients[:size], full.pauli_strings[:size], 11))
        energies.append(float(energy).hex())
    return energies


class PrefixReferences(unittest.TestCase):
    def test_endpoint_can_use_the_reference_matrix_until_its_terms_change(self):
        full = hamiltonian(2, [(1., "ZZ"), (-.25, "XX"), (.3, "XI")])
        prefixes = _HamiltonianPrefixes(full)
        first = prefixes.at(2)
        self.assertIs(first.ordered_sparse_matrix(), prefixes.matrix)
        second = prefixes.at(3)
        self.assertIs(second.ordered_sparse_matrix(), prefixes.matrix)
        self.assertFalse(np.array_equal(first.ordered_sparse_matrix().toarray(), prefixes.matrix.toarray()))
        first.coefficients[0] = 0.
        rebuilt = full._extend_sparse_matrix(None, first.coefficients, first.pauli_strings, 2)
        self.assertEqual(first.ordered_sparse_matrix().toarray().tobytes(), rebuilt.toarray().tobytes())

    def test_pauli_buffers_against_independent_kronecker_products(self):
        pauli = {p: csr_matrix(matrix, dtype=complex) for p, matrix in (
            ("I", [[1, 0], [0, 1]]), ("X", [[0, 1], [1, 0]]),
            ("Y", [[0, -1j], [1j, 0]]), ("Z", [[1, 0], [0, -1]]))}
        for n in range(1, 5):
            source = hamiltonian(n, [(1., "I" * n)])
            zero = csr_matrix((1 << n, 1 << n), dtype=complex)
            for letters in product("IXYZ", repeat=n):
                word = "".join(letters)
                op = pauli[letters[0]]
                for letter in letters[1:]:
                    op = kron(op, pauli[letter], format="csr")
                for coeff in (.375, -.375):
                    expected = zero + coeff * op
                    actual = source._extend_sparse_matrix(None, [coeff], [word], n)
                    for field in ("indices", "indptr", "data"):
                        self.assertEqual(getattr(actual, field).tobytes(), getattr(expected, field).tobytes(),
                                         (word, coeff, field))
                    cancelled = source._extend_sparse_matrix(actual, [-coeff], [word], n)
                    self.assertEqual(cancelled.nnz, 0)

    def test_csr_buffers_and_dense_eigenvalues(self):
        # Includes an exactly cancelled term, a sub-threshold term and both supported Pauli encodings.
        full = hamiltonian(3, [(1.0, "ZII"), (-1.0, "Z(0)"), (1e-13, "XII"), (0.2, "Y(1) X(2)"),
                               (1e-12, "IZI"), (-0.7, "III"), (0.51, "XYZ")])
        prefixes = _HamiltonianPrefixes(full)
        for size in (1, 2, 3, 5, 7):
            result = prefixes.at(size)
            matrix = full._extend_sparse_matrix(None, full.coefficients[:size], full.pauli_strings[:size], 3)
            for field in ("data", "indices", "indptr"):
                self.assertEqual(getattr(prefixes.matrix, field).tobytes(), getattr(matrix, field).tobytes())
            expected = float(np.linalg.eigvalsh(matrix.toarray())[0]).hex()
            self.assertEqual(result.molecule.reference_energy.hex(), expected)
            self.assertEqual(result.molecule.truncated_ground_state_energy.hex(), expected)
            self.assertFalse(np.shares_memory(result.coefficients, full.coefficients))
        with self.assertRaisesRegex(ValueError, "increasing order"):
            prefixes.at(2)

    def test_interleaved_molecules_keep_independent_references(self):
        first = _HamiltonianPrefixes(hamiltonian(2, [(1., "ZI"), (2., "IZ")]))
        second = _HamiltonianPrefixes(hamiltonian(2, [(4., "XI"), (8., "IX")]))
        for size in (1, 2):
            left = first.at(size).molecule.reference_energy
            right = second.at(size).molecule.reference_energy
            self.assertAlmostEqual(left, -1 if size == 1 else -3)
            self.assertAlmostEqual(right, -4 if size == 1 else -12)

    def test_ten_qubit_degenerate_diagonal(self):
        full = hamiltonian(10, [(0.5, "I"*10), (0.75, "Z"+"I"*9)])
        prefixes = _HamiltonianPrefixes(full)
        self.assertEqual(prefixes.at(1).molecule.reference_energy.hex(), (0.5).hex())
        self.assertEqual(prefixes.at(2).molecule.reference_energy.hex(), (-0.25).hex())

    def test_sparse_solver_with_identical_initial_rng_state(self):
        env = dict(os.environ, OMP_NUM_THREADS="1", MKL_NUM_THREADS="1", OPENBLAS_NUM_THREADS="1",
                   VECLIB_MAXIMUM_THREADS="1")
        variants = []
        for reuse in (False, True):
            # ARPACK's default start vector advances native state between solves.
            code = ("import json\nfrom smoke.test_prefix_references import sparse_references\n"
                    f"print(json.dumps(sparse_references({reuse})))")
            run = subprocess.run([sys.executable, "-c", code], cwd=Path(__file__).resolve().parents[1],
                                 env=env, capture_output=True, text=True)
            self.assertEqual(run.returncode, 0, run.stderr + run.stdout)
            variants.append(json.loads(run.stdout))
        self.assertEqual(*variants)

    def test_prefix_never_reports_a_fallback_as_exact(self):
        large = hamiltonian(19, [(1.0, "Z" + "I"*18)])
        with self.assertRaisesRegex(ValueError, "at most 18 qubits"):
            _prefix_hamiltonian(large, 1)
        self.assertIsNone(large._calculate_ground_state_energy(large.coefficients, large.pauli_strings, 19))
        input_ham = hamiltonian(19, [(1., "Z" * 19), (.5, "I" * 19)])
        self.assertIsNone(input_ham.truncate(max_terms=1, target_qubits=19).molecule.truncated_ground_state_energy)
        zero = hamiltonian(11, [(0.0, "I"*11)])
        self.assertEqual(_prefix_hamiltonian(zero, 1).molecule.reference_energy, 0.)
        cancelled = hamiltonian(11, [(1., "Y" * 11), (-1., "Y" * 11)])
        self.assertEqual(cancelled._calculate_ground_state_energy(cancelled.coefficients, cancelled.pauli_strings, 11), 0.)
        invalid = hamiltonian(2, [(1.0, "?Z")])
        with self.assertRaisesRegex(ValueError, "Invalid Pauli word"):
            _prefix_hamiltonian(invalid, 1)
        for coeffs, words in (([1.], ["ZZZ"]), ([1.], ["?Y"]), ([1.], ["X(2)"]), ([1.], ["Noise(0)"]),
                             ([1.], ["X(0) Z(0)"]), ([1., 2.], ["ZI"]), ([np.nan], ["ZI"]), ([1j], ["ZI"])):
            with self.subTest(coeffs=coeffs, words=words), self.assertRaises(ValueError):
                invalid._calculate_ground_state_energy(coeffs, words, 2)

    def test_hf_check_applies_absolute_tolerance_at_molecular_scale(self):
        full = hamiltonian(2, [(-300., "II")])
        with self.assertLogs("core.hf_verification", level="WARNING") as captured:
            passed, info = verify_hf_energy(full, -299.998, atol=1e-6, use_pennylane=True)
        self.assertIn("HF energy mismatch", captured.output[0])
        self.assertFalse(passed)
        self.assertAlmostEqual(info["hf_error"], -.002)
        self.assertTrue(info["pennylane_match"])
        self.assertTrue(verify_hf_energy(full, -299.9999995, atol=1e-6)[0])

    def test_requested_circuit_cross_check_can_fail_verification(self):
        full = hamiltonian(2, [(-300., "II"), (5e-13, "ZI")])
        with self.assertLogs("core.hf_verification", level="WARNING") as captured:
            passed, info = verify_hf_energy(full, -300. - 5e-13, atol=1e-14, use_pennylane=True)
        self.assertIn("differs from matrix HF energy", captured.output[0])
        self.assertTrue(info["hf_matches_expected"])
        self.assertFalse(info["pennylane_match"])
        self.assertFalse(passed)

    def test_projected_reference_against_an_independent_operator(self):
        full = hamiltonian(2, [(-300., "II"), (2., "IY")])
        full.cs_initial_state = np.array([1., 1j, 0., 0.]) / np.sqrt(2)
        passed, info = verify_hf_energy(full, -298., atol=1e-12, use_pennylane=True)
        self.assertTrue(passed)
        self.assertAlmostEqual(info["hf_energy"], -298., places=12)
        self.assertAlmostEqual(info["hf_energy_pennylane"], -298., places=12)
