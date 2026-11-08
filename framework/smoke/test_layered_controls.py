# ABOUTME: Replays fitted and HF-preparing layered circuits against frozen molecular inputs.
# ABOUTME: Checks reported energies, state normalization, initial fidelity and archived high-error endpoints.
from hashlib import sha256
import json
from pathlib import Path
import unittest

import numpy as np
from openfermion import get_sparse_operator

from algorithms import get_algorithm
from experiments.experiment5_term_count_sweep import _prefix_hamiltonian
from experiments.provenance import load_problem
from smoke.layered_controls import reference_angles, simulate


EVIDENCE = Path(__file__).resolve().parent / "reproductions/layered_vqe"


class LayeredControls(unittest.TestCase):
    def test_replay_all_recorded_starts_and_endpoints(self):
        data = json.loads((EVIDENCE / "fits.json").read_text())
        self.assertEqual(sum(len(case["rows"]) for case in data), 24)
        archived = {"hardware_efficient_vqe": .6237935424, "pennylane_vqe": .3510103971}
        for case in data:
            path = EVIDENCE / case["input"]
            self.assertEqual(sha256(path.read_bytes()).hexdigest(), case["input_sha256"])
            h = load_problem(path).hamiltonian
            if h.n_terms != case["terms"]:
                h = _prefix_hamiltonian(h, case["terms"])
            matrix = get_sparse_operator(h.to_openfermion(), n_qubits=h.n_qubits)
            ground = np.linalg.eigvalsh(matrix.toarray())[0]
            reference = h.cs_initial_state
            if reference is None:
                electrons = h.molecule.n_electrons
                reference = np.zeros(2**h.n_qubits, dtype=complex)
                reference[((1 << electrons) - 1) << (h.n_qubits - electrons)] = 1
            self.assertAlmostEqual(ground, case["ground"], places=9)
            for row in case["rows"]:
                name, axes = row["algorithm"], row["axes"]
                hardware = name == "hardware_efficient_vqe"
                gates = "RY" if axes == "Y" else "RY_RZ"
                solver = get_algorithm(name)(h, n_layers=2, rotation_gates=gates)
                solver.build_ansatz()
                with self.subTest(input=case["input"], algorithm=name, start=row["start"], budget=row["budget"]):
                    for key, energy in (("initial_angles", "initial_energy"), ("final_angles", "endpoint")):
                        angles = np.array(row[key])
                        ket = simulate(reference, angles, axes, hardware, 2)
                        self.assertAlmostEqual(np.linalg.norm(ket), 1., places=12)
                        self.assertAlmostEqual(np.vdot(ket, matrix @ ket).real, row[energy], places=9)
                        self.assertAlmostEqual(solver.cost_function(angles), row[energy], places=9)
                    if row["start"] == "reference":
                        angles = reference_angles(reference, 2, axes, hardware)
                        np.testing.assert_array_equal(angles, row["initial_angles"])
                        self.assertAlmostEqual(row["initial_energy"], case["hf"], places=9)
                    if case["terms"] == 200 and row["start"] == "default" and row["budget"] == 64:
                        self.assertAlmostEqual(row["endpoint"] - ground, archived[name], places=9)
