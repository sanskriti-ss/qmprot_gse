# ABOUTME: Checks CASCI accuracy and the HF milestone against counted statevector energies.
# ABOUTME: Exercises off-sector evaluations, early stopping and saved-result reanalysis.
import csv
import json
from contextlib import redirect_stderr
from io import StringIO
import tempfile
import unittest
from pathlib import Path

import numpy as np

from core.hamiltonian_loader import Molecule, QubitHamiltonian
from experiments import cost_to_accuracy as cta
from experiments._common import AccuracyReached, CostEvalCounter, LoadedHamiltonian, build_vqe
from experiments.trainability import TrainabilityRow, _fill_accuracy_metrics, _run_he_at_target


class CostToAccuracy(unittest.TestCase):
    def setUp(self):
        molecule = Molecule("pair", "Two orbitals", 2, 1, -1., "", n_electrons=1)
        hamiltonian = QubitHamiltonian.from_dict({"ZI": 1., "IZ": 2.}, molecule)
        self.loaded = LoadedHamiltonian(hamiltonian, "pair", "basis", 2, 2, False,
                                        casci_energy=-1., hf_energy=1.)
        self.solver = build_vqe(hamiltonian, n_layers=1, rotation_gates="RY", prepare_hf=False)
        # With the leading layer at zero the final rotations select |00>, |11>, |01>.
        self.parameters = np.pi * np.array([[0, 0, 0, 0], [0, 0, 1, 1], [0, 0, 0, 1]])

    def test_below_sector_ground_state_is_only_an_hf_hit(self):
        counter = CostEvalCounter()
        counter.attach(self.solver)
        energies = [float(self.solver.cost_function(p)) for p in self.parameters]
        counter.detach(self.solver)
        np.testing.assert_allclose(energies, [3., -3., -1.])
        row = TrainabilityRow("hardware_efficient_vqe", 4, 4, 3, 3, .01, energies[1], -2., False)
        _fill_accuracy_metrics(row, counter, self.loaded)
        self.assertEqual(row.evals_to_threshold["1.6mHa"], 3)
        self.assertEqual(row.evals_to_hf, 2)
        self.assertAlmostEqual(row.best_error_vs_casci, -2.)
        self.assertAlmostEqual(row.best_abs_error_vs_casci, 0.)
        self.assertEqual([p[0] for p in row.error_trace], [1, 2, 3])
        record = cta.make_record("pair", row)
        self.assertEqual(record["evals_to_hf"], 2)
        self.assertEqual(record["evals_to_1.6mHa"], 3)
        summary = cta.summarize([record])[0]
        self.assertEqual((summary["n_reached_hf"], summary["median_evals_to_hf"]), (1, 2))
        self.assertEqual(summary["median_evals_to_1.6mHa"], 3)

    def test_stop_within_counts_the_crossing_evaluation(self):
        counter = CostEvalCounter(stop_within=(-1., .0016), max_evals=3)
        counter.attach(self.solver)
        self.solver.cost_function(self.parameters[0])
        self.solver.cost_function(self.parameters[1])
        with self.assertRaises(AccuracyReached) as caught:
            self.solver.cost_function(self.parameters[2])
        counter.detach(self.solver)
        self.assertEqual(caught.exception.n_evals, 3)
        self.assertAlmostEqual(caught.exception.energy, -1.)

    def test_he_cannot_stop_at_a_lower_off_sector_energy(self):
        with redirect_stderr(StringIO()) as progress:
            row = _run_he_at_target(self.loaded, target_n_params=2, eval_budget=4, optimizer="COBYLA",
                                    max_iterations=4, convergence_threshold=1e-8, random_seed=1,
                                    init_strategy="zeros", stop_at_accuracy=True)
        self.assertIn("VQE Optimization", progress.getvalue())
        self.assertFalse(row.stopped_at_accuracy)
        self.assertEqual(row.n_cost_evals, 4)
        self.assertEqual(row.evals_to_hf, 1)
        self.assertIsNone(row.evals_to_threshold["1.6mHa"])
        self.assertAlmostEqual(row.best_error_vs_casci, -2.)

    def test_reanalysis_uses_endpoint_bounds_for_one_sided_files(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            molecule = root / "pair"
            molecule.mkdir()
            (molecule / "run_config.json").write_text(json.dumps({"hf_energy": 1., "casci_energy": -1.}))
            with (molecule / "results.csv").open("w", newline="") as stream:
                writer = csv.writer(stream)
                writer.writerow(["algorithm", "actual_n_params", "target_n_params", "n_cost_evals",
                                 "final_energy", "error_vs_casci", "evals_to_1.6mHa"])
                writer.writerow(["hardware_efficient_vqe", 4, 4, 10, -3., -2., 2])
                writer.writerow(["hardware_efficient_vqe", 6, 6, 12, -.999, .001, 4])
            result = cta.analyze_existing(root)
            far, near = result["records"]
            self.assertEqual(far["evals_to_1.6mHa"], "")
            self.assertEqual(far["evals_to_hf"], 10)
            self.assertEqual(near["evals_to_1.6mHa"], 12)
            self.assertEqual(near["bounds_only"], 1)


if __name__ == "__main__":
    unittest.main()
