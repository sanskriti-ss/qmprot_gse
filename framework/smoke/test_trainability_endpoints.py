# ABOUTME: Replays trainability endpoints and distinguishes accepted states from adaptive probes.
# ABOUTME: Exercises budget stops, optimizer convergence and recorded electron-number diagnostics.
from contextlib import redirect_stderr
from io import StringIO
from pathlib import Path
import unittest
import warnings

from core.hamiltonian_loader import Molecule, QubitHamiltonian
from experiments import cost_to_accuracy as cta
from experiments._common import LoadedHamiltonian
from experiments.trainability import _run_adapt_at_target, _run_he_at_target
from experiments.provenance import load_problem


class TrainabilityEndpoints(unittest.TestCase):
    def test_reference_screen_replays_the_same_displaced_energies_at_each_target(self):
        problem = load_problem(Path(__file__).parent / "reproductions/layered_vqe/hydrogen.npz")
        for algorithm in ("qubit_adapt_vqe", "adapt_vqe"):
            options = dict(algorithm=algorithm, gradient_threshold=1e-8, qubit_adapt_max_gradient_candidates=None,
                           optimizer="COBYLA", max_iterations=16, convergence_threshold=1e-8,
                           random_seed=42, eval_budget=64)
            shared = {}
            first = _run_adapt_at_target(problem, target_n_params=1, reference_probes=shared, **options)
            with warnings.catch_warnings(record=True) as observed:
                warnings.simplefilter("always")
                warm = _run_adapt_at_target(problem, target_n_params=2, reference_probes=shared, **options)
                cold = _run_adapt_at_target(problem, target_n_params=2, **options)
            expected = []
            if algorithm == "qubit_adapt_vqe":
                self.assertEqual(warm.n_cost_evals, options["eval_budget"])
                self.assertEqual(warm.stop_reason, "evaluation budget during re-optimization")
                warning = "COBYLA: Invalid MAXFUN; it should be at least num_vars + 2; it is set to 4"
                expected = [(UserWarning, warning)] * 2
            self.assertEqual([(w.category, str(w.message)) for w in observed], expected)
            self.assertEqual(first.n_reused_reference_probes, 0)
            self.assertGreater(warm.n_reused_reference_probes, 0)
            self.assertEqual(warm.n_cost_evals, cold.n_cost_evals)
            self.assertEqual(warm.endpoint, cold.endpoint)
            self.assertEqual(warm.error_trace, cold.error_trace)
            self.assertEqual(warm.probe_evals_to, cold.probe_evals_to)
            self.assertEqual(warm.accepted_evals_to, cold.accepted_evals_to)

    def test_interrupted_probe_can_reach_a_reference_without_acceptance(self):
        mol = Molecule("pair", "pair", 2, 1, -.02, "", n_electrons=1)
        ham = QubitHamiltonian.from_dict({"XX": 1000.}, mol)
        loaded = LoadedHamiltonian(ham, "pair", "basis", 2, 2, False,
                                   casci_energy=-.02, hf_energy=0.)
        row = _run_adapt_at_target(loaded, algorithm="qubit_adapt_vqe", target_n_params=2,
                                   gradient_threshold=1e-8, qubit_adapt_max_gradient_candidates=None,
                                   optimizer="COBYLA", max_iterations=16, convergence_threshold=1e-9,
                                   random_seed=42, eval_budget=2)
        self.assertEqual(row.actual_n_params, 0)
        self.assertIn(row.evals_to_threshold["1.6mHa"], (1, 2))
        self.assertEqual(row.probe_evals_to["1.6mHa"], row.evals_to_threshold["1.6mHa"])
        self.assertIsNone(row.accepted_evals_to["1.6mHa"])
        self.assertEqual(row.accepted_evals_to["hf"], 0)
        self.assertEqual(row.endpoint["parameters"], [])
        self.assertEqual(row.endpoint["selected_operators"], [])
        self.assertEqual(row.endpoint["rescored_energy"], row.final_energy)
        self.assertEqual(row.endpoint["wrong_electron_probability"], 0.)
        record = cta.make_record(loaded.molecule, row)
        summary = cta.summarize([record])[0]
        self.assertEqual(summary["n_reached_1.6mHa"], 1)
        self.assertEqual(summary["n_probe_reached_1.6mHa"], 1)
        self.assertEqual(summary["n_accepted_reached_1.6mHa"], 0)
        self.assertEqual(summary["median_accepted_evals_to_hf"], 0)


    def test_optimizer_success_with_no_accuracy_stop(self):
        loaded = load_problem(Path(__file__).parent / "reproductions/layered_vqe/hydrogen.npz")
        with redirect_stderr(StringIO()) as progress:
            row = _run_he_at_target(loaded, target_n_params=2, optimizer="COBYLA",
                                    max_iterations=500, convergence_threshold=1e-8,
                                    random_seed=42, eval_budget=500)
        self.assertIn("VQE Optimization", progress.getvalue())
        self.assertIs(row.optimizer_success, True)
        self.assertFalse(row.stopped_at_accuracy)
        self.assertEqual(len(row.endpoint["parameters"]), 2)
        self.assertAlmostEqual(row.final_energy, row.endpoint["rescored_energy"], places=12)
        self.assertGreater(row.endpoint["wrong_electron_probability"], 0.)

    def test_budget_retains_initial_incumbent_before_lbfgs_accepts_a_step(self):
        loaded = load_problem(Path(__file__).parent / "reproductions/layered_vqe/hydrogen.npz")
        with redirect_stderr(StringIO()) as progress:
            row = _run_he_at_target(loaded, target_n_params=2, optimizer="L-BFGS-B",
                                    max_iterations=3, convergence_threshold=1e-8,
                                    random_seed=42, eval_budget=3)
        self.assertIn("VQE Optimization", progress.getvalue())
        initial, finished = row.endpoint["accepted_states"]
        self.assertEqual(finished["kind"], "last_incumbent_at_budget")
        self.assertEqual(finished["parameters"], initial["parameters"])
        self.assertEqual(finished["energy"], initial["energy"])
        self.assertLess(row.best_error_vs_casci, row.error_vs_casci)
        self.assertIsNone(row.optimizer_success)

    def test_accuracy_stop_retains_exact_triggering_point_without_hf_preparation(self):
        molecule = Molecule("pair", "pair", 2, 1, 1., "", n_electrons=1)
        ham = QubitHamiltonian.from_dict({"ZI": 1.}, molecule)
        loaded = LoadedHamiltonian(ham, "pair", "basis", 2, 2, False, casci_energy=1., hf_energy=-1.)
        with redirect_stderr(StringIO()) as progress:
            row = _run_he_at_target(loaded, target_n_params=2, optimizer="COBYLA",
                                    max_iterations=16, convergence_threshold=1e-8, random_seed=42,
                                    eval_budget=16, init_strategy="zeros", hf_init=False, stop_at_accuracy=True)
        self.assertIn("VQE Optimization", progress.getvalue())
        self.assertTrue(row.stopped_at_accuracy)
        self.assertEqual(row.accepted_evals_to["1.6mHa"], 1)
        self.assertEqual(row.endpoint["parameters"], [0., 0.])
        self.assertEqual(row.endpoint["wrong_electron_probability"], 1.)
        self.assertEqual(row.endpoint["accepted_states"][-1]["kind"], "crossing_at_accuracy")
        summary = cta.summarize([cta.make_record(loaded.molecule, row)])[0]
        self.assertEqual(summary["n_accepted_reached_1.6mHa"], 1)
        self.assertIsNone(row.optimizer_success)

    def test_each_adaptive_generator_set_is_replayable(self):
        molecule = Molecule("pair", "pair", 2, 1, -1., "", n_electrons=1)
        ham = QubitHamiltonian.from_dict({"XI": 1., "XX": .3}, molecule)
        loaded = LoadedHamiltonian(ham, "pair", "basis", 2, 2, False, casci_energy=-1., hf_energy=0.)
        for algo in ("adapt_vqe", "qubit_adapt_vqe"):
            with self.subTest(algorithm=algo):
                row = _run_adapt_at_target(loaded, algorithm=algo, target_n_params=1,
                                           gradient_threshold=1e-8, qubit_adapt_max_gradient_candidates=None,
                                           optimizer="COBYLA", max_iterations=40, convergence_threshold=1e-8,
                                           random_seed=42, eval_budget=100)
                self.assertEqual(row.actual_n_params, 1)
                initial, fitted = row.endpoint["accepted_states"]
                self.assertEqual(initial["kind"], "encoded_reference")
                self.assertEqual(fitted["selected_operators"], row.endpoint["selected_operators"])
                self.assertEqual(fitted["parameters"], row.endpoint["parameters"])
                self.assertAlmostEqual(fitted["energy"], fitted["rescored_energy"], places=12)
                self.assertAlmostEqual(row.final_energy, fitted["energy"], places=12)
