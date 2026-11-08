# ABOUTME: Checks repeatable trial seeds and molecule weighting in the native prefix study.
# ABOUTME: Verifies the accuracy poster's changing molecular cohorts and uncropped energy fractions.
import csv
from dataclasses import replace
import json
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest

import matplotlib.pyplot as plt
import pandas as pd

from experiments.experiment5_term_count_sweep import TermCountRow, _write_summary_csv, run_term_count_sweep
from plots import make_poster_figures


FRAMEWORK = Path(__file__).resolve().parents[1]


class PairedPrefix(unittest.TestCase):
    def test_paired_trials_survive_roster_changes_and_prefix_subsets(self):
        inputs = FRAMEWORK / "smoke/reproductions/layered_vqe"
        with tempfile.TemporaryDirectory() as tmp:
            def run(name, methods, counts, seeds):
                directory = Path(tmp) / name
                args = [sys.executable, "-m", "experiments.experiment5_term_count_sweep", "--prepared-inputs",
                        str(inputs), "--molecules", "hydrogen", "--cs-target-qubits", "0", "--algorithms",
                        *methods, "--term-counts", *counts, "--seeds", *seeds, "--n-layers", "1",
                        "--max-iter", "16", "--no-plots", "--output-dir", str(directory)]
                p = subprocess.run(args, cwd=FRAMEWORK, capture_output=True, text=True)
                self.assertEqual(p.returncode, 0, p.stdout + p.stderr)
                self.assertNotIn("[WARNING]", p.stderr)
                rows = [json.loads(line) for line in (directory/"cases.jsonl").read_text().splitlines()]
                with (directory/"batch_all_results.csv").open() as stream:
                    csv_rows = list(csv.DictReader(stream))
                expected = [(int(k), a, int(s)) for k in counts for a in methods for s in seeds]
                self.assertEqual([(int(r["n_parameters"]), r["algorithm"], int(r["random_seed"])) for r in csv_rows], expected)
                for r in rows:
                    self.assertAlmostEqual(r["energy"], r["endpoint_sector"]["rescored_energy"], places=9)
                return {(r["prefix_terms"], r["algorithm"], r["random_seed"]): r for r in rows}
            methods = ["hardware_efficient_vqe", "qaoa_inspired_vqe"]
            both = run("both", methods, ["1","15"], ["31","93"])
            subset = run("subset", methods[::-1], ["15"], ["93","31"])
            self.assertEqual(subset, {key: row for key, row in both.items() if key[0] == 15})
            self.assertNotEqual(subset[15, methods[1], 31]["parameters"], subset[15, methods[1], 93]["parameters"])

    def test_duplicate_inputs_rejected_before_work(self):
        for options in (dict(trial_seeds=[2,2]), dict(trial_seeds=[]), dict(random_seed=-1),
                        dict(random_seed=2, trial_seeds=[1]), dict(molecules=["gly", " Gly "]),
                        dict(algorithms=["hva_vqe", "hva_vqe"]), dict(term_counts=[1,1])):
            with self.subTest(options=options), self.assertRaises(ValueError):
                run_term_count_sweep(**options)

    def test_each_molecule_counts_once_when_a_trial_fails(self):
        base = TermCountRow(molecule="gly", algorithm="hva_vqe", n_parameters=1, algorithm_n_parameters=2,
                            n_qubits=4, n_terms_available=15, energy=-2., full_reference_energy=-2.,
                            prefix_reference_energy=-2., error_vs_full_reference=0., abs_error_vs_full_reference=0.,
                            error_vs_prefix_reference=0., abs_error_vs_prefix_reference=0., n_iterations=1,
                            n_cost_evals=4, runtime_seconds=.2, converged=False, random_seed=31)
        rows = [base, replace(base, random_seed=93, abs_error_vs_full_reference=2.),
                replace(base, molecule="ala", abs_error_vs_full_reference=5.),
                replace(base, molecule="ala", random_seed=93, status="failed")]
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "summary.csv"
            _write_summary_csv(path, rows)
            with path.open() as stream:
                record, = csv.DictReader(stream)
        self.assertEqual((record["n_samples"], record["n_molecules"]), ("3","2"))
        self.assertEqual(json.loads(record["trial_seeds_by_molecule"]), {"ala":[31], "gly":[31,93]})
        self.assertEqual(float(record["mean_abs_error_vs_full_reference"]), 3.)
        self.assertEqual(float(record["std_abs_error_vs_full_reference"]), 2.)

    def test_poster_displays_observed_cohort_and_recovery_range(self):
        rows = [("gly",1,1,"ok",.20,.30), ("arg",1,1,"ok",-.02,-.12),
                ("gly",2,1,"stopped_early",.20,.30), ("arg",2,2,"ok",-.03,-.13),
                ("gly",3,1,"stopped_early",.20,.30), ("arg",3,2,"stopped_early",-.03,-.13)]
        df = pd.DataFrame(rows,columns=["molecule","k","attained_k","status","error_vs_casci","error_vs_hf"])
        fig = make_poster_figures._accuracy_poster_figure(df)
        try:
            left, right = fig.axes
            self.assertEqual([t.get_text() for t in left.get_xticklabels()], ["1\n(2/2)","2\n(1/2)","3\n(0/2)"])
            self.assertEqual([t.get_text() for t in right.get_xticklabels()], ["1\n(1/2)","2\n(1/2)","3\n(0/2)"])
            self.assertGreater(right.get_ylim()[1], 1.3)
            mean = next(line for line in left.lines if line.get_label() == "Mean over contributing molecules")
            self.assertAlmostEqual(mean.get_ydata()[0], .11)
            self.assertNotIn("improves", left.get_title())
        finally:
            plt.close(fig)
