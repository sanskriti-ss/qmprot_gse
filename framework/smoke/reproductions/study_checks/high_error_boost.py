# ABOUTME: Reconciles the five March classically boosted VQE heatmap points with their saved rows and warnings.
# ABOUTME: Independently rescores the encoded reference, fitted circuit and Ritz span on a frozen glycine operator.
import csv
from hashlib import sha256
import json
from pathlib import Path
import re
from statistics import mean

import numpy as np

from algorithms.vqe_CB import ClassicallyBoostedVQE
from smoke.iqcc_control import FIXTURE, INPUT_DIGEST, problem
from smoke.physics_acceptance import circuit_state, dense_hamiltonian


FRAMEWORK = Path(__file__).resolve().parents[3]


def historical():
    directory = FRAMEWORK / "plots/20260318_200609"
    records = json.loads((directory / "all_results.json").read_text())
    with (directory / "run_results.csv").open() as f:
        runs = {(r["molecule_abbrev"], r["algorithm_name"]): r for r in csv.DictReader(f)}
    with (directory / "all_results.csv").open() as f:
        summary = list(csv.DictReader(f))
    assert len(records) == len(runs) == len(summary) == 55
    assert all(r["method"] == "unknown" and not r["vqe_energy"] for r in summary)
    log = (FRAMEWORK / "run_all_cs.log").read_text().splitlines()
    warning = "Classical boost failed (ControlledQubitUnitary.__init__() missing 1 required positional argument: 'control_wires'); using VQE energy."
    locations = {}
    for i, line in enumerate(log):
        match = re.search(r"Running cb_vqe on (\w+)$", line)
        if match and "__main__" in line:
            molecule = match[1]
        if warning in line:
            locations[molecule] = i + 1
    rows = {}
    for r in records:
        if r["algorithm_name"] != "cb_vqe":
            continue
        mol = r["molecule_abbrev"]
        csv_row = runs[mol, "cb_vqe"]
        for col in ("calculated_energy", "hf_energy", "reference_energy", "error"):
            assert float(csv_row[col]) == r[col]
        assert r["metadata"]["cb_energy"] == r["metadata"]["vqe_energy"] == r["calculated_energy"]
        assert r["metadata"]["improvement"] == 0.
        det = runs[mol, "qubit_adapt_vqe"]
        assert det["n_parameters"] == det["n_iterations"] == "0"
        assert det["calculated_energy"] == runs[mol, "adapt_vqe"]["calculated_energy"]
        rows[mol] = dict(casci=r["reference_energy"], published_cb=r["calculated_energy"],
                         cb_error_ha=r["error"], cb_reported_hf=r["hf_energy"],
                         encoded_reference=float(det["calculated_energy"]),
                         encoded_reference_minus_casci_ha=float(det["error"]), warning_line=locations[mol])
    assert len(rows) == len(locations) == 5
    return dict(archive_revision="db605e94ac9b9177438409190994c8a6a1d8a007",
                archive_directory=str(directory.relative_to(FRAMEWORK)), log="run_all_cs.log",
                source="algorithms/vqe_CB.py at this revision, lines 277–280 and 408–414",
                warning=warning, experiment="cc-pVDZ, 14 active qubits projected to 8; parent active-space CASCI reference; COBYLA, 100 max iterations, 48 angles",
                data_quality="run_results.csv and all_results.json agree on all five CB rows. all_results.csv labels all 55 methods unknown and omits every VQE energy.",
                interpretation="All five entries are unboosted fitted energies. Archived zero-operator ADAPT energies give an upper bound on the Ritz minimum of a span containing the correctly encoded reference. The March CB HF column uses a half-filled computational determinant. The March output directory lacks the reduced operators and encoded kets needed to reconstruct the exact corrected energies.",
                mean_published_error_ha=mean(r["cb_error_ha"] for r in rows.values()),
                mean_encoded_reference_minus_casci_ha=mean(r["encoded_reference_minus_casci_ha"] for r in rows.values()),
                molecules=rows)


def frozen_control():
    ham = problem(FIXTURE / "glycine.json")
    matrix = dense_hamiltonian(ham)
    settings = dict(n_layers=2, max_iterations=100, optimizer="COBYLA", random_seed=42)
    solver = ClassicallyBoostedVQE(ham, **settings)
    result = solver.run()
    state = circuit_state(solver, result.optimal_parameters, "cb_vqe")
    reference = ham.cs_initial_state
    hf = float(np.vdot(reference, matrix @ reference).real)
    fitted = float(np.vdot(state, matrix @ state).real)
    basis, triangular = np.linalg.qr(np.column_stack((reference, state)))
    assert abs(triangular[1, 1]) > 1e-7
    ritz = float(np.linalg.eigvalsh(basis.conj().T @ matrix @ basis)[0])
    ground = float(np.linalg.eigvalsh(matrix)[0])
    np.testing.assert_allclose([result.hf_energy, result.metadata["vqe_energy"], result.calculated_energy],
                               [hf, fitted, ritz], atol=1e-10, rtol=0)
    assert ground - 1e-10 <= ritz <= min(hf, fitted) + 1e-10
    return dict(input="smoke/reproductions/iqcc_glycine/glycine.json", input_sha256=INPUT_DIGEST,
                scope="Independent STO-3G glycine 12-to-4-qubit projection, first 26 ordered terms. Each energy uses this same prefix.",
                solver_sha256=sha256((FRAMEWORK / "algorithms/vqe_CB.py").read_bytes()).hexdigest(),
                settings=settings, prefix_ground_ha=ground, encoded_hf_ha=hf, fitted_ha=fitted,
                independent_complex_ritz_ha=ritz, reported_boost_ha=result.calculated_energy,
                published_historical_replay=False)


if __name__ == "__main__":
    print(json.dumps(dict(march_heatmap=historical(), contemporary_frozen_control=frozen_control()),
                     indent=2, allow_nan=False))
