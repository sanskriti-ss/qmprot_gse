# ABOUTME: Runs true iQCC on a fixed nonstationary glycine projection from the historical prefix experiment.
# ABOUTME: Reconstructs each accepted circuit state and checks its energy on the original ordered Hamiltonian.
from hashlib import sha256
import json
from pathlib import Path
import shutil
import sys

import numpy as np
from openfermion import QubitOperator, get_sparse_operator

from algorithms.vqe_iqcc_true import iQCC_true_VQE
from core.hamiltonian_loader import Molecule, QubitHamiltonian
from smoke.batch import require
from smoke.physics_acceptance import dense_hamiltonian


FIXTURE = Path(__file__).resolve().parent / "reproductions/iqcc_glycine"
INPUT_DIGEST = "6de2adcfdd7926b5406ab1730e26b776489757b076fe837bbf5fce54a0cfa042"
COUNT = 26


def problem(path):
    require(sha256(path.read_bytes()).hexdigest() == INPUT_DIGEST, "Incorrect pinned glycine input")
    data = json.loads(path.read_text())
    state = np.array([complex(*pair) for pair in data["state"]])
    molecule = Molecule("gly", "glycine", 4, COUNT, data["full_casci"], str(path))
    return QubitHamiltonian(molecule, np.array(data["coefficients"][:COUNT]), data["pauli_strings"][:COUNT],
                            4, COUNT, cs_initial_state=state)


def score_rotations(state, matrix, generators, angles):
    qubits = len(state).bit_length() - 1
    for word, angle in reversed(list(zip(generators, angles, strict=True))):
        pauli = get_sparse_operator(QubitOperator(word), n_qubits=qubits)
        state = np.cos(angle / 2)*state - 1j*np.sin(angle / 2)*(pauli @ state)
    return np.vdot(state, matrix @ state)


def run(directory):
    path = directory / "iqcc-control-input.json"
    shutil.copyfile(FIXTURE / "glycine.json", path)
    settings = json.loads((FIXTURE / "observations.json").read_text())["requested_params"]
    result = iQCC_true_VQE(problem(path), **settings).run()
    report = dict(input_sha256=INPUT_DIGEST, terms=COUNT, settings=settings, energy=result.calculated_energy,
                  history=result.convergence_history, angles=result.optimal_parameters.tolist(), metadata=result.metadata)
    (directory / "iqcc-control.json").write_text(json.dumps(report, indent=2, allow_nan=False) + "\n")


def validate(directory):
    ham = problem(directory / "iqcc-control-input.json")
    matrix = dense_hamiltonian(ham)
    reference = np.linalg.eigvalsh(matrix)[0]
    report = json.loads((directory / "iqcc-control.json").read_text())
    observations = json.loads((FIXTURE / "observations.json").read_text())
    meta, angles, history = report["metadata"], report["angles"], report["history"]
    generators = meta["selected_generators"]
    require(report["input_sha256"] == INPUT_DIGEST and report["terms"] == COUNT and
            report["settings"] == observations["requested_params"] and 2 <= len(angles) == len(generators) == len(history)
            and meta["n_molecular_energy_evals"] == len(history) + 1, "Missing repeated dressing on fixed glycine control")
    energies = [score_rotations(ham.cs_initial_state, matrix, generators[:k], angles[:k]) for k in range(1,len(history)+1)]
    np.testing.assert_allclose(energies, history, atol=1e-9, rtol=0)
    np.testing.assert_allclose([reference, history[-1], report["energy"]],
                               [observations["exact_prefix_energy"], report["energy"], reference], rtol=0, atol=1e-8)
    return dict(rotations=len(angles), energy=report["energy"], prefix_reference=reference,
                largest_rescore_difference_ha=float(np.max(np.abs(np.asarray(energies) - history))))


if __name__ == "__main__":
    run(Path(sys.argv[1]))
