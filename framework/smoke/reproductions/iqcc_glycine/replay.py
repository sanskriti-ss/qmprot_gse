# ABOUTME: Replays historical true-iQCC on a fixed glycine Hamiltonian and compares four upstream snapshots.
# ABOUTME: Uses archived solver modules in separate processes and checks an independent dense ground energy.
from collections import Counter
import contextlib
from functools import reduce
import hashlib
import io
import json
import os
from pathlib import Path
import subprocess
import sys
import tarfile
import tempfile
import warnings

HERE = Path(__file__).resolve().parent
OBSERVED = json.loads((HERE / "observations.json").read_text())


def solve(framework, solver_digest):
    import numpy as np
    from algorithms import vqe_iqcc_true
    from core.backend_manager import BackendConfig
    from core.hamiltonian_loader import Molecule, QubitHamiltonian

    solver_file = Path(vqe_iqcc_true.__file__).resolve()
    if hashlib.sha256(solver_file.read_bytes()).hexdigest() != solver_digest:
        raise ValueError(f"Unexpected solver source {solver_file}")
    encoded = (HERE / "glycine.json").read_bytes()
    if hashlib.sha256(encoded).hexdigest() != OBSERVED["input_sha256"]:
        raise ValueError("The archived glycine input digest differs")
    problem = json.loads(encoded)
    count = OBSERVED["used_terms"]
    coefficients = np.array(problem["coefficients"][:count])
    words = problem["pauli_strings"][:count]
    state = np.array([complex(*pair) for pair in problem["state"]])
    n = len(words[0])

    paulis = dict(I=np.eye(2), X=np.array([[0, 1], [1, 0]]),
                  Y=np.array([[0, -1j], [1j, 0]]), Z=np.diag([1, -1]))
    matrix = sum(coefficient * reduce(np.kron, [paulis[p] for p in word])
                 for coefficient, word in zip(coefficients, words))
    np.testing.assert_array_equal(matrix, matrix.conj().T)
    ground = float(np.linalg.eigvalsh(matrix)[0])
    molecule = Molecule("gly", "glycine", n, count, ground, "pipeline_glycine", molecular_formula="C2H5NO2")
    ham = QubitHamiltonian(molecule, coefficients, words, n, count, cs_initial_state=state)
    with contextlib.redirect_stdout(io.StringIO()), warnings.catch_warnings(record=True) as captured:
        warnings.simplefilter("always")
        solver = vqe_iqcc_true.iQCC_true_VQE(ham, backend_config=BackendConfig.statevector(n),
                                              **OBSERVED["requested_params"])
        result = solver.run()
    for name, module in sys.modules.copy().items():
        if name.split(".")[0] in {"algorithms", "core", "config"} and getattr(module, "__file__", None):
            if not Path(module.__file__).resolve().is_relative_to(framework):
                raise RuntimeError(f"Imported {name} from outside its archived source")
    counts = Counter(f"{w.category.__name__} {Path(w.filename).name}:{w.lineno}: {w.message}" for w in captured)
    return dict(ground=ground, energy=float(result.calculated_energy), backend=solver.device.name,
                trace=[float(x).hex() for x in result.convergence_history], warnings=dict(counts))


def compare():
    import numpy as np

    repo = Path(subprocess.check_output(["git", "rev-parse", "--show-toplevel"], cwd=HERE, text=True).strip())
    reference_trace = None
    with tempfile.TemporaryDirectory(prefix="qmprot-iqcc-") as temp:
        for snapshot in OBSERVED["snapshots"]:
            commit, label = snapshot["commit"], snapshot["label"]
            archive = Path(temp) / label
            archive.mkdir()
            packed = subprocess.check_output(["git", "archive", commit, "framework/config.py",
                                              "framework/core", "framework/algorithms"], cwd=repo)
            with tarfile.open(fileobj=io.BytesIO(packed)) as contents:
                contents.extractall(archive, filter="data")
            framework = archive / "framework"
            env = dict(os.environ, PYTHONPATH=str(framework), OMP_NUM_THREADS="1", OPENBLAS_NUM_THREADS="1",
                       MKL_NUM_THREADS="1", VECLIB_MAXIMUM_THREADS="1", MPLCONFIGDIR=str(Path(temp) / "mpl"))
            output = subprocess.check_output([sys.executable, str(HERE / "replay.py"), str(framework),
                                              snapshot["solver_sha256"]], cwd=framework, env=env, text=True)
            record = json.loads(output)
            np.testing.assert_allclose(record["ground"], OBSERVED["exact_prefix_energy"], atol=1e-10, rtol=0)
            np.testing.assert_allclose(record["energy"], OBSERVED["endpoint"], atol=1e-9, rtol=0)
            np.testing.assert_allclose([float.fromhex(x) for x in record["trace"]],
                                       [float.fromhex(x) for x in OBSERVED["trace_hex"]], atol=1e-9, rtol=0)
            if reference_trace is not None and record["trace"] != reference_trace:
                raise ValueError(f"{label} differs from the other snapshots on this machine")
            if record["warnings"] != OBSERVED["warnings"] or record["backend"] != "lightning.qubit":
                raise ValueError(f"Unexpected backend or diagnostics: {record['backend']}, {record['warnings']}")
            reference_trace = record["trace"]
            gap = 1000 * (record["ground"] - record["energy"])
            print(f"{label:11} {commit[:8]}  exact={record['ground']:.12f}  reported={record['energy']:.12f}"
                  f"  undershoot={gap:.9f} mHa  warnings={sum(record['warnings'].values())}"
                  f"  matches recorded binary64 trace={record['trace'] == OBSERVED['trace_hex']}", flush=True)


if __name__ == "__main__":
    if len(sys.argv) == 1:
        compare()
    else:
        print(json.dumps(solve(Path(sys.argv[1]).resolve(), sys.argv[2])))
