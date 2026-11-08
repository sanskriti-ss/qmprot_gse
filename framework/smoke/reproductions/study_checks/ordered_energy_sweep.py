# ABOUTME: Compares ordered Pauli measurements to Lightning on fixed, molecular and adversarial inputs.
# ABOUTME: Times the initial table build, warmed calls, retained operator data and temporary state lanes.
import argparse
from hashlib import sha256
import json
from pathlib import Path
from statistics import median
from time import perf_counter

import numpy as np
from pennylane_lightning import lightning_qubit_ops
from pennylane_lightning.core._serialize import QuantumScriptSerializer
from pennylane_lightning.lightning_qubit_ops import MeasurementsC128, StateVectorC128
from scipy.sparse import _sparsetools

from core.ordered_measurement import ordered_measurement, _ordered_rows
from experiments.provenance import load_problem


def run(path):
    ham = load_problem(path).hamiltonian
    observable = ham.to_pennylane()
    native = QuantumScriptSerializer("lightning.qubit", False)._ob(observable)
    sv = StateVectorC128(ham.n_qubits)
    evaluate = MeasurementsC128(sv).expval
    start = perf_counter()
    measure = ordered_measurement(ham.n_qubits, observable)
    setup = perf_counter() - start
    sources = {m.__name__: sha256(Path(m.__file__).read_bytes()).hexdigest() for m in (lightning_qubit_ops, _sparsetools)}
    if measure is None:
        raise ValueError(f"Untimed native fallback for {path}, {sources}")
    _ordered_rows.cache_clear()
    energies, before, after = [], [], []
    rng = np.random.default_rng(87)
    for i in range(85):
        x = rng.uniform(-1, 1, 1 << ham.n_qubits).astype(complex)
        if i % 3 == 1:
            x[rng.uniform(size=x.size) < .94] = -0.
        elif i % 3 == 2:
            x.real = np.ldexp(x.real, rng.integers(-680, 10, x.size))
        x /= np.linalg.norm(x)
        sv.UpdateData(x)
        times, vals = {}, {}
        for method in (("candidate", "native") if i % 2 == 0 else ("native", "candidate")):
            start = perf_counter()
            energy = measure(x, evaluate, native) if method == "candidate" else evaluate(native)
            times[method] = perf_counter() - start
            vals[method] = float(energy).hex()
        if vals["native"] != vals["candidate"]:
            raise ValueError(f"Scientific difference for state {i}: {vals}")
        if i == 0:
            cold = times["candidate"]
        else:
            before.append(times["native"])
            after.append(times["candidate"])
        energies.append(vals["native"])
    cache = _ordered_rows.cache_info()
    if cache.misses != 1 or cache.hits != len(energies)-1:
        raise ValueError(f"Unexpected shared plan usage {cache}")
    d, width = ham.n_qubits, 16
    ps = observable.pauli_rep
    flips = [sum(1 << (d-1-i) for i,p in w.items() if p in "XY") for w in ps]
    phases = [sum(1 << (d-1-i) for i,p in w.items() if p in "YZ") for w in ps]
    active = [(x,z,c*((-1)**((x&z).bit_count()//2))) for x,z,c in zip(flips,phases,ps.values()) if not (x&z).bit_count()%2]
    xs, zs, cs = zip(*active)
    mat, indices, signs = _ordered_rows(d, np.array(xs,np.int32).tobytes(), np.array(zs,np.int32).tobytes(), np.array(cs).tobytes())
    return dict(qubits=d, terms=len(ps), lanes=width, states=len(energies), energy_sha256=sha256(json.dumps(energies).encode()).hexdigest(),
                native_median_ms=1e3*median(before), ordered_median_ms=1e3*median(after), setup_ms=setup*1e3, cold_first_call_ms=cold*1e3,
                retained_plan_bytes=sum(a.nbytes for a in (mat.data, mat.indices, mat.indptr, indices, signs)),
                temporary_state_lane_bytes=indices.size*mat.dtype.itemsize, input_sha256=sha256(path.read_bytes()).hexdigest(), wheel_sha256=sources)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("inputs", type=Path, nargs="+")
    args = parser.parse_args()
    print(json.dumps([run(path) for path in args.inputs], indent=2))
