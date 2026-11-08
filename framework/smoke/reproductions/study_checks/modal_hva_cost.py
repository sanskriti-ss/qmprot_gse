# ABOUTME: Measures sampled noisy HVA measurement groups on the eight-qubit glycine fixture.
# ABOUTME: Records native circuit sizes, group results and timings on a bounded Modal CPU worker.
import json
from pathlib import Path

import modal

from smoke.modal_representative import image

app = modal.App("qmprot-hva-cost-sample")


@app.function(image=image, cpu=2, memory=4096, timeout=480, retries=0)
def measure():
    import os
    from hashlib import sha256
    from time import perf_counter

    for name in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "VECLIB_MAXIMUM_THREADS"):
        os.environ[name] = "1"
    import numpy as np
    from threadpoolctl import threadpool_limits

    from core.mixed_measurements import _readout_order
    from experiments._common import NOISE_ALGORITHMS, build_vqe, make_backend_config, sample_initial_parameters
    from experiments.provenance import framework_source, load_problem

    started = perf_counter()
    path = Path("/root/smoke/reproductions/layered_vqe/contextual_gly.npz")
    digest = sha256(path.read_bytes()).hexdigest()
    if digest != "8c07e44128d2bc4463a351e80c37b88d384b605a72a353bb17d9ab19517dbe85":
        raise ValueError("Unexpected glycine fixture")
    hamiltonian = load_problem(path).hamiltonian
    cases = []
    with threadpool_limits(limits=1):
        for channel in ("depolarizing", "phaseflip", "amplitude_damping"):
            solver = build_vqe(
                hamiltonian, algorithm="hva_vqe", n_layers=2,
                backend_config=make_backend_config(8, channel, .01), optimizer="COBYLA",
                max_iterations=150, convergence_threshold=1e-10, random_seed=42,
                supported_algorithms=NOISE_ALGORITHMS,
            )
            angles = sample_initial_parameters(
                solver.n_parameters, strategy="small_random", rng=np.random.default_rng(42),
            )
            solver.cost_fn.construct((angles,), {})
            device = solver.device.target_device
            groups, _ = device.batch_transform(solver.cost_fn.tape)
            indices = (0, len(groups) // 2, len(groups) - 1)
            tapes = [device.expand_fn(groups[index]) for index in indices]
            case = dict(channel=channel, strength=.01, groups=len(groups), parameters=len(angles),
                        operations_per_group=[len(tape.operations) for tape in tapes],
                        shared_readout_order=_readout_order(tapes) is not None, samples=[])
            for index, tape in zip(indices, tapes, strict=True):
                before = perf_counter()
                result = device.batch_execute([tape])
                seconds = perf_counter() - before
                sample = dict(group=index, seconds=seconds, values=np.asarray(result).ravel().tolist())
                case["samples"].append(sample)
                print(channel, json.dumps(sample), flush=True)
            cases.append(case)
    return dict(input_sha256=digest, scientific_source_sha256=sha256(framework_source()).hexdigest(),
                qubits=hamiltonian.n_qubits, terms=hamiltonian.n_terms, seed=42, layers=2,
                worker_seconds=perf_counter() - started, cases=cases)


@app.local_entrypoint()
def main(output: str):
    destination = Path(output)
    if destination.exists():
        raise FileExistsError(destination)
    record = measure.remote()
    with destination.open("x") as stream:
        json.dump(record, stream, indent=2, allow_nan=False)
        stream.write("\n")
