# ABOUTME: Records the parameters, energies and operator sequence evaluated by the experiment CLI.
# ABOUTME: Keeps the optimizer's logical requests and callbacks visible in saved benchmark comparisons.
import hashlib
import json
import math
import os
import sys
from pathlib import Path

import numpy as np


def dataset_names(experiment):
    load = experiment.load_small_hamiltonian
    names = {}

    def load_geometry(*args, **kwargs):
        problem = load(*args, **kwargs)
        names[id(problem.hamiltonian)] = problem.molecule
        return problem

    experiment.load_small_hamiltonian = load_geometry
    return names


def record(vqe, output, case, callback):
    case["workload"] = os.environ["QMPROT_WORKLOAD"]
    evaluate = vqe.cost_function

    def cost(parameters):
        value = evaluate(parameters)
        energy = float(value)
        if not math.isfinite(energy):
            raise ValueError(f"Nonfinite objective in {case}")
        parameters = np.asarray(parameters, dtype="<f8")
        event = dict(case, callback=bool(callback()), energy=energy.hex(),
                     parameters=hashlib.sha256(parameters.tobytes()).hexdigest(),
                     operators=[int(i) for i in getattr(vqe, "selected_operators", ())])
        output.write(json.dumps(event, separators=(",", ":")) + "\n")
        return value

    vqe.cost_function = cost


def trainability(output):
    from experiments import trainability as experiment

    names = dataset_names(experiment)
    counter = experiment.CostEvalCounter
    attach = counter.attach

    def attach_recording(self, vqe):
        target = vqe.max_operators if vqe.name == "qubit_adapt_vqe" else vqe.n_parameters
        case = dict(molecule=names[id(vqe.hamiltonian)], algorithm=vqe.name, target=int(target))
        record(vqe, output, case, lambda: self._in_callback)
        attach(self, vqe)

    counter.attach = attach_recording
    experiment.main()


def noise(output):
    from experiments import noise_resilience as experiment

    names = dataset_names(experiment)
    build = experiment.build_vqe

    def build_recording(*args, **kwargs):
        vqe = build(*args, **kwargs)
        backend = vqe.backend_config
        case = dict(molecule=names[id(vqe.hamiltonian)], algorithm=vqe.name,
                    noise_model=backend.noise_model, strength=float(backend.noise_strength or 0.0))
        in_callback = False
        original_callback = vqe.callback

        def callback(*args, **kwargs):
            nonlocal in_callback
            in_callback = True
            try:
                return original_callback(*args, **kwargs)
            finally:
                in_callback = False

        vqe.callback = callback
        record(vqe, output, case, lambda: in_callback)
        return vqe

    experiment.build_vqe = build_recording
    experiment.main()


if __name__ == "__main__":
    module, destination = sys.argv[1:3]
    sys.argv[:] = [module, *sys.argv[3:]]
    with Path(destination).open("x") as output:
        {"experiments.noise_resilience": noise, "experiments.trainability": trainability}[module](output)
