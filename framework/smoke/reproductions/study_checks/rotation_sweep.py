# ABOUTME: Profiles registered fixed-angle Lightning circuits on real unreduced glycine.
# ABOUTME: Checks every warmed bound tape, state and energy against the original QNode on identical inputs.
import argparse
from hashlib import sha256
import json
from pathlib import Path
from statistics import median
from time import perf_counter

import numpy as np
import pennylane as qml

from algorithms import get_algorithm
from experiments._common import he_layers_for_target_params
from experiments.provenance import load_problem


def examine(hamiltonian, angles):
    layers, _ = he_layers_for_target_params(hamiltonian.n_qubits, angles, rotation_gates="RY", minimum_layers=1)
    solver = get_algorithm("hardware_efficient_vqe")(hamiltonian, n_layers=layers, max_free_parameters=angles)
    circuit = solver.build_ansatz()
    initial = np.random.default_rng(42).normal(0., .1, size=(33, angles))
    circuit.construct((initial[0],), {})
    template = circuit.qtape
    indices = [i for i, info in enumerate(template.par_info) if type(info["op"]) is qml.RY]
    if len(indices) != angles:
        raise ValueError("Unexpected number of independent rotation angles")
    setup = {"construct_and_transform": [], "bind": []}
    elapsed = []
    states, energies = sha256(), sha256()
    for index, vector in enumerate(initial):
        order = tuple(setup) if index % 2 else tuple(reversed(setup))
        tapes = {}
        for name in order:
            start = perf_counter()
            if name == "bind":
                tape = template.bind_new_parameters(vector, indices)
            else:
                circuit.construct((vector,), {})
                circuit._update_gradient_fn(tape=circuit.qtape)
                (tape,), _ = qml.workflow.get_transform_program(circuit, level="device")((circuit.qtape,))
                if tape is not circuit.qtape:
                    raise ValueError("Native transform changed the hardware-efficient tape")
            seconds = perf_counter() - start
            tapes[name] = tape
            if index:
                setup[name].append(seconds)
        if not qml.equal(*tapes.values()):
            raise ValueError("Bound rotation tape differs from the native transformed tape")
        start = perf_counter()
        energy = solver.cost_function(vector)
        seconds = perf_counter() - start
        state = solver.device._statevector.state.tobytes()
        if energy.hex() != float(circuit(vector)).hex() or state != solver.device._statevector.state.tobytes():
            raise ValueError("Energy or state differs from the registered QNode")
        if index:
            energies.update(np.float64(energy).tobytes())
            states.update(state)
            elapsed.append(seconds)
    return dict(angles=angles, layers=layers, warmed_vectors=len(elapsed), energy_sha256=energies.hexdigest(),
                state_sha256=states.hexdigest(), median_objective_seconds=median(elapsed),
                median_setup_seconds={name: median(values) for name, values in setup.items()})


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("input", type=Path)
    parser.add_argument("output", type=Path)
    args = parser.parse_args()
    hamiltonian = load_problem(args.input).hamiltonian
    receipt = dict(qubits=hamiltonian.n_qubits, terms=hamiltonian.n_terms, input_sha256=sha256(args.input.read_bytes()).hexdigest(),
                   cases=[examine(hamiltonian, angles) for angles in (10, 50, 100)])
    args.output.write_text(json.dumps(receipt, indent=2, allow_nan=False) + "\n")
