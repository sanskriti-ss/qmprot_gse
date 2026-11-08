# ABOUTME: Times each warmed stage of the 100-angle hardware-efficient glycine energy calculation.
# ABOUTME: Checks each instrumented scalar against the registered solver at the identical parameters.
import argparse
from hashlib import sha256
import json
from pathlib import Path
from time import perf_counter

import numpy as np
import pennylane as qml
from pennylane_lightning.core._serialize import QuantumScriptSerializer
from pennylane_lightning.lightning_qubit_ops import MeasurementsC128

from algorithms import get_algorithm
from experiments._common import make_backend_config
from experiments.provenance import load_problem


def run(path):
    problem = load_problem(path)
    solver = get_algorithm("hardware_efficient_vqe")(problem.hamiltonian, n_layers=8, max_free_parameters=100,
                prepare_hf=True, random_seed=42, backend_config=make_backend_config(12))
    solver.build_ansatz()
    circuit = solver.cost_fn
    device = circuit.device
    state = device._statevector
    measurement = MeasurementsC128(state.state_vector)
    vectors = np.random.default_rng(42).normal(0, .1, (65, 100))
    energies = []
    stages = dict(construct=0., transform=0., evolve=0., expectation=0.)
    for i, vector in enumerate(vectors):
        start = perf_counter()
        circuit.construct((vector,), {})
        built = perf_counter()
        circuit._update_gradient_fn(tape=circuit.qtape)
        program = qml.workflow.get_transform_program(circuit, level="device")
        tapes, finish = program((circuit.qtape,))
        if len(tapes) != 1:
            raise ValueError("Expected a single statevector circuit")
        if i == 0:
            serialized = QuantumScriptSerializer(device.name, False)._ob(tapes[0].measurements[0].obs)
        transformed = perf_counter()
        state.reset_state()
        state.get_final_state(tapes[0])
        evolved = perf_counter()
        energy = float(finish((measurement.expval(serialized),))[0])
        measured = perf_counter()
        if energy.hex() != float(solver.cost_function(vector)).hex():
            raise ValueError(f"Energy mismatch for vector {i}")
        if i:
            energies.append(energy)
            stages["construct"] += built - start
            stages["transform"] += transformed - built
            stages["evolve"] += evolved - transformed
            stages["expectation"] += measured - evolved
    return dict(molecule=problem.molecule, qubits=problem.hamiltonian.n_qubits, terms=problem.hamiltonian.n_terms,
                angles=solver.n_parameters, warm_evaluations=len(energies), input_sha256=sha256(path.read_bytes()).hexdigest(),
                energy_sha256=sha256(np.asarray(energies).tobytes()).hexdigest(), seconds=stages)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("input", type=Path)
    parser.add_argument("output", type=Path)
    args = parser.parse_args()
    args.output.write_text(json.dumps(run(args.input), indent=2) + "\n")
