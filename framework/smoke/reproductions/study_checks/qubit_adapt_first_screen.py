# ABOUTME: Times the native Qubit-ADAPT reference screen on one frozen molecular Hamiltonian.
# ABOUTME: Compares the full states, energies and gradients when paired probes share gate decomposition.
import argparse
from collections import Counter
from hashlib import sha256
import json
from pathlib import Path
from time import perf_counter

import numpy as np
import pennylane as qml
from pennylane_lightning.core._serialize import QuantumScriptSerializer
from pennylane_lightning.lightning_qubit_ops import MeasurementsC128

from algorithms.vqe_qubit_adapt import QubitAdaptVQE
from core.backend_manager import analytic_energy
from core.pauli_measurement import adaptive_state_measurement
from experiments._common import make_backend_config
from experiments.provenance import load_problem


def first_screen(problem, rebind):
    start = perf_counter()
    solver = QubitAdaptVQE(problem.hamiltonian, backend_config=make_backend_config(problem.n_qubits_final))
    solver.build_ansatz()
    circuit, observable = solver.cost_fn, solver._energy_observable
    measure = adaptive_state_measurement(solver.n_qubits, observable)
    if measure is None or solver.device.name != "lightning.qubit":
        raise ValueError("The control requires an eligible single-thread Lightning CPU backend")
    state_trace, energy_trace = sha256(), sha256()
    seen, supports = set(), Counter()

    def record(state, expectation, full):
        amplitudes = state.tobytes()
        seen.add(sha256(amplitudes).digest())
        state_trace.update(amplitudes)
        supports[int(np.count_nonzero(state))] += 1
        energy = measure(state, expectation, full)
        energy_trace.update(np.float64(energy).tobytes())
        return energy

    if rebind:
        device = solver.device
        state = device._statevector
        expval = MeasurementsC128(state.state_vector).expval
        serialized = QuantumScriptSerializer(device.name, False)._ob(observable)
        positive = None

        def probe(parameters):
            nonlocal positive
            operator, = solver.selected_operators
            angle, = parameters
            if angle > 0:
                circuit.construct((parameters,), {})
                circuit._update_gradient_fn(tape=circuit.qtape)
                program = qml.workflow.get_transform_program(circuit, level="device")
                (tape,), finish = program((circuit.qtape,))
                slot, = [i for i, gate in enumerate(tape.operations) if type(gate) is qml.MultiRZ]
                if tape.operations[slot].data[0].tobytes() != (2 * angle).tobytes():
                    raise ValueError("The decomposed MultiRZ angle differs from this Pauli rotation")
                positive = operator, angle, tape, slot, finish
            else:
                previous_operator, previous_angle, source, slot, finish = positive
                if (operator, angle) != (previous_operator, -previous_angle):
                    raise ValueError("Expected negative partner of the immediately preceding Pauli probe")
                gates = list(source.operations)
                gates[slot] = qml.MultiRZ(2 * angle, wires=gates[slot].wires)
                tape = qml.tape.QuantumScript(gates, source.measurements, shots=source.shots)
            state.reset_state()
            state.get_final_state(tape)
            return finish((record(state.state, expval, serialized),))[0]

        solver._measure_energy = probe
    else:
        solver._measure_energy = analytic_energy(circuit, observable, measure_state=record)
    before_screen = perf_counter()
    gradients = solver._compute_gradients()
    finish = perf_counter()
    return dict(mode="rebind" if rebind else "control", setup_and_screen_seconds=finish-start,
                screen_seconds=finish-before_screen, pool=len(solver.operator_pool), probes=sum(supports.values()),
                distinct_states=len(seen), nonzero_amplitudes=dict(supports), state_sha256=state_trace.hexdigest(),
                energy_sha256=energy_trace.hexdigest(), gradient_sha256=sha256(gradients.tobytes()).hexdigest(),
                selected_index=int(np.argmax(gradients)), gradient_max=float(np.max(gradients)))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("input", type=Path)
    parser.add_argument("output", type=Path)
    args = parser.parse_args()
    problem = load_problem(args.input)
    runs = [first_screen(problem, rebind) for rebind in (False, True, True, False)]
    science = [{key: value for key, value in run.items() if key != "mode" and not key.endswith("_seconds")}
               for run in runs]
    if not all(item == science[0] for item in science):
        raise ValueError("Scientific output differs between paired first screens")
    args.output.write_text(json.dumps(dict(input_sha256=sha256(args.input.read_bytes()).hexdigest(),
                                          qubits=problem.n_qubits_final, terms=problem.hamiltonian.n_terms,
                                          runs=runs), indent=2) + "\n")
