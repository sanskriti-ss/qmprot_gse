# ABOUTME: Propagates hydrogen density matrices with explicit Kraus maps on each supported noisy ansatz.
# ABOUTME: Checks channel strength, layer placement, density normalization and full-Hamiltonian scoring.
import contextlib
from functools import reduce
import io
from pathlib import Path
import unittest

import numpy as np
import pennylane as qml
from pyscf import lib

from active_space_truncation.run_pipeline import run_pipeline
from algorithms import ALGORITHMS
from core.backend_manager import BackendConfig
from experiments.noise_resilience import DEFAULT_NOISE_MODELS, LAYERED_ALGORITHMS, NOISE_ALGORITHMS
from smoke.physics_acceptance import dense_hamiltonian, PAULI


def channel_operators(model, strength):
    i, x, y, z = (PAULI[p] for p in "IXYZ")
    if model == "depolarizing":
        return [np.sqrt(1-strength)*i] + [np.sqrt(strength/3)*p for p in (x, y, z)]
    if model == "phaseflip":
        return [np.sqrt(1-strength)*i, np.sqrt(strength)*z]
    if model == "amplitude_damping":
        return [np.diag([1., np.sqrt(1-strength)]), np.array([[0., np.sqrt(strength)], [0., 0.]])]
    raise ValueError(model)


def density_score(solver, parameters, ham, model, strength, expected_layers):
    wires = list(range(solver.n_qubits))
    rho = np.zeros_like(ham)
    rho[0, 0] = 1.
    operators = channel_operators(model, strength) if model else []
    channel_name = {"phaseflip": "PhaseFlip", "depolarizing": "DepolarizingChannel",
                    "amplitude_damping": "AmplitudeDamping"}.get(model)
    solver.cost_fn.construct((parameters,), {})
    blocks = []
    adjacent = False
    for gate in solver.cost_fn.qtape.operations:
        if isinstance(gate, qml.operation.Channel):
            if gate.name != channel_name or float(gate.parameters[0]) != strength:
                raise ValueError(f"Unexpected channel {gate}")
            if not adjacent:
                blocks.append([])
            wire = int(gate.wires[0])
            blocks[-1].append(wire)
            kraus = [reduce(np.kron, [op if q == wire else PAULI["I"] for q in wires]) for op in operators]
            rho = sum(k @ rho @ k.conj().T for k in kraus)
            adjacent = True
        else:
            if gate.name == "StatePrep":
                state = np.asarray(gate.state_vector(wire_order=wires)).reshape(-1)
                rho = np.outer(state, state.conj())
            else:
                unitary = qml.matrix(gate, wire_order=wires)
                rho = unitary @ rho @ unitary.conj().T
            adjacent = False
    if blocks != [wires]*expected_layers or (blocks and not adjacent):
        raise ValueError(f"Incorrect channel coverage {blocks} or missing final channel block")
    np.testing.assert_allclose(np.trace(rho), 1., atol=1e-11, rtol=0.)
    np.testing.assert_allclose(rho, rho.conj().T, atol=1e-11, rtol=0.)
    if np.linalg.eigvalsh(rho)[0] < -1e-11:
        raise ValueError("Density matrix has a negative eigenvalue")
    return float(np.trace(ham @ rho).real)


class NoisePhysics(unittest.TestCase):
    def test_reported_energies_on_coherent_hydrogen_circuits(self):
        path = Path(__file__).resolve().parent / "geometries/hydrogen/hydrogen.h5"
        log = io.StringIO()
        with contextlib.redirect_stdout(log), lib.temporary_env(lib.StreamObject, stdout=log):
            problem = run_pipeline("hydrogen", basis="sto-3g", h5_path=str(path), quiet=True)
        h = problem["hamiltonian"].qubit_hamiltonian
        ham = dense_hamiltonian(h)
        for algorithm in NOISE_ALGORITHMS:
            for model in DEFAULT_NOISE_MODELS:
                with self.subTest(algorithm=algorithm, model=model):
                    backend = BackendConfig.noisy(h.n_qubits, model, 0.02)
                    solver = ALGORITHMS[algorithm](h, n_layers=2, random_seed=42, backend_config=backend)
                    solver.build_ansatz()
                    theta = np.linspace(-0.73, 0.51, solver.n_parameters)
                    layers = 2 if algorithm in LAYERED_ALGORITHMS else 1
                    score = density_score(solver, theta, ham, model, 0.02, layers)
                    self.assertAlmostEqual(score, float(solver.cost_function(theta)), places=10)
