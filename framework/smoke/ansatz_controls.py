# ABOUTME: Probes the physical generator and preparation claims of registered molecular ansatzes.
# ABOUTME: Measures spin, charge, circuit dependence on the cost Hamiltonian and phase-only evolutions.
"""Run under framework with QMPROT_GEOMETRIES=smoke/geometries python -m smoke.ansatz_controls."""
import contextlib
from itertools import product
import io
import json
from pathlib import Path

import numpy as np
import pennylane as qml
from openfermion import FermionOperator, QubitOperator, get_sparse_operator, hermitian_conjugated, s_squared_operator
from pyscf import lib
from scipy.linalg import expm

from algorithms import ALGORITHMS
from core.hamiltonian_loader import Molecule, QubitHamiltonian
from experiments._common import load_small_hamiltonian
from smoke.physics_acceptance import circuit_state, dense_hamiltonian, leakage


def state_at(name, h, theta, **options):
    solver = ALGORITHMS[name](h, random_seed=42, **options)
    solver.build_ansatz()
    values = theta(solver.n_parameters)
    return circuit_state(solver, values, name), solver


def controls():
    with contextlib.redirect_stdout(io.StringIO()) as log, lib.temporary_env(lib.StreamObject, stdout=log):
        source = load_small_hamiltonian("hydrogen", "sto-3g", None)
    h = source.hamiltonian
    matrix = dense_hamiltonian(h)
    hf = np.eye(16)[:, 0b1100]
    data = json.loads((Path(__file__).parent / "reproductions/iqcc_glycine/glycine.json").read_text())
    molecule = Molecule("gly", "glycine", 4, 27, data["full_casci"], "")
    gly = QubitHamiltonian(molecule, np.array(data["coefficients"]), data["pauli_strings"], 4, 27,
                           cs_initial_state=np.array([complex(*z) for z in data["state"]]))
    rows = {}

    spin = get_sparse_operator(s_squared_operator(2), n_qubits=4).toarray()
    paired, _ = state_at("ucc_vqe", h, lambda n: np.array([np.pi/2, 0.]))
    excitation = FermionOperator("2^ 0") + FermionOperator("3^ 1")
    generator = get_sparse_operator(excitation - hermitian_conjugated(excitation), n_qubits=4).toarray()
    singlet = expm(np.pi/4 * generator) @ hf
    rows["paired_ucc_single"] = dict(qubit_gate_spin_squared=np.vdot(paired, spin @ paired).real,
        fermionic_spin_squared=np.vdot(singlet, spin @ singlet).real,
        qubit_gate_off_particle_sector=leakage(paired, 2))

    angles = lambda n: np.linspace(-0.73, 0.51, n)
    a, _ = state_at("qaoa_inspired_vqe", h, angles, n_layers=2)
    b, _ = state_at("qaoa_inspired_vqe", gly, angles, n_layers=2)
    rows["qaoa_equal_width"] = dict(state_max_error=float(np.max(np.abs(a-b))),
        hydrogen_score=float(np.vdot(a, matrix @ a).real),
        glycine_score=float(np.vdot(b, dense_hamiltonian(gly) @ b).real))
    _, mixer = state_at("qaoa_inspired_vqe", h, lambda n: np.array([0., 0., 0., 0., 0., 0.6]),
                        n_layers=1, mixer_type="XY")
    mixed = hf.copy()
    for gate in mixer.cost_fn.qtape.operations:
        if gate.name in ("RX", "RY"):
            mixed = qml.matrix(gate, wire_order=range(4)) @ mixed
    rows["qaoa_xy_mixer"] = dict(off_particle_sector=leakage(mixed, 2), beta=0.6)

    qubit_adapt = ALGORITHMS["qubit_adapt_vqe"](h)
    qubit_adapt.build_ansatz()
    generators = ["Y0 X2", "Y0 X3"]
    qubit_adapt.selected_operators = [qubit_adapt.operator_pool.index(g) for g in generators]
    mixed = circuit_state(qubit_adapt, np.full(2, np.pi/4), "qubit_adapt_vqe")
    cs_pool = ALGORITHMS["qubit_adapt_vqe"](gly)
    cs_pool.build_ansatz()
    rows["qubit_adapt"] = dict(hydrogen_pool=len(qubit_adapt.operator_pool), projected_pool=len(cs_pool.operator_pool),
        generators=generators, angle=float(np.pi/4), off_particle_sector=leakage(mixed, 2))

    inspired = ALGORITHMS["iqcc_inspired_vqe"](h)
    inspired.selected_operators = [QubitOperator("Y0")]
    inspired.parameters = np.zeros(1)
    inspired.build_ansatz()
    rows["iqcc_inspired_argument"] = dict(stored_angle=0., requested_angle=0.73,
        public_cost=inspired.cost_function(np.array([0.73])), actual_circuit=float(inspired.cost_fn(np.array([0.73]))))
    for name, settings in (("hardware_efficient_vqe", {"n_layers": 2}), ("pennylane_vqe", {"n_layers": 2}),
                           ("nn_ae_vqe", {"decoder_style": "mirror"}), ("nn_ae_vqe", {"decoder_style": "independent"})):
        psi, solver = state_at(name, h, np.zeros, **settings)
        rows[name + ("_" + settings["decoder_style"] if "decoder_style" in settings else "")] = dict(hf_fidelity=float(abs(np.vdot(hf, psi))**2),
                                          energy=float(np.vdot(psi, matrix @ psi).real),
                                          parameters=solver.n_parameters)

    _, autoencoder = state_at("nn_ae_vqe", h, angles, decoder_style="mirror",
                              n_enc_layers=1, encoder_style="ry_brick")
    encoded = hf.copy()
    # The published hydrogen determinant is prepared by two X gates before the RY wall.
    encoder_gates = autoencoder.cost_fn.qtape.operations[2:2 + h.n_qubits + len(autoencoder._enc_cnot_fwd)]
    for gate in encoder_gates:
        encoded = qml.matrix(gate, wire_order=range(h.n_qubits)) @ encoded
    tensor = encoded.reshape(2**autoencoder.n_latent, -1)
    latent_density = tensor @ tensor.conj().T
    rows["nn_ae_latent"] = dict(encoded_qubits=h.n_qubits, latent_qubits=autoencoder.n_latent,
                                purity=float(np.trace(latent_density @ latent_density).real))

    gly_prefix = QubitHamiltonian(molecule, gly.coefficients[:6], gly.pauli_strings[:6], 4, 6,
                                  cs_initial_state=gly.cs_initial_state)
    psi, solver = state_at("hva_vqe", gly_prefix, angles, n_layers=2)
    group_matrix = dense_hamiltonian(gly_prefix)
    rows["hva_diagonal_prefix"] = dict(groups=len(solver._commuting_groups), parameters=solver.n_parameters,
        energy_difference=float(np.vdot(psi, group_matrix @ psi).real -
                                np.vdot(gly.cs_initial_state, group_matrix @ gly.cs_initial_state).real))

    grouped, hva = state_at("hva_vqe", h, angles, n_layers=2)
    group_terms = [QubitHamiltonian.from_dict({p: c for c, p in group}, h.molecule)
                   for group in hva._commuting_groups]
    group_matrices = [dense_hamiltonian(group) for group in group_terms]
    state = hf.copy()
    for i, angle in enumerate(angles(hva.n_parameters)):
        state = expm(-1j*angle*group_matrices[i % len(group_matrices)]) @ state
    rows["hva_hydrogen_groups"] = dict(groups=len(group_matrices), terms=len(h.coefficients),
        hamiltonian_sum_error=float(np.max(np.abs(sum(group_matrices)-matrix))),
        dense_state_fidelity=float(abs(np.vdot(grouped, state))**2))

    gradients = {}
    for letters in product("IXYZ", repeat=4):
        weight = sum(x != "I" for x in letters)
        if weight not in (1, 2, 4):
            continue
        op = QubitOperator(tuple((i, p) for i, p in enumerate(letters) if p != "I"))
        p = get_sparse_operator(op, n_qubits=4).toarray()
        slope = abs(np.vdot(hf, .5j*(p @ matrix-matrix @ p) @ hf))
        gradients[weight] = max(gradients.get(weight, 0.), float(slope))
    rows["h2_initial_screen"] = dict(max_slope_by_pauli_weight=gradients,
        hf_minus_exact=float(np.vdot(hf, matrix @ hf).real-np.linalg.eigvalsh(matrix)[0]))
    return rows


if __name__ == "__main__":
    print(json.dumps(controls(), indent=2))
