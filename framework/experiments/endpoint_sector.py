# ABOUTME: Rescores variational endpoints and measures electron number and spin in the parent orbitals.
# ABOUTME: Reconstructs contextual and Ritz states and keeps sector diagnostics outside fit budgets.
from copy import copy

import numpy as np
import pennylane as qml
from scipy import sparse

from contextual_subspace.core import pauli_to_sparse


def _state(vqe, angles, selected_operators):
    if selected_operators is not None and list(selected_operators) != vqe.selected_operators:
        if vqe.name not in ("adapt_vqe", "qubit_adapt_vqe"):
            raise ValueError("An explicit pool selection is only supported for ADAPT endpoints")
        vqe = copy(vqe)
        vqe.selected_operators = list(selected_operators)
    if vqe.name == "adapt_vqe":
        circuit = vqe._build_circuit(angles)
    elif vqe.name == "qubit_adapt_vqe":
        circuit = vqe._build_circuit()
    else:
        circuit = vqe.cost_fn
    circuit.construct((angles,), {})
    operations = circuit.qtape.operations
    device = qml.device("default.mixed" if vqe.backend_config.is_noisy else "default.qubit", wires=vqe.n_qubits)

    @qml.qnode(device, interface=None, diff_method=None)
    def read_state():
        for operation in operations:
            qml.apply(operation)
        return qml.state()

    return np.asarray(read_state())


def _reference_ket(vqe):
    device = qml.device("default.qubit", wires=vqe.n_qubits)

    @qml.qnode(device)
    def reference():
        vqe._prepare_initial_state()
        return qml.state()

    return np.asarray(reference())


def _ritz_state(trial, reference, hamiltonian):
    """Return a normalized lowest-energy representative in the CB solver's two-state span."""
    basis, triangular = np.linalg.qr(np.column_stack((reference, trial)))
    if abs(triangular[1, 1]) <= np.finfo(float).eps * max(basis.shape):
        return reference, False
    projected = basis.conj().T @ (hamiltonian @ basis)
    energies, vectors = np.linalg.eigh(projected)
    degenerate = bool(energies[1] - energies[0] <= np.finfo(float).eps * max(1., abs(energies[0])) * 4)
    return basis @ vectors[:, 0], degenerate


def _expectation(matrix, state):
    if state.ndim == 1:
        return float(np.vdot(state, matrix @ state).real)
    entries = matrix.tocoo()
    return float(np.dot(entries.data, state[entries.col, entries.row]).real)


def _sector_indices(metadata, n, dimension):
    indices = np.arange(1 << n)
    keep = np.ones(len(indices), dtype=bool)
    for constraint in metadata["stabilizers"]:
        word = constraint["pauli"]
        eigenvalue = constraint["eigenvalue"]
        if len(word) != n or word.count("Z") != 1 or set(word) - {"I", "Z"} or eigenvalue not in (-1, 1):
            raise ValueError(f"Invalid diagonal stabilizer {constraint}")
        bit = n - 1 - word.index("Z")
        keep &= ((indices >> bit) & 1) == int(eigenvalue < 0)
    kept = indices[keep]
    if len(kept) != dimension:
        raise ValueError("Stabilizer sector dimension differs from the reduced state")
    return kept


def _undo_rotations(amplitudes, metadata, n):
    for rotation in reversed(metadata["rotations"]):
        word, angle = rotation["pauli"], rotation["angle_rad"]
        if len(word) != n or set(word) - set("IXYZ") or not np.isfinite(angle):
            raise ValueError(f"Invalid projection rotation {rotation}")
        amplitudes = (np.cos(angle / 2) * amplitudes
                      - 1j * np.sin(angle / 2) * (pauli_to_sparse(word) @ amplitudes))
    return amplitudes


def _mapped_diagonal(mapping, density):
    """Evaluate diag(W rho W†) in blocks without constructing a parent-space density matrix."""
    output = np.empty(mapping.shape[0])
    for start in range(0, len(output), 64):
        block = mapping[start:start + 64]
        values = block.conjugate().multiply(block @ density)
        output[start:start + block.shape[0]] = np.asarray(values.sum(axis=1)).real.ravel()
    return output


def _spin_raising(n, basis):
    # Alpha and beta are adjacent, interleaved JW modes. The partner swap has positive fermionic sign.
    rows, columns = [], []
    for alpha_wire in range(0, n, 2):
        alpha = 1 << (n - 1 - alpha_wire)
        beta = alpha >> 1
        occupied_beta = basis[((basis & alpha) == 0) & ((basis & beta) != 0)]
        rows.extend(occupied_beta ^ (alpha | beta))
        columns.extend(occupied_beta)
    return sparse.csr_matrix((np.ones(len(rows)), (rows, columns)), shape=(len(basis), len(basis)))


def _physical_observables(loaded, state):
    prep = loaded.preparation or {}
    electrons = prep.get("active_electrons", loaded.hamiltonian.molecule.n_electrons)
    spin = prep.get("spin", loaded.hamiltonian.molecule.spin)
    meta = loaded.cs_metadata or {}
    result = dict(expected_electron_count=electrons, electron_count_probabilities=None,
                  wrong_electron_probability=None, mean_electron_count=None, mean_spin_z=None,
                  target_spin_z_probability=None, mean_spin_squared=None)
    n = loaded.n_qubits_active
    mapping = None
    if loaded.cs_reduced:
        if any(key not in meta for key in ("rotations", "stabilizers", "original_qubits", "reduced_qubits")):
            return {**result, "sector_status": "missing_parent_transform"}
        if meta["original_qubits"] != n or meta["reduced_qubits"] != loaded.n_qubits_final:
            raise ValueError("Contextual transform dimensions disagree with the prepared Hamiltonian")
        kept = _sector_indices(meta, n, len(state))
        if state.ndim == 1:
            parent = np.zeros(1 << n, dtype=complex)
            parent[kept] = state
            state = _undo_rotations(parent, meta, n)
        else:
            embedding = sparse.csr_matrix((np.ones(len(kept)), (kept, np.arange(len(kept)))),
                                          shape=(1 << n, len(kept)))
            mapping = _undo_rotations(embedding, meta, n).tocsr()
    elif len(state) != 1 << n:
        raise ValueError("Endpoint dimension differs from the unreduced orbital space")

    if mapping is not None:
        probabilities = _mapped_diagonal(mapping, state)
    elif state.ndim == 1:
        probabilities = abs(state)**2
    else:
        probabilities = np.diag(state).real
    if np.min(probabilities) < -1e-10 or not np.isclose(probabilities.sum(), 1., rtol=0, atol=1e-9):
        raise ValueError("Endpoint populations are negative or unnormalized")
    probabilities = np.maximum(probabilities, 0.)
    probabilities /= probabilities.sum()
    basis = np.arange(len(probabilities))
    population = np.fromiter((i.bit_count() for i in range(len(basis))), dtype=int)
    weights = np.bincount(population, weights=probabilities, minlength=n + 1)
    result.update(electron_count_probabilities={str(i): float(p) for i, p in enumerate(weights)},
                  mean_electron_count=float(population @ probabilities))
    if electrons is not None:
        if not 0 <= electrons <= n:
            raise ValueError("Target active electron count lies outside the parent register")
        result["wrong_electron_probability"] = float(probabilities[population != electrons].sum())
    if n % 2 == 0:
        alpha_bits = sum(1 << (n - 1 - wire) for wire in range(0, n, 2))
        n_alpha = np.fromiter(((i & alpha_bits).bit_count() for i in range(len(basis))), dtype=int)
        sz = n_alpha - 0.5 * population
        result["mean_spin_z"] = float(sz @ probabilities)
        if spin is not None:
            result["target_spin_z_probability"] = float(probabilities[2 * sz == spin].sum())
        raising = _spin_raising(n, basis)
        if mapping is not None:
            raising = raising @ mapping
        if state.ndim == 1:
            raised = raising @ state
            raising_norm = float(np.vdot(raised, raised).real)
        else:
            raising_norm = _expectation(raising.conj().T @ raising, state)
        result["mean_spin_squared"] = float((sz * (sz + 1)) @ probabilities + raising_norm)
    result["sector_status"] = "measured" if electrons is not None else "unknown_target_electron_count"
    return result


def summarize_endpoint(loaded, vqe, parameters, selected_operators=None):
    """Replay a fitted circuit analytically. All probabilities refer to active parent spin orbitals."""
    state = _state(vqe, np.asarray(parameters, dtype=float), selected_operators)
    ham = vqe.hamiltonian
    matrix = ham.ordered_sparse_matrix()
    result = {"state_kind": "statevector" if state.ndim == 1 else "density_matrix"}
    if vqe.name == "cb_vqe":
        if state.ndim != 1:
            raise ValueError("Classical boosting requires an analytic pure state")
        state, degenerate = _ritz_state(state, _reference_ket(vqe), matrix)
        result.update(state_kind="ritz_state", ritz_ground_degenerate=degenerate)
    return {**result, "rescored_energy": _expectation(matrix, state), **_physical_observables(loaded, state)}


def model_reference(loaded):
    """Distance to the range of energy expectations over all states of the actual operator."""
    ham = loaded.hamiltonian
    casci = loaded.casci_energy
    energy = offset = distance = None
    status = "qubit_limit"
    if ham.n_qubits <= 18:
        # A CSR matrix has at most one nonzero per row for each distinct bit
        # flip pattern in the Pauli sum. Avoid building a matrix whose upper
        # bound would exceed 1 GiB, even if its qubit count is under the cap.
        from core.hamiltonian_loader import pauli_masks

        dimension = 1 << ham.n_qubits
        flips = set()
        for raw_word in ham.pauli_strings:
            word = str(raw_word)
            if '(' in word or word == "Identity":
                word = ham._openfermion_to_pauli_string(word, ham.n_qubits)
            if len(word) == ham.n_qubits and set(word) <= set("IXYZ"):
                flips.add(pauli_masks(word)[0])
        estimated_bytes = dimension * min(dimension, len(flips)) * 24
        if estimated_bytes > 1024 ** 3:
            status = "estimated_memory_limit"
        else:
            status = "computed"
            matrix = ham.ordered_sparse_matrix()
            energy = ham._sparse_ground_state_energy(matrix, ham.n_qubits)
            if casci is not None:
                offset = energy - casci
                distance = max(0., offset)
                # A computational-basis expectation above CASCI already witnesses the upper side.
                if offset < 0 and matrix.diagonal().real.max() < casci:
                    upper = -ham._sparse_ground_state_energy(-matrix, ham.n_qubits)
                    distance = max(0., casci - upper)
    return dict(operator_ground_energy=energy, casci_energy=casci, operator_ground_minus_casci=offset,
                minimum_possible_abs_error_to_casci=distance, operator_ground_status=status,
                encoded_reference_energy=loaded.encoded_reference_energy)
