# ABOUTME: Measures real statevectors with ordered molecular Pauli terms on single-thread Lightning.
# ABOUTME: Shares term indices across adjacent basis rows while preserving every rounded multiply and add.
from functools import lru_cache
from hashlib import sha256
import os
from pathlib import Path

import numpy as np
from scipy.sparse import csr_matrix

from core.hamiltonian_loader import parity_signs
from core.pauli_measurement import lightning_thread_count


@lru_cache(maxsize=1)
def _row_dtype():
    from pennylane_lightning import lightning_qubit_ops
    from scipy.sparse import _sparsetools

    # These wheel pairs round the term product separately from its addition. The arm64 real kernel uses FMA.
    formats = {
        ("5b135d1d26722fb14ee7d2ecefea9cddc517f2ceca944c0241187470172e109d",
         "9436d040c36250dfcd73335b75ccaf52c315469aab03fed3f02999f899398233"): np.dtype(np.complex128),
        ("d41ad298e9eaab23f5accbbc012f7731dfb459c61230ce11f96368e8c2bfc9be",
         "54d239dc364f96e6b82ff983fe9f2f71598af74b4f931b0f5198a8411b0375d3"): np.dtype(np.float64),
    }
    pair = tuple(sha256(Path(module.__file__).read_bytes()).hexdigest()
                 for module in (lightning_qubit_ops, _sparsetools))
    return formats.get(pair)


@lru_cache(maxsize=1)
def _ordered_rows(n_qubits, flip_bits, phase_bits, weights):
    flips = np.frombuffer(flip_bits, dtype=np.int32)
    phases = np.frombuffer(phase_bits, dtype=np.int32)
    coefficients = np.frombuffer(weights, dtype=np.float64)
    dimension = 1 << n_qubits
    width = 16
    count = dimension // width
    dtype = _row_dtype()
    if dtype is None:
        return None
    table_bytes = count * len(flips) * (4 + dtype.itemsize) + 4 * (count + 1)
    if table_bytes > 16 * 1024**2:
        return None
    patterns, slots = np.unique((flips % width) * width + phases % width, return_inverse=True)
    coordinates = dimension * len(patterns)
    if table_bytes + coordinates * 4 + len(patterns) * width > 16 * 1024**2 or coordinates * dtype.itemsize > 8 * 1024**2:
        return None
    blocks = np.arange(count, dtype=np.int32)[:, None]
    indices = ((blocks ^ (flips // width)) * len(patterns) + slots).astype(np.int32)
    parity = parity_signs(n_qubits)
    values = parity[blocks & (phases // width)] * coefficients
    offsets = np.arange(count + 1, dtype=np.int32) * len(flips)
    # Duplicate column positions are intentional. Every original term has its own addition.
    matrix = csr_matrix((values.ravel().astype(dtype, copy=False), indices.ravel(), offsets),
                        shape=(count, count * len(patterns)))
    lane = np.arange(width, dtype=np.int32)
    source = blocks[:, None, :] * width + ((patterns // width)[:, None] ^ lane)
    signs = parity[(patterns % width)[:, None] & lane]
    return matrix, source, signs


def _term_key(n_qubits, sentence, coefficients, native):
    from pennylane_lightning.core._serialize import QuantumScriptSerializer

    serializer = QuantumScriptSerializer("lightning.qubit", False)
    if not isinstance(native, serializer.hamiltonian_obs) or not np.array_equal(native.get_coeffs(), coefficients):
        return None
    terms = native.get_ops()
    if len(terms) != len(sentence) or any(t != serializer._pauli_word(w) for t, w in zip(terms, sentence)):
        return None
    flips, phases, weights = [], [], []
    for word, coefficient in sentence.items():
        flip = sum(1 << (n_qubits - wire - 1) for wire, p in word.items() if p in "XY")
        phase = sum(1 << (n_qubits - wire - 1) for wire, p in word.items() if p in "YZ")
        ys = (flip & phase).bit_count()
        # Odd-Y words map a real vector to an imaginary vector and add zero to its real component.
        if ys % 2 == 0:
            flips.append(flip)
            phases.append(phase)
            weights.append(-coefficient if ys % 4 else coefficient)
    if weights:
        return (n_qubits, np.asarray(flips, np.int32).tobytes(),
                np.asarray(phases, np.int32).tobytes(), np.asarray(weights).tobytes())
    return None


def ordered_measurement(n_qubits, observable):
    """Return the analytic real-state measurement when term order and arithmetic can be verified."""
    sentence = observable.pauli_rep
    if (not 6 <= n_qubits <= 12 or not sentence or len(sentence) < 64
            or os.environ.get("OMP_NUM_THREADS") != "1"):
        return None
    threads = lightning_thread_count()
    if threads is None or threads() != 1 or _row_dtype() is None:
        return None
    coeffs = np.asarray(list(sentence.values()))
    if (coeffs.dtype != np.float64 or not np.isfinite(coeffs).all()
            or np.max(abs(coeffs)) >= np.finfo(float).max / (8 * len(coeffs) * (1 << n_qubits))):
        return None
    checked = None
    key = None

    def measure(state, expectation, native):
        nonlocal checked, key
        if state.dtype != np.complex128 or threads() != 1 or state.imag.any() or not np.max(abs(state.real)) <= 2:
            return expectation(native)
        if native is not checked:
            key = _term_key(n_qubits, sentence, coeffs, native)
            checked = native
        plan = _ordered_rows(*key) if key is not None else None
        if plan is None:
            return expectation(native)
        matrix, source, signs = plan
        amplitudes = state.real if matrix.dtype.kind == "f" else state
        lanes = amplitudes[source]
        lanes *= signs
        applied = (matrix @ lanes.reshape(matrix.shape[1], -1)).ravel()
        # Lightning's real inner product begins at +0 and visits amplitudes in register order.
        products = applied.real * state.real
        products[0] += 0.
        return float(np.add.accumulate(products)[-1])

    return measure
