# ABOUTME: Measures sparse analytic states with the original Lightning Pauli kernels and term order.
# ABOUTME: Omits terms that contribute exact zero in every populated bra coordinate.
from collections import OrderedDict
import ctypes
from functools import lru_cache
from hashlib import sha256
from importlib.metadata import PackageNotFoundError, version
import os
from pathlib import Path

import numpy as np

from core.base_vqe import recent_state_energies


@lru_cache(maxsize=1)
def lightning_thread_count():
    """Read Lightning's OpenMP runtime, or identify the inspected serial macOS arm64 build."""
    try:
        from pennylane_lightning import lightning_qubit_ops
        if version("pennylane-lightning") != "0.38.0":
            return None
        library = Path(lightning_qubit_ops.__file__)
        count = getattr(ctypes.CDLL(str(library)), "omp_get_max_threads", None)
    except (ImportError, OSError, PackageNotFoundError):
        return None
    if count is not None:
        count.restype = ctypes.c_int
        count.argtypes = []
        return count
    # This wheel instantiates HamiltonianApplyInPlace<StateVectorLQubitManaged<double>,false>.
    if sha256(library.read_bytes()).hexdigest() == "5b135d1d26722fb14ee7d2ecefea9cddc517f2ceca944c0241187470172e109d":
        return lambda: 1
    return None


def adaptive_state_measurement(n_qubits, observable):
    """Choose a bounded measurement path for a single analytic circuit and molecular observable."""
    if os.environ.get("OMP_NUM_THREADS") != "1":
        return None
    sentence = observable.pauli_rep
    if not sentence or len(sentence) < 64:
        return None
    threads = lightning_thread_count()
    reuse = recent_state_energies(n_qubits)
    if reuse is None or threads is None or threads() != 1:
        return None
    coeffs = np.asarray(list(sentence.values()))
    if not np.isrealobj(coeffs) or not np.isfinite(coeffs).all():
        return None
    # Keep every intermediate finite, even at positions where the final inner product multiplies by zero.
    if np.max(abs(coeffs)) > np.finfo(float).max / (8 * len(coeffs) * (1 << n_qubits)):
        return None
    masks = np.asarray([sum(1 << (n_qubits - 1 - w) for w, p in word.items() if p in ("X", "Y"))
                        for word in sentence], dtype=np.intp)
    subsets = OrderedDict()
    native_terms = None
    ordered = False

    def measure(state, expectation, full):
        nonlocal native_terms, ordered
        if threads() != 1:
            return expectation(full)
        if native_terms is None:
            from pennylane_lightning.core._serialize import QuantumScriptSerializer
            serializer = QuantumScriptSerializer("lightning.qubit", False)
            native_terms = full.get_ops() if isinstance(full, serializer.hamiltonian_obs) else []
            # Bind the mask order to the actual serialized observable, including duplicate-word consolidation.
            if native_terms and np.array_equal(full.get_coeffs(), coeffs) and len(native_terms) == len(sentence):
                ordered = all(native == serializer._pauli_word(word)
                              for native, word in zip(native_terms, sentence))
        if not ordered:
            return reuse(state, expectation, full)
        support = np.flatnonzero(state)
        if len(support) == 0 or len(support) > 16 or not np.all(abs(state[support]) <= 2):
            return reuse(state, expectation, full)
        transitions = tuple(np.unique(np.bitwise_xor.outer(support, support)))
        if transitions not in subsets:
            indices = np.flatnonzero(np.isin(masks, transitions))
            # Apply the selected operators on the full state with Lightning's unchanged dot product.
            # With one reduction thread, the omitted Pauli actions add exactly zero on the entire support.
            subset = full
            if 0 < len(indices) < len(masks):
                subset = type(full)(coeffs[indices], [native_terms[i] for i in indices])
            subsets[transitions] = subset
            if len(subsets) > 8:
                subsets.popitem(last=False)
        else:
            subsets.move_to_end(transitions)
        return reuse(state, expectation, subsets[transitions])

    return measure
