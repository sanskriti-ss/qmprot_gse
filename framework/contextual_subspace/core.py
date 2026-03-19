"""
Contextual Subspace VQE - Core Implementation

Implements quasi-quantized models for noncontextual Hamiltonians,
noncontextual ground state finding, and contextual subspace approximations.
Reference: https://arxiv.org/pdf/2002.05693.pdf
"""

import numpy as np
import scipy as sp
from scipy.optimize import minimize_scalar
from scipy.sparse import coo_matrix
import itertools
from functools import reduce
from copy import deepcopy
from datetime import datetime, timedelta
from typing import Dict, List, Tuple, Optional, Any


# -----------------------------------------------------------------------------
# Pauli algebra
# -----------------------------------------------------------------------------

def commute(x: str, y: str) -> int:
    """Check if two Pauli operators (as strings e.g. 'XIZYZ') commute. Returns 1 if commute, 0 if anticommute."""
    assert len(x) == len(y), f"Length mismatch: {x} vs {y}"
    s = 1
    for i in range(len(x)):
        if x[i] != 'I' and y[i] != 'I' and x[i] != y[i]:
            s = s * (-1)
    return 1 if s == 1 else 0


def contextualQ(S: List[str], verbose: bool = False) -> Any:
    """Check if a set S of Pauli strings is contextual. If verbose, return [is_contextual, Z, T]."""
    T = []
    Z = []
    for i in range(len(S)):
        if any(not commute(S[i], S[j]) for j in range(len(S))):
            T.append(S[i])
        else:
            Z.append(S[i])
    for i in range(len(T)):
        for j in range(len(T)):
            for k in range(j, len(T)):
                if i != j and i != k and commute(T[i], T[j]) and commute(T[i], T[k]) and not commute(T[j], T[k]):
                    return [True, None, None] if verbose else True
    return [False, Z, T] if verbose else False


def contextualQ_ham(ham: Dict[str, float], verbose: bool = False) -> Any:
    """Check if a Hamiltonian (dict of Pauli -> coeff) is contextual."""
    S = list(ham.keys())
    T = []
    Z = []
    for i in range(len(S)):
        if any(not commute(S[i], S[j]) for j in range(len(S))):
            T.append(S[i])
        else:
            Z.append(S[i])
    for i in range(len(T)):
        for j in range(len(T)):
            for k in range(j, len(T)):
                if i != j and i != k and commute(T[i], T[j]) and commute(T[i], T[k]) and not commute(T[j], T[k]):
                    return True
    return [False, Z, T] if verbose else False


def pauli_mult(p: str, q: str) -> Tuple[str, complex]:
    """Multiply two Pauli strings; return (result_string, sign) where p*q == sign * result."""
    assert len(p) == len(q)
    sgn = 1
    out = ''
    for i in range(len(p)):
        if p[i] == 'I':
            out += q[i]
        elif q[i] == 'I':
            out += p[i]
        elif p[i] == 'X':
            if q[i] == 'X':
                out += 'I'
            elif q[i] == 'Y':
                out += 'Z'
                sgn = sgn * 1j
            elif q[i] == 'Z':
                out += 'Y'
                sgn = sgn * -1j
        elif p[i] == 'Y':
            if q[i] == 'Y':
                out += 'I'
            elif q[i] == 'Z':
                out += 'X'
                sgn = sgn * 1j
            elif q[i] == 'X':
                out += 'Z'
                sgn = sgn * -1j
        elif p[i] == 'Z':
            if q[i] == 'Z':
                out += 'I'
            elif q[i] == 'X':
                out += 'Y'
                sgn = sgn * 1j
            elif q[i] == 'Y':
                out += 'X'
                sgn = sgn * -1j
    return (out, sgn)


# -----------------------------------------------------------------------------
# Independent generating set for commuting Paulis
# -----------------------------------------------------------------------------

def to_indep_set(G_w_in: Dict[str, List]) -> Tuple[List, Dict]:
    """Given a commuting set of Pauli strings (dict mapping each to None),
    return an independent generating set and the mapping from original elements
    to their equivalent product in the new set."""
    G_w = deepcopy(G_w_in)
    G_w_keys = [[str(g), 1] for g in G_w.keys()]
    G_w_keys_orig = [str(g) for g in G_w.keys()]
    generators = []

    for i in range(len(G_w_keys[0][0])):
        fx = fy = fz = None
        j = 0
        while fx is None and j < len(G_w_keys):
            if G_w_keys[j][0][i] == 'X' and not any(G_w_keys[j][0] == g[0] for g in generators):
                fx = G_w_keys[j]
            j += 1
        j = 0
        while fy is None and j < len(G_w_keys):
            if G_w_keys[j][0][i] == 'Y' and not any(G_w_keys[j][0] == g[0] for g in generators):
                fy = G_w_keys[j]
            j += 1
        j = 0
        while fz is None and j < len(G_w_keys):
            if G_w_keys[j][0][i] == 'Z' and not any(G_w_keys[j][0] == g[0] for g in generators):
                fz = G_w_keys[j]
            j += 1

        if fx is not None:
            generators.append(fx)
            for j in range(len(G_w_keys)):
                if G_w_keys[j][0][i] == 'X':
                    G_w[G_w_keys_orig[j]] = G_w[G_w_keys_orig[j]] + [fx]
                    sgn = G_w_keys[j][1] * fx[1]
                    mult_res = pauli_mult(G_w_keys[j][0], fx[0])
                    G_w_keys[j] = [mult_res[0], G_w_keys[j][1] * sgn * mult_res[1]]

        if fz is not None:
            generators.append(fz)
            for j in range(len(G_w_keys)):
                if G_w_keys[j][0][i] == 'Z':
                    G_w[G_w_keys_orig[j]] = G_w[G_w_keys_orig[j]] + [fz]
                    sgn = G_w_keys[j][1] * fz[1]
                    mult_res = pauli_mult(G_w_keys[j][0], fz[0])
                    G_w_keys[j] = [mult_res[0], G_w_keys[j][1] * sgn * mult_res[1]]

        if fx is not None and fz is not None:
            for j in range(len(G_w_keys)):
                if G_w_keys[j][0][i] == 'Y':
                    G_w[G_w_keys_orig[j]] = G_w[G_w_keys_orig[j]] + [fx, fz]
                    sgn = G_w_keys[j][1] * fx[1]
                    mult_res = pauli_mult(G_w_keys[j][0], fx[0])
                    G_w_keys[j] = [mult_res[0], G_w_keys[j][1] * sgn * mult_res[1]]
                    sgn = G_w_keys[j][1] * fz[1]
                    mult_res = pauli_mult(G_w_keys[j][0], fz[0])
                    G_w_keys[j] = [mult_res[0], G_w_keys[j][1] * sgn * mult_res[1]]
        elif fy is not None:
            generators.append(fy)
            for j in range(len(G_w_keys)):
                if G_w_keys[j][0][i] == 'Y':
                    G_w[G_w_keys_orig[j]] = G_w[G_w_keys_orig[j]] + [fy]
                    sgn = G_w_keys[j][1] * fy[1]
                    mult_res = pauli_mult(G_w_keys[j][0], fy[0])
                    G_w_keys[j] = [mult_res[0], G_w_keys[j][1] * sgn * mult_res[1]]

    for j in range(len(G_w_keys)):
        G_w[G_w_keys_orig[j]] = G_w[G_w_keys_orig[j]] + [G_w_keys[j]]

    return generators, G_w


# -----------------------------------------------------------------------------
# Quasi-quantized model for noncontextual Hamiltonians
# -----------------------------------------------------------------------------

def quasi_model(ham_dict: Dict[str, float]) -> Tuple[List, List, Dict]:
    """Build quasi-quantized model for a noncontextual Hamiltonian.
    Returns [G, Ci1s, all_mappings] where G is universally commuting generators,
    Ci1s are anticommuting representatives, and all_mappings maps Hamiltonian
    terms to their decomposition."""
    terms = [str(k) for k in ham_dict.keys()]
    check = contextualQ(terms, verbose=True)
    assert not check[0], "Hamiltonian must be noncontextual"
    Z = check[1]
    T = check[2]

    C = []
    while T:
        C.append([T.pop()])
        for i in range(len(T) - 1, -1, -1):
            t = T[i]
            if commute(C[-1][0], t):
                C[-1].append(t)
                T.remove(t)

    Gprime = [[z, 1] for z in Z]
    Ci1s = []
    for Cii in C:
        Ci = list(Cii)
        Ci1 = Ci.pop()
        Ci1s.append(Ci1)
        for c in Ci:
            mult = pauli_mult(c, Ci1)
            Gprime.append([mult[0], mult[1]])

    G_p = dict.fromkeys([g[0] for g in Gprime], [])
    G, G_mappings = to_indep_set(G_p)

    G = list(dict.fromkeys([g[0] for g in G]))
    i = len(G) - 1
    while i >= 0:
        if all(G[i][j] == 'I' for j in range(len(G[i]))):
            del G[i]
        i -= 1

    Gprime_str = list(dict.fromkeys([g[0] for g in Gprime]))
    for g in list(G_mappings.keys()):
        ps = G_mappings[g]
        sgn = int(np.real(np.prod([p[1] for p in ps])))
        ps = [[p[0] for p in ps], sgn]
        i = len(ps[0]) - 1
        while i >= 0:
            if all(ps[0][i][j] == 'I' for j in range(len(ps[0][i]))):
                del ps[0][i]
            i -= 1
        G_mappings[g] = ps

    all_mappings = dict.fromkeys(terms)
    for z in Z:
        mapping = G_mappings[z]
        all_mappings[z] = [mapping[0]] + [[]] + [mapping[1]]

    for Ci1 in Ci1s:
        all_mappings[Ci1] = [[], [Ci1], 1]

    for i in range(len(C)):
        Ci = C[i]
        Ci1 = Ci1s[i]
        for Cij in Ci:
            mult = pauli_mult(Cij, Ci1)
            if mult[0] in G_mappings:
                mapping = G_mappings[mult[0]]
                all_mappings[Cij] = [mapping[0]] + [[Ci1]] + [float(np.real(mult[1] * mapping[1]))]
            else:
                # Fallback when commuting part is identity or not in generating set
                all_mappings[Cij] = [[], [Ci1], float(np.real(mult[1]))]

    return G, Ci1s, all_mappings


def energy_function_form(ham_dict: Dict[str, float], model: Tuple) -> List:
    """Build energy function form from Hamiltonian and quasi-model."""
    terms = [str(k) for k in ham_dict.keys()]
    q = model[0]
    r = model[1]
    out = []
    for t in terms:
        mappings = model[2][t]
        coeff = ham_dict[t] * mappings[2]
        q_indices = [q.index(qi) for qi in mappings[0]]
        r_indices = [r.index(ri) for ri in mappings[1]]
        out.append([coeff, q_indices, r_indices, t])
    return [len(q), len(r), out]


def energy_function(fn_form: List):
    """Create energy function from fn_form."""
    dim_q = fn_form[0]

    def evalfn(*args):
        total = 0.0
        for t in fn_form[2]:
            if len(t[1]) == 0 and len(t[2]) == 0:
                total += t[0]
            elif len(t[1]) > 0 and len(t[2]) == 0:
                total += t[0] * reduce(lambda x, y: x * y, [args[i] for i in t[1]])
            elif len(t[1]) == 0 and len(t[2]) > 0:
                total += t[0] * reduce(lambda x, y: x * y, [args[dim_q + i] for i in t[2]])
            else:
                total += (
                    t[0]
                    * reduce(lambda x, y: x * y, [args[i] for i in t[1]])
                    * reduce(lambda x, y: x * y, [args[dim_q + i] for i in t[2]])
                )
        return np.real(total)

    return evalfn


def angular(args: Tuple) -> Tuple:
    """Unit vector in spherical coordinates from angles."""
    if len(args) == 1:
        return (np.cos(args[0]), np.sin(args[0]))
    return (np.cos(args[0]), *[np.sin(args[0]) * a for a in angular(args[1:])])


def find_gs_noncon(
    ham_noncon: Dict[str, float],
    method: str = 'differential_evolution',
    model: Optional[Tuple] = None,
    fn_form: Optional[List] = None,
    energy: Optional[callable] = None,
    timer: bool = False,
    return_all: bool = False,
) -> Any:
    """Find noncontextual ground state via numerical minimization.

    If ``return_all=True``, also return all candidate ep_states sorted by
    energy as a second return value.  Callers can iterate through these to
    find an alternative sector compatible with the HF initial state.
    """
    if model is None:
        model = quasi_model(ham_noncon)

    start_time = datetime.now()

    if fn_form is None:
        fn_form = energy_function_form(ham_noncon, model)

    if energy is None:
        energy = energy_function(fn_form)

    bounds = [(0, np.pi) for _ in range(fn_form[1] - 2)] + [(0, 2 * np.pi)]
    best_guesses = []

    if fn_form[1] == 0:
        for q in itertools.product([1, -1], repeat=fn_form[0]):
            best_guesses.append([energy(*q), [list(q), []]])

    elif fn_form[1] == 2:
        for q in itertools.product([1, -1], repeat=fn_form[0]):
            sol = minimize_scalar(lambda x: energy(*q, np.cos(x), np.sin(x)))
            best_guesses.append([sol['fun'], [list(q), [np.cos(sol['x']), np.sin(sol['x'])]]])

    elif fn_form[1] > 2:
        for q in itertools.product([1, -1], repeat=fn_form[0]):
            if method == 'shgo':
                sol = sp.optimize.shgo(lambda x: energy(*q, *angular(tuple(x))), bounds)
            elif method == 'dual_annealing':
                sol = sp.optimize.dual_annealing(lambda x: energy(*q, *angular(tuple(x))), bounds)
            elif method == 'basinhopping':
                sol = sp.optimize.basinhopping(lambda x: energy(*q, *angular(tuple(x))), bounds)
            elif method == 'shgo_sobol':
                sol = sp.optimize.shgo(lambda x: energy(*q, *angular(tuple(x))), bounds, n=200, iters=5, sampling_method='sobol')
            else:
                sol = sp.optimize.differential_evolution(lambda x: energy(*q, *angular(tuple(x))), bounds)
            best_guesses.append([sol['fun'], [list(q), list(angular(sol['x']))]])

    best = min(best_guesses, key=lambda x: x[0])

    all_sorted = sorted(best_guesses, key=lambda x: x[0])

    if return_all:
        if timer:
            return best + [model, fn_form], all_sorted, datetime.now() - start_time
        return best + [model, fn_form], all_sorted

    if timer:
        return best + [model, fn_form], datetime.now() - start_time
    return best + [model, fn_form]


# -----------------------------------------------------------------------------
# Diagonalization and rotations
# -----------------------------------------------------------------------------

def diagonalize_epistemic(
    model: Tuple, fn_form: List, ep_state: List
) -> Tuple[List, List, np.ndarray]:
    """Return rotation sequence and diagonalized generators for epistemic state."""
    assert len(ep_state[0]) == fn_form[0]
    assert len(model[0]) == fn_form[0]
    assert len(ep_state[1]) == fn_form[1]
    assert len(model[1]) == fn_form[1]

    rotations = []

    if fn_form[1] > 0:
        for i in range(1, fn_form[1]):
            theta = np.arctan2(ep_state[1][i], np.sqrt(sum(ep_state[1][j] ** 2 for j in range(i))))
            if i == 1 and ep_state[1][0] < 0:
                theta = np.pi - theta
            generator = pauli_mult(model[1][0], model[1][i])
            sgn = generator[1].imag
            rotations.append([sgn * theta, generator[0]])

        GuA = deepcopy(model[0] + [model[1][0]])
        ep_state_trans = deepcopy(ep_state[0] + [1])
    else:
        GuA = deepcopy(model[0])
        ep_state_trans = deepcopy(ep_state[0])

    for i in range(len(GuA)):
        g = GuA[i]

        if not any((all(g[k] == 'I' or k == j for k in range(len(g))) and g[j] == 'Z') for j in range(len(g))):

            if all(p == 'I' or p == 'Z' for p in g):
                Zs = []
                for m in range(len(g)):
                    if g[m] == 'Z' and all(h[m] == 'I' for h in GuA[:i]):
                        Zs.append(m)
                assert len(Zs) > 0
                m = Zs[0]
                K = ''.join('Y' if o == m else 'I' for o in range(len(g)))
                rotations.append(['pi/2', K])
                for m in range(len(GuA)):
                    if not commute(GuA[m], K):
                        p = pauli_mult(K, GuA[m])
                        GuA[m] = p[0]
                        ep_state_trans[m] = 1j * p[1] * ep_state_trans[m]

            g = GuA[i]
            assert any(p != 'I' and p != 'Z' for p in g), f"Unexpected diagonal: {g}"

            J = ''
            found = False
            for j in range(len(g)):
                if g[j] == 'X':
                    J += 'Y' if not found else 'X'
                    found = True
                elif g[j] == 'Y':
                    J += 'X' if not found else 'Y'
                    found = True
                else:
                    J += g[j]

            rotations.append(['pi/2', J])
            for m in range(len(GuA)):
                if not commute(GuA[m], J):
                    p = pauli_mult(J, GuA[m])
                    GuA[m] = p[0]
                    ep_state_trans[m] = 1j * p[1] * ep_state_trans[m]

    return rotations, GuA, np.real(np.array(ep_state_trans))


def apply_rotation(rotation: List, p: str) -> Dict[str, float]:
    """Apply rotation [angle, generator] to Pauli p; return dict of resulting Paulis and coeffs."""
    out = {}
    if not commute(rotation[1], p):
        if rotation[0] == 'pi/2':
            q = pauli_mult(rotation[1], p)
            out[q[0]] = (1j * q[1]).real
        else:
            out[p] = np.cos(rotation[0])
            q = pauli_mult(rotation[1], p)
            out[q[0]] = (1j * q[1] * np.sin(rotation[0])).real
    else:
        out[p] = 1.0
    return out


def pauli_to_sparse(P: str):
    """Pauli string to sparse matrix (scipy csr)."""
    x = ''.join('0' if P[i] in ('I', 'Z') else '1' for i in range(len(P)))
    x = int(x, 2)
    z = ''.join('0' if P[i] in ('I', 'X') else '1' for i in range(len(P)))
    z = int(z, 2)
    y = sum(1 for i in range(len(P)) if P[i] == 'Y')
    rows = list(range(2 ** len(P)))
    cols = [r ^ x for r in rows]
    vals = []
    for r in range(2 ** len(P)):
        sgn = bin(r & z)
        vals.append(((-1.0) ** sum(int(sgn[i]) for i in range(2, len(sgn)))) * ((-1j) ** y))
    return coo_matrix((vals, (rows, cols))).tocsr()




# -----------------------------------------------------------------------------
# Greedy DFS for noncontextual sub-Hamiltonian
# -----------------------------------------------------------------------------

def greedy_dfs(
    ham: Dict[str, float],
    cutoff: float,
    criterion: str = 'weight'
) -> List[List[str]]:
    """Greedy DFS to find maximal noncontextual sub-Hamiltonian. cutoff in seconds."""
    weight = {k: abs(ham[k]) for k in ham}
    possibilities = [k for k, v in sorted(weight.items(), key=lambda item: -item[1])]
    best_guesses = [[]]
    stack = [[[], 0]]
    start_time = datetime.now()
    delta = timedelta(seconds=cutoff)
    i = 0

    while datetime.now() - start_time < delta and stack:
        while i < len(possibilities):
            next_set = stack[-1][0] + [possibilities[i]]
            if not contextualQ(next_set):
                stack.append([next_set, i + 1])
            i += 1

        if criterion == 'weight':
            new_weight = sum(abs(ham[p]) for p in stack[-1][0])
            old_weight = sum(abs(ham[p]) for p in best_guesses[-1])
            if new_weight > old_weight:
                best_guesses.append(stack[-1][0])

        if criterion == 'size' and len(stack[-1][0]) > len(best_guesses[-1]):
            best_guesses.append(stack[-1][0])

        top = stack.pop()
        i = top[1]

    return best_guesses
