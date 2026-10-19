"""Independent permanent-based amplitude reference - the correctness gate for the kernel.

This deliberately shares NO code with `kernel.py`: it computes each output amplitude directly
from the textbook formula

    <t| phi(U) |s> = per(U_{s,t}) / sqrt(prod_i s_i! * prod_j t_j!)

where U_{s,t} takes s_i copies of input column i and t_j copies of output row j. If the fast
SLOS recurrence and this brute-force permanent agree, the kernel is physically correct. Only
used in tests / the agreement gate (permanents are exponential; fine for small n).
"""
from __future__ import annotations

import math
from collections.abc import Sequence
from itertools import combinations

import torch

from ..masks import Mask
from .basis import get_layers


def permanent(mat: torch.Tensor) -> torch.Tensor:
    """Permanent of a square matrix via Ryser's formula (complex-safe, O(2^n n))."""
    n = mat.shape[0]
    if n == 0:
        return torch.tensor(1.0 + 0j, dtype=mat.dtype)
    total = torch.zeros((), dtype=mat.dtype)
    for k in range(1, n + 1):
        for cols in combinations(range(n), k):
            rowsum = mat[:, list(cols)].sum(dim=1)      # sum over the chosen columns, per row
            prod = rowsum.prod()
            sign = (-1.0) ** (n - k)
            total = total + sign * prod
    return total                                     # Ryser: sum_S (-1)^(n-|S|) prod_i sum_{j in S} a_ij


def _modes_from_occupation(occ: Sequence[int]):
    return [j for j, c in enumerate(occ) for _ in range(int(c))]


def _layer_occupations(layers, k: int) -> list[tuple[int, ...]]:
    """Layer k of the basis as occupation tuples, in the kernel's state order."""
    out = []
    for t in layers.states[k]:
        occ = [0] * layers.m
        for pos in t:
            occ[int(pos)] += 1
        out.append(tuple(occ))
    return out


def product_reference_probs(U, state, mask: str = Mask.FOCK.value) -> list[torch.Tensor]:
    """Probabilities per photon number for a `states.ProductSuperposition`, from explicit
    permanents: the state is expanded into its d^sources Fock components (`state.blocks()`) and
    every output amplitude is summed over the components of that photon-number block.

    Returns n + 1 tensors [S_k] over `get_layers(state.n, state.m, mask).states[k]` — the same
    ordering `ProductSLOSKernel.probs` uses. Exponential in both the components and the
    permanents: the oracle for that kernel, small sources only.
    """
    U = torch.as_tensor(U, dtype=torch.complex128)
    layers = get_layers(state.n, state.m, mask)
    blocks = state.blocks()
    out = []
    for k in range(state.n + 1):
        occs = _layer_occupations(layers, k)
        probs = torch.zeros(len(occs), dtype=torch.float64)
        block = blocks.get(k)
        if block is None:
            out.append(probs)
            continue
        for idx, t in enumerate(occs):
            out_modes = _modes_from_occupation(t)
            t_norm = math.prod(math.factorial(int(c)) for c in t)
            amp = torch.zeros((), dtype=torch.complex128)
            for comp, w in zip(block.occupations.tolist(), block.amplitudes.tolist()):
                in_modes = _modes_from_occupation(comp)
                s_norm = math.prod(math.factorial(int(c)) for c in comp)
                sub = permanent(U[out_modes][:, in_modes]) if out_modes else torch.tensor(1.0 + 0j, dtype=torch.complex128)
                amp = amp + complex(w) * sub / math.sqrt(s_norm * t_norm)
            probs[idx] = (amp.conj() * amp).real
        out.append(probs)
    return out


def reference_probs(U, n: int, m: int, input_occupation: Sequence[int],
                    mask: str = Mask.FOCK.value) -> torch.Tensor:
    """Probabilities over the SAME final basis ordering the kernel uses (get_layers)."""
    U = torch.as_tensor(U, dtype=torch.complex128)
    in_modes = _modes_from_occupation(input_occupation)
    s_norm = math.prod(math.factorial(int(c)) for c in input_occupation)
    layers = get_layers(n, m, mask)
    probs = torch.zeros(len(layers.final_states), dtype=torch.float64)
    for idx, t in enumerate(layers.final_states):
        out_modes = _modes_from_occupation(t)
        if not out_modes:                                # vacuum <-> vacuum
            amp = torch.tensor(1.0 + 0j, dtype=torch.complex128)
        else:
            sub = U[out_modes][:, in_modes]              # rows=output, cols=input
            t_norm = math.prod(math.factorial(int(c)) for c in t)
            amp = permanent(sub) / math.sqrt(s_norm * t_norm)
        probs[idx] = (amp.conj() * amp).real
    return probs


# ---- partial distinguishability oracles (docs/DISTINGUISHABILITY.md) -----------------------

def _multisets(m: int, k: int):
    """Occupations of k photons over m modes, as tuples."""
    from itertools import combinations_with_replacement
    for rows in combinations_with_replacement(range(m), k):
        occ = [0] * m
        for r in rows:
            occ[r] += 1
        yield tuple(occ)


def species_probs(U, components: Sequence[tuple[complex, Sequence[int]]], species: Sequence[int]) -> dict:
    """Output distribution {occupation: probability} of n photons in `species` (one species index
    per photon SLOT; photons of equal species interfere, different species never do) for a
    coherent input sum_c amp_c |cols_c> over Fock components that share the slot structure
    (`components` = [(amp, cols)], cols[slot] the input column of that slot — the emitter chain
    expanded over its label words is one such sum). Species-resolved outputs are enumerated
    explicitly: the amplitude of (t_mu)_mu is sum_c amp_c prod_mu per(U[rows(t_mu), cols_c[slots of mu]])
    / sqrt(prod t_mu!), squared and summed into the coarse pattern. Exponential in everything;
    the oracle for the `indist=` paths, n <= 6.
    """
    from itertools import product as iproduct

    import numpy as np
    U = torch.as_tensor(U, dtype=torch.complex128)
    m = U.shape[0]
    species = list(species)
    n = len(species)
    if any(len(c) != n for _, c in components):
        raise ValueError("every component must give one column per slot")
    groups = sorted(set(species))
    slots = {mu: [j for j in range(n) if species[j] == mu] for mu in groups}
    amps = np.array([complex(a) for a, _ in components])
    # per species: the permanent of every (output multiset, component) pair, tabulated once
    tables = []
    for mu in groups:
        ts = list(_multisets(m, len(slots[mu])))
        tab = np.empty((len(ts), len(components)), dtype=np.complex128)
        cache = {}
        for ti, t in enumerate(ts):
            rows = _modes_from_occupation(t)
            for ci, (_, cols) in enumerate(components):
                key = (ti, tuple(cols[j] for j in slots[mu]))
                if key not in cache:
                    cache[key] = complex(permanent(U[rows][:, list(key[1])]))
                tab[ti, ci] = cache[key]
        norms = np.array([math.prod(math.factorial(int(c)) for c in t) for t in ts], dtype=np.float64)
        tables.append((ts, tab, norms))
    out: dict = {}
    for choice in iproduct(*[range(len(ts)) for ts, _, _ in tables]):
        prod = amps.copy()
        norm = 1.0
        total = np.zeros(m, dtype=np.int64)
        for (ts, tab, norms), ti in zip(tables, choice):
            prod = prod * tab[ti]
            norm *= norms[ti]
            total += np.asarray(ts[ti])
        amp = prod.sum()
        key = tuple(int(v) for v in total)
        out[key] = out.get(key, 0.0) + float(abs(amp) ** 2) / norm
    return out


def gram_probs(U, cols: Sequence[int], S) -> dict:
    """Output distribution {occupation: probability} of n photons with pairwise internal-state
    overlaps S [n, n] (Gram matrix, S_ii = 1) entering columns `cols` — the double permutation
    sum P(t) = (1/prod t_j!) sum_{sigma, tau} prod_i S[sigma(i), tau(i)] U[r_i, cols[sigma(i)]]
    conj(U[r_i, cols[tau(i)]]) (Tichy, PRA 91, 022316). (n!)^2 terms per output: n <= 5. The
    oracle that does not know about species: S = x J + (1 - x) I must equal the x-model mixture.
    """
    from itertools import permutations
    U = torch.as_tensor(U, dtype=torch.complex128)
    S = torch.as_tensor(S, dtype=torch.complex128)
    n, m = len(cols), U.shape[0]
    perms = list(permutations(range(n)))
    out = {}
    for t in _multisets(m, n):
        rows = _modes_from_occupation(t)
        acc = 0.0 + 0.0j
        for sg in perms:
            a = 1.0 + 0.0j
            for i in range(n):
                a *= complex(U[rows[i], cols[sg[i]]])
            for tu in perms:
                b = 1.0 + 0.0j
                for i in range(n):
                    b *= complex(S[sg[i], tu[i]]) * complex(U[rows[i], cols[tu[i]]].conj())
                acc += a * b
        out[t] = acc.real / math.prod(math.factorial(int(c)) for c in t)
    return out


# ---- non-linear boson sampling oracles (docs/NONLINEAR.md) ----------------------------------

def fock_space_unitary(U, n: int) -> torch.Tensor:
    """The n-photon representation of the mode unitary U, dense: [S, S] complex128 over
    `get_layers(n, m).final_states`, entry (s, t) = per(U[s, t]) / sqrt(s! t!). S^2 permanents: n <= 4.
    """
    U = torch.as_tensor(U, dtype=torch.complex128)
    occ = get_layers(n, U.shape[0]).final_states
    F = torch.zeros(len(occ), len(occ), dtype=torch.complex128)
    for a, s in enumerate(occ):
        rs = _modes_from_occupation(s)
        for b, t in enumerate(occ):
            nrm = math.prod(math.factorial(int(c)) for c in s) * math.prod(math.factorial(int(c)) for c in t)
            F[a, b] = permanent(U[rs][:, _modes_from_occupation(t)]) / math.sqrt(nrm)
    return F


def _site_list(unitaries, sites):
    L = len(unitaries) - 1
    out = []
    for layer, mode, f in sites:
        if not 1 <= int(layer) <= L:
            raise ValueError(f"site layer {layer}: between unitaries, 1..{L}")
        out.append((int(layer), int(mode), [float(v) for v in f]))
    return out


def nonlinear_state_amplitudes(unitaries, input_occupation: Sequence[int], sites) -> torch.Tensor:
    """Oracle (a) of non-linear boson sampling: the state vector, gate by gate on the dense Fock space.

    unitaries  [U_0, ..., U_L], each [m, m] (rows = outputs); sites  [(layer, mode, f)], layer in 1..L:
    exp(-i f[n_mode]) applied between U_{layer-1} and U_layer, f a phase per photon number (length >= n+1).
    Returns the output amplitudes [S] complex128 over `get_layers(n, m).final_states`: each U_l as its dense
    n-photon matrix (`fock_space_unitary`), each site as a diagonal phase. Shares nothing with the branch sum.
    """
    input_occupation = [int(c) for c in input_occupation]
    n = sum(input_occupation)
    m = len(input_occupation)
    occ = get_layers(n, m).final_states
    psi = torch.zeros(len(occ), dtype=torch.complex128)
    psi[occ.index(tuple(input_occupation))] = 1.0
    sites = _site_list(unitaries, sites)
    for layer, U in enumerate(unitaries):
        for lay, mode, f in sites:
            if lay == layer:
                ph = [complex(math.cos(f[s[mode]]), -math.sin(f[s[mode]])) for s in occ]
                psi = psi * torch.tensor(ph, dtype=torch.complex128)
        psi = fock_space_unitary(U, n) @ psi
    return psi


def nonlinear_branch_amplitudes(unitaries, input_occupation: Sequence[int], sites) -> torch.Tensor:
    """Oracle (b) of non-linear boson sampling: the branch sum of permanents, same output order as oracle (a).

    Every site is expanded on the full grid theta_l = 2 pi l / (n+1), l = 0..n:
    exp(-i f(N)) = sum_l c_l exp(i theta_l N) on 0 <= N <= n, c_l = (1/(n+1)) sum_j exp(-i f(j) - i theta_l j),
    so the circuit is the sum over the (n+1)^(#sites) branch unitaries U_L D_L ... D_1 U_0 (D a phase on the
    site modes) weighted by the product of the c's, and <s| . |t> = sum_branches c per(U_b[s, t]) / sqrt(s! t!).
    """
    from itertools import product as iproduct
    input_occupation = [int(c) for c in input_occupation]
    n = sum(input_occupation)
    m = len(input_occupation)
    sites = _site_list(unitaries, sites)
    us = [torch.as_tensor(U, dtype=torch.complex128) for U in unitaries]
    th = [2 * math.pi * k / (n + 1) for k in range(n + 1)]
    coef = [[sum(complex(math.cos(-f[j] - t * j), math.sin(-f[j] - t * j)) for j in range(n + 1)) / (n + 1) for t in th]
            for _, _, f in sites]
    cols = _modes_from_occupation(input_occupation)
    tn = math.prod(math.factorial(c) for c in input_occupation)
    occ = get_layers(n, m).final_states
    out = torch.zeros(len(occ), dtype=torch.complex128)
    for br in iproduct(range(n + 1), repeat=len(sites)):
        T = us[0]
        c = 1.0 + 0.0j
        for layer in range(1, len(us)):
            d = torch.ones(m, dtype=torch.complex128)
            for k, (lay, mode, _) in enumerate(sites):
                if lay == layer:
                    d[mode] = complex(math.cos(th[br[k]]), math.sin(th[br[k]]))
            T = us[layer] @ (d[:, None] * T)
        for k in range(len(sites)):
            c *= coef[k][br[k]]
        for a, s in enumerate(occ):
            nrm = math.sqrt(math.prod(math.factorial(int(v)) for v in s) * tn)
            out[a] += c * permanent(T[_modes_from_occupation(s)][:, cols]) / nrm
    return out
