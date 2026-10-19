"""Direct correctness gate for the compact (photon-position) Fock basis: its state set and its
raising edges must match an INDEPENDENT occupation-vector enumeration (the textbook construction
this module optimizes). Kept separate from the kernel/permanent tests so a basis regression is
localized here rather than surfacing as a distant amplitude mismatch.
"""
import math
from itertools import product

import pytest

from photonic_engines.masks import Mask
from photonic_engines.slos.basis import FockLayers, positions_to_occupation


def _mask_ok(occ, mask, m):
    if mask == Mask.FOCK.value:
        return True
    if mask == Mask.UNBUNCHED.value:
        return all(x <= 1 for x in occ)
    if mask == Mask.DUAL_RAIL.value:
        return all(occ[2 * i] + occ[2 * i + 1] <= 1 for i in range(m // 2))
    if mask == Mask.SPARSE_1IN4.value:
        return all(sum(occ[b:b + 4]) <= 1 for b in range(0, m, 4))
    if mask == Mask.TAIL_EMPTY.value:
        return occ[-1] == 0 and occ[-2] == 0
    raise ValueError(mask)


def _brute_states(k, m, mask):
    """Independent enumeration: all length-m occupation vectors summing to k respecting the mask."""
    out = set()
    cap = k
    for occ in product(range(cap + 1), repeat=m):
        if sum(occ) == k and _mask_ok(occ, mask, m):
            out.add(occ)
    return out


CASES = [(2, 4, "fock"), (3, 4, "fock"), (2, 5, "unbunched"),
         (2, 6, "dual_rail"), (2, 8, "sparse_1in4"), (2, 5, "tail_empty"), (3, 5, "fock")]


@pytest.mark.parametrize("n,m,mask", CASES)
def test_states_match_independent_enumeration(n, m, mask):
    fl = FockLayers(n, m, mask)
    for k in range(n + 1):
        got = {positions_to_occupation(t, m) for t in fl.layers[k]}
        assert len(got) == len(fl.layers[k])               # no duplicate states
        assert got == _brute_states(k, m, mask), f"layer {k} mismatch"


@pytest.mark.parametrize("n,m,mask", CASES)
def test_edges_match_creation_operator(n, m, mask):
    fl = FockLayers(n, m, mask)
    occ = [[positions_to_occupation(t, m) for t in layer] for layer in fl.layers]
    for k in range(n):
        # occupation-keyed index for the child layer (fl.index is keyed by position tuples)
        child_idx = {o: i for i, o in enumerate(occ[k + 1])}
        # independently derive the expected raising edges as occupation transitions
        expected = set()
        for p_pos, p in enumerate(occ[k]):
            for j in range(m):
                child = (*p[:j], p[j] + 1, *p[j + 1:])
                if child in child_idx:                     # child survives the mask
                    expected.add((p_pos, child_idx[child], j, round(math.sqrt(p[j] + 1), 5)))
        # edges are four flat arrays (parent, child, mode, factor) since the build-transient fix
        pa, ca, ja, fa = fl.edges[k]
        got = {(pp, cp, j, round(float(f), 5))
               for pp, cp, j, f in zip(pa.tolist(), ca.tolist(), ja.tolist(), fa.tolist())}
        assert got == expected, f"edge mismatch at layer {k}->{k+1}"


def test_positions_representation_is_compact():
    # a k-photon state is stored as a length-k tuple (not length-m): the whole point
    fl = FockLayers(2, 64, "fock")
    assert all(len(t) == 2 for t in fl.layers[2])
    assert all(len(t) == 1 for t in fl.layers[1])


@pytest.mark.parametrize("m", [127, 128, 129, 200, 257])
def test_wide_mode_counts_do_not_overflow(m):
    """Mode positions are int8 up to 127 modes and int16 beyond: the basis, the raising edges
    and the kernel must stay exact past 128 modes (a regression: int8 wrapped mode 128 to -128)."""
    import numpy as np
    import torch

    from photonic_engines.slos import FockSLOSKernel, reference_probs
    from photonic_engines.slos.basis_torch import TorchFockLayers
    n = 2
    L = FockLayers(n, m)
    assert L.states[n].dtype == (np.int8 if m <= 127 else np.int16)
    assert int(L.states[n].max()) == m - 1 and int(L.states[n].min()) == 0
    assert len({tuple(r) for r in L.states[n].tolist()}) == math.comb(m + n - 1, n)
    T = TorchFockLayers(n, m)
    for k in range(n + 1):
        assert np.array_equal(L.states[k], T.states[k].cpu().numpy())
    occ = [0] * m
    occ[0], occ[m - 1] = 1, 1
    U = torch.linalg.qr(torch.randn(m, m, dtype=torch.complex128, generator=torch.Generator().manual_seed(0)))[0]
    p = FockSLOSKernel(n, m, occ, dtype=torch.complex128).probs(U)
    assert abs(float(p.sum()) - 1) < 1e-12
    assert float((p - reference_probs(U, n, m, occ)).abs().max()) < 1e-14
