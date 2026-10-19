"""The vectorized FockLayers must reproduce the original recursive/dict construction exactly
(same state order, same edge set) on cases larger than the brute-force enumeration reaches."""
import math

import numpy as np
import pytest

from photonic_engines.slos.basis import FockLayers, _block_shift, _per_mode_max


def _python_layers(n, m, mask):
    """The pre-vectorization builder (photonic_benchmarks.sloscore.basis), kept as oracle."""
    per_max = _per_mode_max(mask, n, m)
    shift = _block_shift(mask)
    keep = (lambda t: True) if not shift else (lambda t: len({p >> shift for p in t}) == len(t))

    def enumerate_k(k):
        out, cur = [], []

        def rec(slot, start, run_mode, run_count):
            if slot == k:
                t = tuple(cur)
                if keep(t):
                    out.append(t)
                return
            for a in range(start, m):
                cnt = run_count + 1 if a == run_mode else 1
                if cnt > per_max[a]:
                    continue
                cur.append(a)
                rec(slot + 1, a, a, cnt)
                cur.pop()
        if k == 0:
            return [()]
        rec(0, 0, -1, 0)
        return out

    layers = [enumerate_k(k) for k in range(n + 1)]
    index = [{s: i for i, s in enumerate(layer)} for layer in layers]
    edges = []
    for k in range(n):
        rows = []
        for c_pos, child in enumerate(layers[k + 1]):
            L, i = len(child), 0
            while i < L:
                a, j = child[i], i + 1
                while j < L and child[j] == a:
                    j += 1
                rows.append((index[k][child[:i] + child[i + 1:]], c_pos, a, math.sqrt(j - i)))
                i = j
        edges.append(rows)
    return layers, edges


CASES = [(4, 6, "fock"), (5, 7, "fock"), (3, 9, "fock"), (4, 8, "unbunched"),
         (4, 8, "dual_rail"), (3, 12, "sparse_1in4"), (4, 7, "tail_empty"), (6, 6, "fock")]


@pytest.mark.parametrize("n,m,mask", CASES)
def test_vectorized_equals_python_builder(n, m, mask):
    fl = FockLayers(n, m, mask)
    layers, edges = _python_layers(n, m, mask)
    for k in range(n + 1):
        assert [tuple(r) for r in fl.states[k].tolist()] == layers[k], f"layer {k}"
    for k in range(n):
        pa, ca, ja, fa = fl.edges[k]
        got = sorted(zip(pa.tolist(), ca.tolist(), ja.tolist(), np.round(fa.astype(np.float64), 5).tolist()))
        exp = sorted((p, c, j, round(f, 5)) for p, c, j, f in edges[k])
        assert got == exp, f"edges {k}"


def test_output_norms_and_occupations():
    fl = FockLayers(4, 5, "fock")
    occ = fl.final_occupations()
    assert occ.shape == (fl.output_size(), 5) and (occ.sum(1) == 4).all()
    norms = fl.output_prob_norms()
    expected = [math.prod(math.factorial(int(x)) for x in row) for row in occ.tolist()]
    assert norms.tolist() == expected
    assert fl.final_states == [tuple(r) for r in occ.tolist()]
