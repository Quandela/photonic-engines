"""Triton emitter kernel (clifford/emitter_triton.py): per-step weights equal the reference
matrix-cofactor weights, and the end-to-end sampler matches the strong simulation. Runs under
the Triton CPU interpreter without a GPU (slow: small cases only)."""
import math
import os

import numpy as np
import pytest

torch = pytest.importorskip("torch")
if not torch.cuda.is_available():
    os.environ.setdefault("TRITON_INTERPRET", "1")     # must precede the first `import triton`
pytest.importorskip("triton")

from helpers import haar_unitary

from photonic_engines import EmitterMPS, spin_gate
from photonic_engines.clifford.emitter_reference import gap_kraus_terms, matrix_cofactor_weights
from photonic_engines.clifford.emitter_triton import emitter_step_weights, sample_triton
from photonic_engines.slos.emitter import EmitterSLOSKernel

DEV = "cuda" if torch.cuda.is_available() else "cpu"
C128 = torch.complex128


def _caterpillar(n, phis, final=None):
    gates = [spin_gate(math.pi / 2, 0)] + [spin_gate(math.pi / 2, p) for p in phis]
    return EmitterMPS.caterpillar(n, gates, final=final)


@pytest.mark.parametrize("seed,impl", [(0, "seq"), (1, "seq"), (0, "scan")])
def test_step_weights_match_reference(seed, impl):
    n = 5
    st = _caterpillar(n, [0.7, math.pi, 0.2, 1.4], final=(0.6, 0.8j))
    U = haar_unitary(st.m, seed + 5)
    rng = np.random.default_rng(seed)
    for k in range(1, n + 1):
        T = rng.permutation(n)[:k].tolist()
        rows = rng.integers(0, st.m, size=k - 1).tolist()
        w_ref, _ = matrix_cofactor_weights(U, st, T, rows)
        kept, terms = gap_kraus_terms(st, T)
        s0 = torch.stack([t[0] for t in terms]).to(DEV); mats = torch.stack([t[1] for t in terms]).to(DEV)
        f = torch.stack([t[2] for t in terms]).to(DEV)
        t2s = torch.zeros(len(terms), dtype=torch.long, device=DEV)
        for max_lanes, seg in ((1, None), (1 << 16, None), (1 << 16, 2)):   # 1 lane/term, many, several programs/term
            w, _ = emitter_step_weights(U.to(DEV), st, torch.tensor([kept], device=DEV),
                                        torch.tensor([rows], dtype=torch.long, device=DEV).reshape(1, k - 1),
                                        s0, mats, f, t2s, 1, max_lanes=max_lanes, seg=seg, impl=impl)
            assert torch.allclose(w[0].cpu(), w_ref, atol=1e-12 * max(1.0, float(w_ref.max()))), (k, max_lanes, seg)


@pytest.mark.parametrize("B,mat_mode,rows", [
    (1, "scalar", [7, 2, 5, 0]), (4, "hoist", [3, 3, 1, 0]), (8, "scalar", [2, 2, 2, 2]),
    (2, "scalar", [4, 1, 4, 1]), (4, "scalar", [6, 6, 6, 1]),
])
def test_seq_kernel_blocks(B, mat_mode, rows):
    """block count / padding / matrix mode variants of the generated kernel, with repeated rows
    (2020 collapse: mixed-radix walk, halved top digit as a single row or an odd group)."""
    from photonic_engines.clifford import emitter_triton as et
    n = 5
    st = _caterpillar(n, [0.3, 1.1, math.pi, 0.5], final=(0.6, 0.8j))
    U = haar_unitary(st.m, 11)
    rng = np.random.default_rng(3)
    T = rng.permutation(n).tolist()
    w_ref, _ = matrix_cofactor_weights(U, st, T, rows)
    kept, terms = gap_kraus_terms(st, T)
    s0 = torch.stack([t[0] for t in terms]).to(DEV); mats = torch.stack([t[1] for t in terms]).to(DEV)
    f = torch.stack([t[2] for t in terms]).to(DEV); t2s = torch.zeros(len(terms), dtype=torch.long, device=DEV)
    old = et.SEQ_MAT_MODE
    try:
        et.SEQ_MAT_MODE = mat_mode
        et.set_config({8: (2, B)}, "seq")
        for tl_ in (1, 1 << 17):                       # one long lane per term / many short segments
            w, _ = emitter_step_weights(U.to(DEV), st, torch.tensor([kept], device=DEV),
                                        torch.tensor([rows], dtype=torch.long, device=DEV).reshape(1, n - 1),
                                        s0, mats, f, t2s, 1, target_total_lanes=tl_, impl="seq")
            assert torch.allclose(w[0].cpu(), w_ref, atol=1e-12 * max(1.0, float(w_ref.max()))), tl_
    finally:
        et.SEQ_MAT_MODE = old
        et.set_config(None, "seq")


@pytest.mark.parametrize("state,seed,impl", [
    (_caterpillar(3, [0, 0]), 1, "seq"),                              # traced spin
    (_caterpillar(4, [math.pi, 0, 0.9], final=(1, 0)), 2, "scan"),    # projected
])
def test_sampler_matches_strong_simulation(state, seed, impl):
    U = haar_unitary(state.m, seed)
    k = EmitterSLOSKernel(state, dtype=C128)
    p = k.probs(U)
    p = p / p.sum()
    # interpreter: the seq kernel runs one tiny program per (term, segment) -> keep the CPU shots low
    shots = 3000 if DEV == "cuda" else (300 if impl == "seq" else 200)
    occ = sample_triton(state, U, shots, generator=torch.Generator().manual_seed(seed), device=DEV, impl=impl).cpu()
    assert (occ.sum(1) == state.n).all()
    idx = {t: i for i, t in enumerate(k.layers.final_states)}
    emp = torch.zeros_like(p)
    for row in occ.tolist():
        emp[idx[tuple(row)]] += 1
    emp /= shots
    tvd = 0.5 * float((emp - p).abs().sum())
    floor = 0.5 * float(torch.sqrt(p * (1 - p) / shots).sum()) * math.sqrt(2 / math.pi)
    assert tvd < 2.0 * floor, (tvd, floor)


def test_step_weights_stream_chunking():
    """The padded candidate product split into stream chunks gives the same samples."""
    import photonic_engines.clifford.emitter_triton as et
    from photonic_engines import EmitterMPS, spin_gate
    from photonic_engines.clifford import emitter_sample
    n, m = 4, 8
    st = EmitterMPS.caterpillar(n, [spin_gate(math.pi / 2, 0)] * n, m=m)
    U = haar_unitary(m, 3)
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    a = emitter_sample(st, U.to(dev), 16, generator=torch.Generator().manual_seed(1), backend="triton", device=dev)
    old = et._BMM_BUDGET
    et._BMM_BUDGET = 1
    try:
        b = emitter_sample(st, U.to(dev), 16, generator=torch.Generator().manual_seed(1), backend="triton", device=dev)
    finally:
        et._BMM_BUDGET = old
    assert torch.equal(a.cpu(), b.cpu())
