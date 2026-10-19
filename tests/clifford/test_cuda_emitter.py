"""CUDA thread-per-lane emitter kernel (clifford/cuda_emitter.py, docs/EMITTER_SAMPLER.md §8), run
as its CPU emulation: the kernel C++ is compiled by the host compiler and every lane's TPL
threads are std::threads joined by a barrier, so the code under test is the GPU kernel's
(odometer, chains, Kogge-Stone exchange), not a re-implementation.  Needs a C++20 compiler;
skipped otherwise.  The GPU path itself is exercised by tests/clifford/check_cuda_emitter.py."""
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
from photonic_engines.clifford import cuda_emitter
from photonic_engines.clifford.emitter_reference import gap_kraus_terms, matrix_cofactor_weights
from photonic_engines.clifford.emitter_triton import emitter_step_weights, sample_triton

if not cuda_emitter.emul_available():
    pytest.skip("no C++20 compiler for the kernel emulation", allow_module_level=True)


def _caterpillar(n, phis, final=None):
    gates = [spin_gate(math.pi / 2, 0)] + [spin_gate(math.pi / 2, p) for p in phis]
    return EmitterMPS.caterpillar(n, gates, final=final)


def _terms(st, T):
    kept, terms = gap_kraus_terms(st, T)
    s0 = torch.stack([t[0] for t in terms]); mats = torch.stack([t[1] for t in terms]); f = torch.stack([t[2] for t in terms])
    return kept, s0, mats, f, torch.zeros(len(terms), dtype=torch.long)


def _weights(U, st, kept, rows, s0, mats, f, t2s, ttl, tpl, gs):
    k = len(kept)
    w, _ = emitter_step_weights(U, st, torch.tensor([kept]), torch.tensor([rows], dtype=torch.long).reshape(1, k - 1),
                                s0, mats, f, t2s, 1, target_total_lanes=ttl, impl="cuda_emul", seq_kw={"tpl": tpl, "gs": gs})
    return w[0]


@pytest.mark.parametrize("tpl,gs", [(1, 0), (1, 1), (2, 0), (2, 1), (4, 0), (4, 1)])
def test_step_weights_match_reference(tpl, gs):
    """Two labels per slot, every step k = 1..n (K1 = 0 included), random and repeated rows,
    one long lane per term and many short segments."""
    n = 5
    st = _caterpillar(n, [0.7, math.pi, 0.2, 1.4], final=(0.6, 0.8j))
    U = haar_unitary(st.m, 5)
    rng = np.random.default_rng(tpl)
    for k in range(1, n + 1):
        T = rng.permutation(n)[:k].tolist()
        kept, s0, mats, f, t2s = _terms(st, T)
        for rows in (rng.integers(0, st.m, size=k - 1).tolist(), [0] * (k - 1), [1, 1, 0, 0][:k - 1]):
            w_ref, _ = matrix_cofactor_weights(U, st, T, rows)
            for ttl in (1, 1 << 17):
                w = _weights(U, st, kept, rows, s0, mats, f, t2s, ttl, tpl, gs)
                assert torch.allclose(w, w_ref, atol=1e-13 * max(1.0, float(w_ref.max()))), (k, rows, ttl)


@pytest.mark.parametrize("tpl", [1, 4, 8])
def test_one_label_wide_tiers(tpl):
    """One label per slot (NL = 1), K up to 13 (tiers 12 and 16 with idle padding columns, KH = 2..13),
    rows drawn from 4 values so the 2020 collapse is deep (odd and even multiplicities, halved top digit)."""
    rng = np.random.default_rng(4)
    n = 13
    ops = rng.normal(size=(n, 1, 2, 2)) + 1j * rng.normal(size=(n, 1, 2, 2))
    st = EmitterMPS(ops, tuple((k,) for k in range(n)), s0=np.array([1.0, 0.0]), m=n + 3, final=np.array([0.6, 0.8j]))
    U = haar_unitary(st.m, 7)
    for k in (7, 9, 12, 13):
        T = rng.permutation(n)[:k].tolist()
        rows = rng.integers(0, 4, size=k - 1).tolist()
        w_ref, _ = matrix_cofactor_weights(U, st, T, rows)
        kept, s0, mats, f, t2s = _terms(st, T)
        for gs in (0, 1):
            for ttl in (1, 1 << 17):
                w = _weights(U, st, kept, rows, s0, mats, f, t2s, ttl, tpl, gs)
                assert torch.allclose(w, w_ref, atol=1e-13 * max(1.0, float(w_ref.max()))), (k, gs, ttl)


@pytest.mark.parametrize("tpl", [1, 4])
def test_wq_variant_matches_reference(tpl):
    """Kernel variant PE_WQ (w folded into the backward seed; the CPU-emulated one of the three): same weights as the
    matrix reference, one and two labels, random and repeated rows, every step."""
    rng = np.random.default_rng(11 + tpl)
    st2 = _caterpillar(5, [0.7, math.pi, 0.2, 1.4], final=(0.6, 0.8j))
    ops = rng.normal(size=(9, 1, 2, 2)) + 1j * rng.normal(size=(9, 1, 2, 2))
    st1 = EmitterMPS(ops, tuple((k,) for k in range(9)), s0=np.array([1.0, 0.0]), m=12, final=np.array([0.6, 0.8j]))
    cuda_emitter.set_variant(wq=1)
    try:
        for st in (st2, st1):
            U = haar_unitary(st.m, 9)
            for k in range(1, st.n + 1):
                T = rng.permutation(st.n)[:k].tolist()
                kept, s0, mats, f, t2s = _terms(st, T)
                for rows in (rng.integers(0, st.m, size=k - 1).tolist(), [0] * (k - 1)):
                    w_ref, _ = matrix_cofactor_weights(U, st, T, rows)
                    for ttl in (1, 1 << 17):
                        w = _weights(U, st, kept, rows, s0, mats, f, t2s, ttl, tpl, 1)
                        assert torch.allclose(w, w_ref, atol=1e-13 * max(1.0, float(w_ref.max()))), (st.n, k, rows, ttl)
    finally:
        cuda_emitter.set_variant()


@pytest.mark.parametrize("gs", [0, 1])
def test_sym_lane_matches_reference(gs):
    """Kernel variant PE_SYM (TPL = 2 lanes without the block scan: phase A forward / backward, exchange, phase B the other
    way, thread 1 in reversed column order): same weights as the matrix reference, one and two labels, every step
    (idle padding columns on both threads), random and repeated rows, one and many segments."""
    rng = np.random.default_rng(29 + gs)
    st2 = _caterpillar(5, [0.7, math.pi, 0.2, 1.4], final=(0.6, 0.8j))
    ops = rng.normal(size=(11, 1, 2, 2)) + 1j * rng.normal(size=(11, 1, 2, 2))
    st1 = EmitterMPS(ops, tuple((k,) for k in range(11)), s0=np.array([1.0, 0.0]), m=14, final=np.array([0.6, 0.8j]))
    cuda_emitter.set_variant(sym=1)
    try:
        for st in (st2, st1):
            U = haar_unitary(st.m, 13)
            for k in range(1, st.n + 1):
                T = rng.permutation(st.n)[:k].tolist()
                kept, s0, mats, f, t2s = _terms(st, T)
                for rows in (rng.integers(0, st.m, size=k - 1).tolist(), [1] * (k - 1)):
                    w_ref, _ = matrix_cofactor_weights(U, st, T, rows)
                    for ttl in (1, 1 << 17):
                        w = _weights(U, st, kept, rows, s0, mats, f, t2s, ttl, 2, gs)
                        assert torch.allclose(w, w_ref, atol=1e-13 * max(1.0, float(w_ref.max()))), (st.n, k, rows, ttl)
    finally:
        cuda_emitter.set_variant()


def test_tables_and_tiers():
    """K1 = 0 tables are the one-term identity; the tier ladder serves K <= 40 in steps of 4."""
    rows = torch.zeros((3, 1), dtype=torch.long)
    drow, mult, radix, N, scale = cuda_emitter.emitter_tables(rows, 0)
    assert N.tolist() == [1, 1, 1] and scale.tolist() == [1.0, 1.0, 1.0] and mult.dtype == torch.int32
    assert cuda_emitter.tier(1)[0] == 8 and cuda_emitter.tier(8)[0] == 8 and cuda_emitter.tier(9)[0] == 12
    assert cuda_emitter.tier(40)[0] == 40
    with pytest.raises(StopIteration):
        cuda_emitter.tier(41)
    cuda_emitter.set_config({8: (1, 1, None)})
    try:
        assert cuda_emitter.tier(5) == (8, 1, 1) and cuda_emitter.tier(5, lanes_per_term=1.0) == (8, 1, 1)
        cuda_emitter.set_config({8: (16, 0, None)}, regime="scatter")      # one lane per term: the scatter table
        assert cuda_emitter.tier(5, lanes_per_term=1.0) == (8, 16, 0) and cuda_emitter.tier(5, lanes_per_term=50.0) == (8, 1, 1)
        assert cuda_emitter.regime_of(None) == "seg" and cuda_emitter.regime_of(cuda_emitter.SCATTER_LANES - 1) == "scatter"
    finally:
        cuda_emitter.set_config(None)


def test_sampler_draws_the_same_samples_as_the_triton_kernel():
    """End to end (several streams, lanes spread over the terms of every stream): identical
    weights give identical draws from the same generator."""
    if os.environ.get("TRITON_INTERPRET", "0") != "1":
        pytest.skip("CPU tensors through Triton need its interpreter (CPU-only hosts); on a GPU host the device path is "
                    "tests/clifford/check_cuda_emitter.py (job 26, run 20261005_213240: Triton refuses CPU pointers)")
    n = 5
    st = _caterpillar(n, [0.3, 1.1, math.pi, 0.5])
    U = haar_unitary(st.m, 3)
    a = sample_triton(st, U, 6, generator=torch.Generator().manual_seed(2), device="cpu", impl="seq")
    b = sample_triton(st, U, 6, generator=torch.Generator().manual_seed(2), device="cpu", impl="cuda_emul")
    assert torch.equal(a, b)
    if not cuda_emitter.available():                      # "auto" without cupy is the seq kernel
        c = sample_triton(st, U, 6, generator=torch.Generator().manual_seed(2), device="cpu", impl="auto")
        assert torch.equal(a, c)
    with pytest.raises(RuntimeError, match="cupy"):
        if not cuda_emitter.available():
            sample_triton(st, U, 2, device="cpu", impl="cuda")
        else:
            raise RuntimeError("cupy present: GPU path available")
