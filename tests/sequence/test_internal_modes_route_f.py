"""Internal photon modes, the channel route (docs/SEQUENCE.md §14): the fine Kraus set of a bin on the populated
internal span. Oracles: the expanded construction (labels (l, c) on m K modes, U (x) 1_K, outcomes summed over c,
through the plain SequenceKernel) and a 50-digit brute-force evaluation of K_tau = (1/sqrt(tau!)) sum_sigma
prod_j O_j[R_sigma(j), c_sigma(j)]."""
import itertools
import math

import mpmath
import numpy as np
import pytest
import torch
from helpers import haar_unitary

from photonic_engines import spin_gate
from photonic_engines.sequence import BinEmitter, ModeBinEmitter, SequenceKernel, bin_emitter
from photonic_engines.sequence_modes import ModeSequenceKernel


def _mode_emitter(n, L, K, d, modes, seed, final=None):
    """A generic complete emitter: per bin a random isometry [L K d, d] split into the A[l, k] blocks."""
    rng = np.random.default_rng(seed)
    ops = np.empty((n, L, K, d, d), dtype=complex)
    for b in range(n):
        z = rng.normal(size=(L * K * d, d)) + 1j * rng.normal(size=(L * K * d, d))
        q, _ = np.linalg.qr(z)
        ops[b] = q.reshape(L, K, d, d)
    s0 = np.zeros(d, dtype=complex)
    s0[0] = 1
    return bin_emitter(ops, modes, s0, final)


def _expanded(emitters, m, K):
    """The expanded construction: emitter labels (l, c) -> input mode c m + in(l), a plain kernel on m K modes."""
    ems = []
    for e in emitters:
        ops = e.ops if isinstance(e, ModeBinEmitter) else e.ops[:, :, None]
        n, L, Kj, d, _ = ops.shape
        full = np.zeros((n, L, K, d, d), dtype=complex)
        full[:, :, :Kj] = ops
        ops_exp = full.transpose(0, 2, 1, 3, 4).reshape(n, K * L, d, d)                          # label c L + l
        modes = tuple(c * m + e.modes[l] for c in range(K) for l in range(L))
        ems.append(BinEmitter(ops_exp, modes, e.s0, e.final))
    return SequenceKernel(ems, m * K)


def _superops(K, coarse, S):
    """Per spatial outcome t, sum_{coarse[q] = t} K_q (x) conj(K_q): [S, chi^2, chi^2]."""
    chi = K.shape[-1]
    ops = torch.einsum("qab,qcd->qacbd", K, K.conj()).reshape(K.shape[0], chi * chi, chi * chi)
    return torch.zeros((S, chi * chi, chi * chi), dtype=K.dtype).index_add(0, coarse, ops)


def _expanded_superops(emitters, m, K, U, b):
    ex = _expanded(emitters, m, K)
    Ub = torch.block_diag(*[U] * K)                                                              # mode c m + i
    Kx = ex.kraus(Ub, b)
    occ = ex.patterns.reshape(-1, K, m).sum(1)
    plain = SequenceKernel([BinEmitter(np.zeros((1, 1, 1, 1)), (j,), [1]) for j in range(len(emitters))], m)
    return _superops(Kx, plain.pattern_index(occ), plain.S)


CASES = [  # (k emitters, m, K, labels L)
    (1, 2, 3, 2), (2, 4, 2, 2), (2, 4, 3, 2), (2, 4, 8, 2), (3, 6, 2, 2), (2, 3, 3, 1),
]


@pytest.mark.parametrize("k,m,K,L", CASES)
def test_route_f_matches_expanded_construction(k, m, K, L):
    """Per spatial outcome, the channel of the fine set equals the expanded construction's (superoperator, divided by
    chi: the normalised Choi scale) to 1e-14 (k <= 2) / 1e-13 (k = 3); the set is complete."""
    n = 2
    ems = [_mode_emitter(n, L, K, 2, tuple(range(L * j, L * j + L)), seed=10 * k + j) for j in range(k)]
    U = haar_unitary(m, 7 + k + K)
    mk = SequenceKernel(ems, m)
    assert isinstance(mk, ModeSequenceKernel)
    for b in range(n):
        Kq, cq = mk.kraus(U, b), mk.coarse(U, b)
        assert Kq.shape[0] == cq.shape[0] and int(cq.max()) < mk.S
        got = _superops(Kq, cq, mk.S)
        ref = _expanded_superops(ems, m, K, U, b)
        err = float((got - ref).abs().max()) / mk.chi
        assert err <= (1e-14 if k <= 2 else 1e-13), err
        eye = torch.eye(mk.chi, dtype=Kq.dtype)
        assert float((torch.einsum("qba,qbc->ac", Kq.conj(), Kq) - eye).abs().max()) < 1e-13


def test_route_f_unequal_modes_and_plain_emitter():
    """Emitters with different K (3 and 2) and a plain BinEmitter share the first internal modes."""
    m, n = 6, 2
    ems = [_mode_emitter(n, 2, 3, 2, (0, 1), 1), _mode_emitter(n, 2, 2, 2, (2, 3), 2),
           BinEmitter.caterpillar(n, [spin_gate(math.pi / 2, 0)] * n, (4, 5))]
    U = haar_unitary(m, 3)
    mk = SequenceKernel(ems, m)
    assert mk.K == 3
    got = _superops(mk.kraus(U, 1), mk.coarse(U, 1), mk.S)
    ref = _expanded_superops(ems, m, 3, U, 1)
    assert float((got - ref).abs().max()) / mk.chi <= 1e-13


def _reference_fine(mk, U, b):
    """50-digit K_tau over the kernel's own fine patterns: (1/sqrt(tau!)) sum over assignments of photons to events."""
    mpmath.mp.dps = 50
    stage = next(iter(mk._stages.values()))
    ops = mk.stage_ops(b)
    Kb, m, chi = stage.Kb, mk.m, mk.chi
    O = []                                                                                       # O_j[r][c]: chi x chi
    for j, A in enumerate(ops):
        Aj = mk._embed(j, A).numpy()                                                             # [L, K, chi, chi]
        Oj = [[mpmath.matrix(chi, chi) for _ in range(Kb)] for _ in range(m)]
        for r in range(m):
            for c in range(Kb):
                for l, col in enumerate(mk.emitters[j].modes):
                    u = mpmath.mpc(complex(U[r, col]))
                    for x in range(chi):
                        for y in range(chi):
                            if Aj[l, c, x, y] != 0:
                                Oj[r][c][x, y] += u * mpmath.mpc(complex(Aj[l, c, x, y]))
        O.append(Oj)
    out = []
    for tau in stage.fine_occ:
        events = [(i % m, i // m) for i, t in enumerate(tau) for _ in range(t)]                  # (row, internal)
        acc = mpmath.matrix(chi, chi)
        for perm in itertools.permutations(range(len(events))):
            P = mpmath.eye(chi)
            for j in range(mk.k):
                r, c = events[perm[j]]
                P = O[j][r][c] * P
            acc += P
        out.append(acc / mpmath.sqrt(math.prod(math.factorial(int(t)) for t in tau)))
    return out


@pytest.mark.parametrize("k,m,K", [(2, 3, 2), (3, 3, 2)])
def test_route_f_against_high_precision_reference(k, m, K):
    """Each fine operator against the 50-digit brute force: max error <= 1e-14 (measured ~1e-16 on these cases)."""
    ems = [_mode_emitter(1, 1, K, 2, (j,), seed=40 + j) for j in range(k)]
    U = haar_unitary(m, 41)
    mk = SequenceKernel(ems, m)
    Kq = mk.kraus(U, 0).numpy()
    ref = _reference_fine(mk, U, 0)
    err = max(abs(complex(ref[q][x, y]) - Kq[q, x, y]) for q in range(len(ref)) for x in range(mk.chi) for y in range(mk.chi))
    assert err <= 1e-14, err


def test_internal_rotation_and_zero_modes():
    """A -> A V (V unitary on the internal space, common to the emitters) leaves the channel unchanged; an internal
    mode on which every operator is zero is dropped exactly (bit-identical operators)."""
    m, n, K = 4, 1, 3
    ems = [_mode_emitter(n, 2, K, 2, (2 * j, 2 * j + 1), seed=50 + j) for j in range(2)]
    U = haar_unitary(m, 51)
    rng = np.random.default_rng(52)
    V, _ = np.linalg.qr(rng.normal(size=(K, K)) + 1j * rng.normal(size=(K, K)))
    rot = [bin_emitter(np.einsum("blkij,kc->blcij", e.ops, V), e.modes, e.s0) for e in ems]
    a, b = SequenceKernel(ems, m), SequenceKernel(rot, m)
    assert float((_superops(a.kraus(U, 0), a.coarse(U, 0), a.S) - _superops(b.kraus(U, 0), b.coarse(U, 0), b.S)).abs().max()) < 1e-14
    padded = [bin_emitter(np.concatenate([e.ops, np.zeros_like(e.ops[:, :, :1])], 2), e.modes, e.s0) for e in ems]
    c = SequenceKernel(padded, m)
    assert c.K == K + 1 and torch.equal(a.kraus(U, 0), c.kraus(U, 0)) and torch.equal(a.coarse(U, 0), c.coarse(U, 0))


def test_spin_independent_shape_is_the_plain_channel():
    """A[l, k] = v_k A[l]: with the span rotation (span_rtol) the populated span is one mode and the channel is the
    plain emitter's, to rounding."""
    m, n = 4, 2
    plain = [BinEmitter.caterpillar(n, [spin_gate(math.pi / 2, 0.3 * j)] * n, (2 * j, 2 * j + 1)) for j in range(2)]
    v = np.array([0.6, 0.8j, 0.0])
    ems = [bin_emitter(e.ops[:, :, None] * v[None, None, :, None, None], e.modes, e.s0) for e in plain]
    U = haar_unitary(m, 60)
    mk = SequenceKernel(ems, m, span_rtol=1e-12)
    ref = SequenceKernel(plain, m)
    for b in range(n):
        got = _superops(mk.kraus(U, b), mk.coarse(U, b), mk.S)
        want = _superops(ref.kraus(U, b), ref.coarse(U, b), ref.S)
        assert float((got - want).abs().max()) < 1e-13
    assert next(iter(mk._stages.values())).Kb == 1


def test_compressed_set_keeps_the_channel():
    """compress=rtol: per outcome at most min(F_t, chi^2) operators, the same channel to 1e-14, fewer operators when
    the fine set exceeds chi^2 (k = 2, K = 8: 64 fine operators per collision-free outcome, chi^2 = 16)."""
    m, n = 4, 1
    ems = [_mode_emitter(n, 2, 8, 2, (2 * j, 2 * j + 1), seed=70 + j) for j in range(2)]
    U = haar_unitary(m, 71)
    mk = SequenceKernel(ems, m)
    K0, c0 = mk.kraus(U, 0), mk.coarse(U, 0)
    K1, c1 = mk.kraus(U, 0, compress=1e-12), mk.coarse(U, 0, compress=1e-12)
    assert float((_superops(K0, c0, mk.S) - _superops(K1, c1, mk.S)).abs().max()) < 1e-14
    n0 = torch.bincount(c0, minlength=mk.S)
    n1 = torch.bincount(c1, minlength=mk.S)
    assert bool((n1 <= torch.minimum(n0, torch.tensor(mk.chi ** 2))).all()) and K1.shape[0] < K0.shape[0]


@pytest.mark.parametrize("rank", [1, 3, 4, 5, 9])
def test_compression_finds_the_exact_rank(rank):
    """Outcomes whose stacked operators have exact rank r (built as products [F, r] x [r, chi^2], F = 20 > r) are
    compressed to exactly r operators at rtol = 1e-12, with the same channel to 1e-14 relative."""
    from photonic_engines.slos.internal import compress_by_outcome
    chi, F, T = 4, 20, 3
    g = torch.Generator().manual_seed(rank)

    def rnd(*shape):
        return torch.complex(torch.randn(*shape, generator=g, dtype=torch.float64),
                             torch.randn(*shape, generator=g, dtype=torch.float64))
    B = rnd(T, F, rank) @ rnd(T, rank, chi * chi)                                              # [T, F, chi^2], rank r
    K = B.reshape(T * F, chi, chi)
    coarse = torch.arange(T).repeat_interleave(F)
    Kc, cc = compress_by_outcome(K, coarse, 1e-12)
    assert torch.equal(torch.bincount(cc, minlength=T), torch.full((T,), rank))
    a, b = _superops(K, coarse, T), _superops(Kc, cc, T)
    assert float((a - b).abs().max() / a.abs().max()) < 1e-14


def test_route_f_per_bin_unitaries():
    m, n = 4, 2
    ems = [_mode_emitter(n, 2, 2, 2, (2 * j, 2 * j + 1), seed=80 + j) for j in range(2)]
    Us = torch.stack([haar_unitary(m, 81), haar_unitary(m, 82)])
    mk = SequenceKernel(ems, m)
    for b in range(n):
        got = _superops(mk.kraus(Us, b), mk.coarse(Us, b), mk.S)
        assert float((got - _expanded_superops(ems, m, 2, Us[b], b)).abs().max()) / mk.chi <= 1e-14


@pytest.mark.gpu
def test_route_f_cuda_matches_cpu():
    """The chunked gather / contract / index_add layer on the GPU gives the CPU operators to rounding, chunked or not."""
    if not torch.cuda.is_available():
        pytest.skip("no CUDA device")
    m, n = 6, 1
    ems = [_mode_emitter(n, 2, 3, 2, (2 * j, 2 * j + 1), seed=95 + j) for j in range(3)]
    U = haar_unitary(m, 96)
    cpu = SequenceKernel(ems, m)
    for chunk in (1 << 30, 1 << 12):
        gpu = SequenceKernel(ems, m, device="cuda", chunk_bytes=chunk)
        assert torch.equal(cpu.coarse(U, 0), gpu.coarse(U, 0).cpu())
        assert float((cpu.kraus(U, 0) - gpu.kraus(U, 0).cpu()).abs().max()) < 1e-14


def test_route_f_chunking_is_exact():
    """A small chunk budget (one raised mode per chunk) gives the same operators as one chunk."""
    m, n = 4, 1
    ems = [_mode_emitter(n, 2, 3, 2, (2 * j, 2 * j + 1), seed=97 + j) for j in range(2)]
    U = haar_unitary(m, 98)
    a = SequenceKernel(ems, m).kraus(U, 0)
    b = SequenceKernel(ems, m, chunk_bytes=1).kraus(U, 0)
    assert float((a - b).abs().max()) < 1e-15


def test_route_f_autograd_is_not_cached():
    """With U requiring grad the operators carry a fresh graph at every call: two backward passes work, and an
    equal-valued second unitary gets its own gradient (no cached graph of the first)."""
    m, n = 4, 1
    ems = [_mode_emitter(n, 2, 2, 2, (2 * j, 2 * j + 1), seed=99 + j) for j in range(2)]
    mk = SequenceKernel(ems, m)
    U0 = haar_unitary(m, 100)
    U1 = U0.clone().requires_grad_(True)
    U2 = U0.clone().requires_grad_(True)
    for U in (U1, U1, U2):
        (mk.kraus(U, 0).abs() ** 2).sum().backward()
    assert U2.grad is not None and torch.allclose(2 * U2.grad, U1.grad)
    assert not mk._kraus_m


def test_fine_stages_are_shared_and_cleared():
    """Fine stages are cached once per (k, m, K_b, device) across kernels and dropped by clear_layers_cache."""
    from photonic_engines.slos import internal
    from photonic_engines.slos.basis import clear_layers_cache
    m = 4
    ems = [_mode_emitter(1, 2, 2, 2, (2 * j, 2 * j + 1), seed=110 + j) for j in range(2)]
    U = haar_unitary(m, 111)
    a, b = SequenceKernel(ems, m), SequenceKernel(ems, m)
    a.kraus(U, 0)
    b.kraus(U, 0)
    assert next(iter(a._stages.values())) is next(iter(b._stages.values()))
    clear_layers_cache()
    assert not internal._STAGES


def test_route_f_refusals():
    """Not yet supported combinations raise instead of computing a channel that ignores them."""
    m, n = 4, 1
    ems = [_mode_emitter(n, 2, 2, 2, (2 * j, 2 * j + 1), seed=90 + j) for j in range(2)]
    U = haar_unitary(m, 91)
    e = ems[0]
    lost = np.zeros((n, 1, 2, 2), dtype=complex)
    with_lost = ModeBinEmitter(e.ops, e.modes, e.s0, lost=lost)
    with pytest.raises(NotImplementedError, match="lost"):              # lost itself: test_internal_modes_lost
        SequenceKernel([with_lost, ems[1]], m).kraus(U, 0, compress=1e-12, route="R")
    with pytest.raises(NotImplementedError, match="order"):              # indist itself: test_internal_modes_indist
        SequenceKernel(ems, m, indist=0.9, order=1).kraus(U, 0)
    with pytest.raises(NotImplementedError, match="max_bytes"):
        SequenceKernel(ems, m, max_bytes=1000).kraus(U, 0)
    half = ModeBinEmitter(e.ops * 0.5, e.modes, e.s0)
    with pytest.raises(ValueError, match="not trace-preserving"):
        SequenceKernel([half, ems[1]], m)
    for name in ("sample", "bin_marginals", "sequence_probability"):
        with pytest.raises(NotImplementedError, match="internal modes"):
            getattr(SequenceKernel(ems, m), name)(U, 1)
